import torch
import torch.nn as nn
import pytorch3d
import pytorch3d.loss
import pytorch3d.renderer
import pytorch3d.transforms
import numpy as np
import os
from . import networks
from .renderer import *
from .utils import misc, arap, custom_loss
from .dataloaders import get_sequence_loader, get_image_loader
from .utils.skinning import skinning
import lpips
from einops import rearrange


def validate_tensor_to_device(x, device):
    if torch.any(torch.isnan(x)):
        return None
    else:
        return x.to(device)


def collapseBF(x):
    return None if x is None else rearrange(x, 'b f ... -> (b f) ...')
def expandBF(x, b, f):
    return None if x is None else rearrange(x, '(b f) ... -> b f ...', b=b, f=f)


def lookat_forward_to_rot_matrix(vec_forward, up=[0,1,0]):
    vec_forward = nn.functional.normalize(vec_forward, p=2, dim=-1)  # x right, y up, z forward
    up = torch.FloatTensor(up).to(vec_forward.device)
    vec_right = up.expand_as(vec_forward).cross(vec_forward, dim=-1)
    vec_right = nn.functional.normalize(vec_right, p=2, dim=-1)
    vec_up = vec_forward.cross(vec_right, dim=-1)
    vec_up = nn.functional.normalize(vec_up, p=2, dim=-1)
    rot_mat = torch.stack([vec_right, vec_up, vec_forward], -2)
    return rot_mat


def forward_net(net, input=None, seq_idx=None, frame_idx=None):
    if isinstance(net, networks.ParamAct):
        return net()
    elif isinstance(net, networks.LookupTable):
        return net(seq_idx, frame_idx=frame_idx)
    else:
        b, f = input.shape[:2]
        out = net(collapseBF(input))
        return expandBF(out, b=b, f=f)  # BxFx...


class PriorPredictor(nn.Module):
    def __init__(self, cfgs, num_verts_base):
        super().__init__()
        self.shape_prior_type = cfgs.get('shape_prior_type', 'offset')
        self.num_verts_base = num_verts_base
        if self.shape_prior_type == 'offset':
            self.netShapePrior = networks.ParamAct((self.num_verts_base, 3), activation=None)
        else:
            raise ValueError(f'Wrong shape prior type: {self.shape_prior_type}')

    def forward(self):
        return forward_net(self.netShapePrior)


class FramePredictor(nn.Module):
    def __init__(self, cfgs, num_verts_base, num_seqs=1, num_frames=1):
        super().__init__()
        self.in_image_size = cfgs.get('in_image_size', 128)
        self.latent_dimension = cfgs.get('latent_dimension', 256)

        ## deformation and articulation network
        self.num_body_bones = cfgs.get('num_body_bones', 4)
        self.num_legs = cfgs.get('num_legs', 0)
        self.num_leg_bones = cfgs.get('num_leg_bones', 0)
        self.num_verts_base = num_verts_base
        self.num_arti_params = self.num_body_bones * 3 + self.num_legs * self.num_leg_bones * 3
        self.netDeform = networks.Encoder(3, self.num_verts_base*3 + self.num_arti_params, in_size=self.in_image_size, zdim=self.latent_dimension, nf=32, activation=None)

        ## rigid pose (viewpoint) network
        self.cam_pos_z_offset = cfgs.get('cam_pos_z_offset', 10.)
        self.max_trans_xy_range_ratio = cfgs.get('max_trans_xy_range_ratio', 1.)
        self.max_trans_xy_range = self.max_trans_xy_range_ratio * self.cam_pos_z_offset * 0.325  # 3.25*ratio
        self.max_trans_z_range_ratio = cfgs.get('max_trans_z_range_ratio', 1.)
        self.max_trans_z_range = self.max_trans_z_range_ratio * self.cam_pos_z_offset * 0.325  # 3.25*ratio
        self.max_trans_range = torch.FloatTensor([self.max_trans_xy_range, self.max_trans_xy_range, self.max_trans_z_range])
        self.rot_rep = cfgs.get('rot_rep', 'euler_angle')
        if self.rot_rep == 'euler_angle':
            num_pose_params = 6
            self.max_rot_range = torch.FloatTensor([cfgs.get('max_rot_x_range', 180), cfgs.get('max_rot_y_range', 180), cfgs.get('max_rot_z_range', 180)])
        elif self.rot_rep == 'quaternion':
            num_pose_params = 7
        elif self.rot_rep == 'lookat':
            num_pose_params = 6
            self.lookat_init = cfgs.get('lookat_init', None)
            self.lookat_zeroy = cfgs.get('lookat_zeroy', False)
        else:
            raise NotImplementedError
        self.netPose = networks.Encoder(3, num_pose_params, in_size=self.in_image_size, zdim=self.latent_dimension, nf=32, activation=None)
        
        ## texture network
        self.tex_im_size = cfgs.get('tex_im_size', 128)
        self.netTexture = networks.EncoderDecoder(3, 3, in_size=self.in_image_size, out_size=self.tex_im_size, zdim=self.latent_dimension, nf=32, activation='sigmoid')

    def forward_posenet(self, input=None):
        pose = forward_net(self.netPose, input)  # BxFx6
        trans_pred = pose[...,-3:].tanh() * self.max_trans_range.to(pose.device)

        if self.rot_rep == 'euler_angle':
            rot_pred = pose[...,:3].tanh()
            rot_pred = rot_pred * self.max_rot_range.to(pose.device) /180 * np.pi

        elif self.rot_rep == 'quaternion':
            quat_init = torch.FloatTensor([0.01,0,0,0]).to(pose.device)
            rot_pred = pose[...,:4] + quat_init
            rot_pred = nn.functional.normalize(rot_pred, p=2, dim=-1)
            rot_pred = rot_pred * rot_pred[...,:1].sign()  # make real part non-negative

        elif self.rot_rep == 'lookat':
            self.vec_forward_raw = pose[...,:3]
            if self.lookat_init is not None:
                self.vec_forward_raw = self.vec_forward_raw + torch.FloatTensor(self.lookat_init).to(pose.device)
            if self.lookat_zeroy:
                self.vec_forward_raw = self.vec_forward_raw * torch.FloatTensor([1,0,1]).to(pose.device)
            rot_mat = lookat_forward_to_rot_matrix(self.vec_forward_raw)
            rot_pred = rot_mat.reshape(*pose.shape[:-1], -1)  # Nx(3x3) flattened into Nx9

        else:
            raise NotImplementedError

        pose = torch.cat([rot_pred, trans_pred], -1)
        return pose

    def forward(self, images=None):
        pose = self.forward_posenet(images)  # BxFx6
        texture = forward_net(self.netTexture, images)  # 0~1
        deform = forward_net(self.netDeform, images)
        arti_params = deform[..., self.num_verts_base*3:self.num_verts_base*3 + self.num_arti_params]
        arti_params = arti_params / 10.  # reduce disruption when enabled
        verts_deform = deform[..., :self.num_verts_base*3]
        verts_deform = verts_deform / 100.  # reduce disruption when enabled
        return pose, texture, verts_deform, arti_params


class Video3D:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.device = cfgs.get('device', 'cpu')

        self.num_epochs = cfgs.get('num_epochs', 10)
        self.lr = cfgs.get('lr', 1e-4)
        self.make_optimizer = lambda model, lr=self.lr, betas=(0.9, 0.999), weight_decay=0: torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=betas, weight_decay=weight_decay)
        self.use_scheduler = cfgs.get('use_scheduler', False)
        if self.use_scheduler:
            scheduler_milestone = cfgs.get('scheduler_milestone', [1,2,3,4,5])
            scheduler_gamma = cfgs.get('scheduler_gamma', 0.5)
            self.make_scheduler = lambda optim: torch.optim.lr_scheduler.MultiStepLR(optim, milestones=scheduler_milestone, gamma=scheduler_gamma)

        self.renderer = Renderer(cfgs)
        self.netFrame = FramePredictor(cfgs, num_verts_base=self.renderer.num_verts_total)
        self.enable_prior = cfgs.get('enable_prior', False)
        if self.enable_prior:
            self.netPrior = PriorPredictor(cfgs, num_verts_base=self.renderer.num_verts_total)
            self.prior_lr = cfgs.get('prior_lr', self.lr)
            self.sym_prior_shape = cfgs.get('sym_prior_shape', False)

        self.cam_pos_z_offset = cfgs.get('cam_pos_z_offset', 10.)
        self.in_image_size = cfgs.get('in_image_size', 128)
        self.out_image_size = cfgs.get('out_image_size', 128)
        self.background_mode = cfgs.get('background_mode', 'background')
        self.rot_rep = cfgs.get('rot_rep', 'euler_angle')
        self.arti_epochs = np.arange(*cfgs.get('arti_epochs', [0, self.num_epochs]))
        self.skinning_temperature = cfgs.get('skinning_temperature', 1)
        self.body_bones_type = cfgs.get('body_bones_type', 'max_distance')
        self.static_root_bones = cfgs.get('static_root_bones', False)
        self.tex_epochs = np.arange(*cfgs.get('tex_epochs', [0, self.num_epochs]))
        self.tex_im_size = cfgs.get('tex_im_size', 128)
        
        self.seqshape_epochs = np.arange(*cfgs.get('seqshape_epochs', [0, self.num_epochs]))
        self.sym_seq_shape = cfgs.get('sym_seq_shape', False)
        self.avg_seqshape_epochs = np.arange(*cfgs.get('avg_seqshape_epochs', [0, 0]))
        self.rand_avg_seqshape_prob = cfgs.get('rand_avg_seqshape_prob', 0.)
        self.mesh_regularization_mode = cfgs.get('mesh_regularization_mode', 'seq')
        self.mesh_reg_decay_epoch = cfgs.get('mesh_reg_decay_epoch', 0)
        self.mesh_reg_decay_rate = cfgs.get('mesh_reg_decay_rate', 1)
        self.avg_texture_epochs = np.arange(*cfgs.get('avg_texture_epochs', [0, 0]))
        self.rand_avg_texture_prob = cfgs.get('rand_avg_texture_prob', 0.)
        self.alternate_pose_epochs = np.arange(*cfgs.get('alternate_pose_epochs', [0, 0]))
        self.alternate_pose_iters = cfgs.get('alternate_pose_iters', 0)
        self.pose_xflip_reg_epochs = np.arange(*cfgs.get('pose_xflip_reg_epochs', [0, 0]))
        self.pose_zflip_epochs = np.arange(*cfgs.get('pose_zflip_epochs', [0, 0]))
        self.pose_zflip_rect_loss_epochs = np.arange(*cfgs.get('pose_zflip_rect_loss_epochs', [0, 0]))
        self.pose_zflip_no_other_losses = cfgs.get('pose_zflip_no_other_losses', False)

        ## perceptual loss
        if cfgs.get('perceptual_loss_weight', 0.) > 0:
            self.perceptual_loss_use_lin = cfgs.get('perceptual_loss_use_lin', True)
            self.perceptual_loss = lpips.LPIPS(net='vgg', lpips=self.perceptual_loss_use_lin)

        self.total_accum_loss = 0.
        self.all_scores = torch.Tensor()

    @staticmethod
    def get_data_loaders(cfgs, data_type='sequence', in_image_size=256, out_image_size=256, batch_size=64, num_workers=4, train_data_dir=None, val_data_dir=None, test_data_dir=None):
        train_loader = val_loader = test_loader = None
        random_shuffle_train_samples = cfgs.get('random_shuffle_train_samples', False)
        load_background = cfgs.get('background_mode', 'none') == 'background'

        if data_type == 'sequence':
            skip_beginning = cfgs.get('skip_beginning', 4)
            skip_end = cfgs.get('skip_end', 4)
            num_frames = cfgs.get('num_frames', 2)
            min_seq_len = cfgs.get('min_seq_len', 10)
            random_sample_train_frames = cfgs.get('random_sample_train_frames', False)
            get_loader = lambda is_train, **kwargs: get_sequence_loader(
                batch_size=batch_size,
                num_workers=num_workers,
                in_image_size=in_image_size,
                out_image_size=out_image_size,
                skip_beginning=skip_beginning,
                skip_end=skip_end,
                num_frames=num_frames,
                min_seq_len=min_seq_len,
                load_background=load_background,
                random_xflip=False,
                random_sample=random_sample_train_frames if is_train else False,
                dense_sample=is_train,
                shuffle=random_shuffle_train_samples if is_train else False,
                **kwargs)

        elif data_type == 'image':
            get_loader = lambda is_train, **kwargs: get_image_loader(
                batch_size=batch_size,
                num_workers=num_workers,
                in_image_size=in_image_size,
                out_image_size=out_image_size,
                load_background=load_background,
                random_xflip=False,
                shuffle=random_shuffle_train_samples if is_train else False,
                **kwargs)

        else:
            raise ValueError(f"Unexpected data type: {data_type}")

        if train_data_dir is not None:
            assert os.path.isdir(train_data_dir), f"Training data directory does not exist: {train_data_dir}"
            print(f"Loading training data from {train_data_dir}")
            train_loader = get_loader(is_train=True, data_dir=train_data_dir)

        if val_data_dir is not None:
            assert os.path.isdir(val_data_dir), f"Validation data directory does not exist: {val_data_dir}"
            print(f"Loading validation data from {val_data_dir}")
            val_loader = get_loader(is_train=False, data_dir=val_data_dir)

        if test_data_dir is not None:
            assert os.path.isdir(test_data_dir), f"Testing data directory does not exist: {test_data_dir}"
            print(f"Loading testing data from {test_data_dir}")
            test_loader = get_loader(is_train=False, data_dir=test_data_dir)

        return train_loader, val_loader, test_loader

    def load_model_state(self, cp):
        self.netFrame.load_state_dict(cp["netFrame"])
        if self.enable_prior:
            self.netPrior.load_state_dict(cp["netPrior"])

    def load_optimizer_state(self, cp):
        self.optimizerFrame.load_state_dict(cp["optimizerFrame"])
        if self.use_scheduler:
            if 'schedulerFrame' in cp:
                self.schedulerFrame.load_state_dict(cp["schedulerFrame"])
        if self.enable_prior:
            self.optimizerPrior.load_state_dict(cp["optimizerPrior"])
            if self.use_scheduler:
                if 'schedulerPrior' in cp:
                    self.schedulerPrior.load_state_dict(cp["schedulerPrior"])

    def get_model_state(self):
        state = {"netFrame": self.netFrame.state_dict()}
        if self.enable_prior:
            state["netPrior"] = self.netPrior.state_dict()
        return state

    def get_optimizer_state(self):
        state = {"optimizerFrame": self.optimizerFrame.state_dict()}
        if self.use_scheduler:
            state["schedulerFrame"] = self.schedulerFrame.state_dict()
        if self.enable_prior:
            state["optimizerPrior"] = self.optimizerPrior.state_dict()
            if self.use_scheduler:
                state["schedulerPrior"] = self.schedulerPrior.state_dict()
        return state

    def to(self, device):
        self.device = device
        self.netFrame.to(device)
        if self.enable_prior:
            self.netPrior.to(device)
        self.renderer.to(device)
        if hasattr(self, 'perceptual_loss'):
            self.perceptual_loss.to(device)

    def set_train(self):
        self.netFrame.train()
        if self.enable_prior:
            self.netPrior.train()

    def set_eval(self):
        self.netFrame.eval()
        if self.enable_prior:
            self.netPrior.eval()

    def reset_optimizers(self):
        print("Resetting optimizers...")
        self.optimizerFrame = self.make_optimizer(self.netFrame)
        if self.use_scheduler:
            self.schedulerFrame = self.make_scheduler(self.optimizerFrame)
        if self.enable_prior:
            self.optimizerPrior = self.make_optimizer(self.netPrior, lr=self.prior_lr)
            if self.use_scheduler:
                self.schedulerPrior = self.make_scheduler(self.optimizerPrior)

    def backward(self):
        self.optimizerFrame.zero_grad()
        if self.enable_prior and self.backward_prior:
            self.optimizerPrior.zero_grad()
        self.total_accum_loss.backward()
        self.optimizerFrame.step()
        if self.enable_prior and self.backward_prior:
            self.optimizerPrior.step()
        self.total_accum_loss = 0.

    def scheduler_step(self):
        if self.use_scheduler:
            self.schedulerFrame.step()
            if self.enable_prior:
                self.schedulerPrior.step()

    def compose_shape(self, prior_shape_verts_delta, seq_shape_verts_delta, arti_params):
        init_shape_verts = self.renderer.init_verts[0]
        if prior_shape_verts_delta is not None:
            prior_shape_verts = init_shape_verts + prior_shape_verts_delta  # Vx3
        else:
            prior_shape_verts = init_shape_verts  # Vx3
        
        ## seq_shape_verts_delta cannot be None (but could be 0)
        seq_shape_verts = prior_shape_verts.expand_as(seq_shape_verts_delta) + seq_shape_verts_delta  # BxFxVx3

        if arti_params is not None:
            frame_shape_verts, aux = skinning(seq_shape_verts, arti_params, self.netFrame.num_body_bones, self.netFrame.num_legs, self.netFrame.num_leg_bones, body_bones_type=self.body_bones_type, temperature=self.skinning_temperature, static_root_bones=self.static_root_bones)  # BxFxVx3
        else:
            frame_shape_verts = seq_shape_verts  # BxFxVx3
            aux = {}

        return prior_shape_verts, seq_shape_verts, frame_shape_verts, aux

    def zflip_pose(self, pose):
        if self.rot_rep == 'lookat':
            vec_forward = pose[:,:,6:9]  # BxFx3
            trans_pred = pose[:,:,9:]  # BxFx3
            vec_forward_zflip = vec_forward * torch.FloatTensor([1,1,-1]).view(1,1,3).to(pose.device)
            rot_mat = lookat_forward_to_rot_matrix(vec_forward_zflip)
            rot_pred = rot_mat.reshape(*pose.shape[:-1], -1)
            pose_zflip = torch.cat([rot_pred, trans_pred], -1)
        else:
            raise NotImplementedError
        return pose_zflip

    def predict_prior_paramters(self):
        if self.enable_prior:
            prior_shape_verts_delta = self.netPrior()
            if self.sym_prior_shape:
                prior_shape_verts_delta = self.renderer.symmetrize_shape(prior_shape_verts_delta[None, None])
            self.backward_prior = True
        else:
            prior_shape_verts_delta = None
        return prior_shape_verts_delta

    def predict_frame_paramters(self, input_image, epoch):
        batch_size, num_frames, _, h0, w0 = input_image.shape  # BxFxCxHxW
        h = w = self.out_image_size

        ## predict on all frames
        pose, texture, seq_shape_verts_delta, arti_params = self.netFrame(input_image)  # BxFx...
        seq_shape_verts_delta = seq_shape_verts_delta.view(batch_size, num_frames, -1, 3)

        if epoch not in self.seqshape_epochs:
            seq_shape_verts_delta = seq_shape_verts_delta * 0  # disable per sequence shape

        if epoch in self.avg_seqshape_epochs and self.rand_avg_seqshape_prob > 0:
            switch = torch.rand(batch_size, num_frames, 1,1).to(seq_shape_verts_delta.device)
            switch = (switch < self.rand_avg_seqshape_prob).float()
            mean_seq_shape_verts_delta = seq_shape_verts_delta.mean(1, keepdim=True).repeat(1, num_frames, 1,1)
            seq_shape_verts_delta = switch * mean_seq_shape_verts_delta + (1-switch) * seq_shape_verts_delta

        if self.sym_seq_shape:
            seq_shape_verts_delta = self.renderer.symmetrize_shape(seq_shape_verts_delta)

        if epoch not in self.arti_epochs:
            arti_params = None
        if arti_params is not None:
            arti_params = arti_params.view(batch_size, num_frames, -1, 3)

        if epoch in self.avg_texture_epochs and self.rand_avg_texture_prob > 0:
            switch = torch.rand(batch_size, num_frames, 1,1,1).to(texture.device)
            switch = (switch < self.rand_avg_texture_prob).float()
            mean_texture = texture.mean(1, keepdim=True).repeat(1, num_frames, 1,1,1)
            texture = switch * mean_texture + (1-switch) * texture

        return seq_shape_verts_delta, arti_params, pose, texture

    def render(self, pose, texture, shape, background_mode='none', image_gt=None, bg_image=None, render_flow=True):
        ## render all frames
        rendered, flow_pred, meshes_pred = self.renderer(pose, texture, shape, render_flow=render_flow)  # BxFx...
        mask_pred = rendered[..., 3]  # BxFxHxW
        image_pred = rendered[..., :3].permute(0,1,4,2,3)  # BxFxCxHxW
        if flow_pred is not None:
            flow_pred = flow_pred.permute(0,1,4,2,3)  # BxFx(x,y)xHxW, -1~1

        ## blend with a background image
        if background_mode == 'background':
            image_pred = image_pred * mask_pred.unsqueeze(2) + bg_image * (1-mask_pred.unsqueeze(2))
        ## original image as background
        elif background_mode == 'input':
            image_pred = image_pred * mask_pred.unsqueeze(2) + image_gt * (1-mask_pred.unsqueeze(2))
        return image_pred, mask_pred, flow_pred, meshes_pred

    def compute_reconstruction_losses(self, image_pred, image_gt, mask_pred, mask_gt, mask_dt, mask_valid, flow_pred, flow_gt, background_mode='none', reduce=False):
        losses = {}
        batch_size, num_frames, _, h, w = image_pred.shape  # BxFxCxHxW

        ## mask L2 loss
        mask_pred_valid = mask_pred * mask_valid
        mask_loss = (mask_pred_valid - mask_gt) ** 2
        losses['mask_loss'] = mask_loss.view(batch_size, num_frames, -1).mean(2)

        mask_pred_binary = (mask_pred_valid > 0.).float().detach()
        mask_both_binary = collapseBF(mask_pred_binary * mask_gt)  # BFxHxW
        mask_both_binary = (nn.functional.avg_pool2d(mask_both_binary.unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0.99).float().detach()  # erode by 1 pixel
        mask_both_binary = expandBF(mask_both_binary, b=batch_size, f=num_frames)  # BxFxHxW

        ## RGB L1 loss
        rgb_loss = (image_pred - image_gt).abs()
        if background_mode in ['background', 'input']:
            pass
        else:
            rgb_loss = rgb_loss * mask_both_binary.unsqueeze(2)
        losses['rgb_loss'] = rgb_loss.view(batch_size, num_frames, -1).mean(2)

        ## perceptual loss
        if self.cfgs.get('perceptual_loss_weight', 0.) > 0:
            if background_mode in ['background', 'input']:
                perc_image_pred = image_pred
                perc_image_gt = image_gt
            else:
                perc_image_pred = image_pred * mask_both_binary.unsqueeze(2) + 0.5 * (1-mask_both_binary.unsqueeze(2))
                perc_image_gt = image_gt * mask_both_binary.unsqueeze(2) + 0.5 * (1-mask_both_binary.unsqueeze(2))
            losses['perceptual_loss'] = self.perceptual_loss(collapseBF(perc_image_pred) *2-1, collapseBF(perc_image_gt) *2-1).view(batch_size, num_frames)

        ## flow loss between consecutive frames
        if flow_pred is not None:
            flow_loss = (flow_pred - flow_gt) ** 2.
            flow_loss_mask = mask_both_binary[:,:-1].unsqueeze(2).expand_as(flow_gt)

            ## ignore frames where GT flow is too large (likely inaccurate)
            large_flow = (flow_gt.abs() > 0.5).float() * flow_loss_mask
            large_flow = (large_flow.view(batch_size, num_frames-1, -1).sum(2) > 0).float()
            self.large_flow = large_flow

            flow_loss = flow_loss * flow_loss_mask * (1 - large_flow[:,:,None,None,None])
            num_mask_pixels = flow_loss_mask.reshape(batch_size, num_frames-1, -1).sum(2).clamp(min=1)
            losses['flow_loss'] = (flow_loss.reshape(batch_size, num_frames-1, -1).sum(2) / num_mask_pixels)

        if reduce:
            for k, v in losses.item():
                losses[k] = v.mean()
        return losses

    def compute_regularizers(self, texture, pose, meshes_pred, seq_shape, frame_shape, input_image, epoch, prior_shape=None, is_zori_better=None):
        losses = {}
        aux = {}
        batch_size, num_frames, _, _, _ = texture.shape  # BxFxCxHxW

        ## flipped pose regularizer
        if epoch in self.pose_xflip_reg_epochs:
            image_xflip = input_image.flip(4)
            pose_xflip = self.netFrame.forward_posenet(image_xflip)
            aux['pose_xflip'] = pose_xflip
            if self.rot_rep == 'euler_angle':
                pose_xflip_xflip = pose_xflip * torch.FloatTensor([1,-1,-1,-1,1,1]).to(pose_xflip.device)  # flip rot y & z, trans x
                losses["pose_xflip_regularizer"] = ((pose_xflip_xflip - pose) ** 2.).mean()
            elif self.rot_rep == 'quaternion':
                rot_euler = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(pose[...,:4]), convention='XYZ')
                pose_euler = torch.cat([rot_euler, pose[...,4:]], -1)
                rot_xflip_euler = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(pose_xflip[...,:4]), convention='XYZ')
                pose_xflip_euler = torch.cat([rot_xflip_euler, pose_xflip[...,4:]], -1)
                pose_xflip_euler_xflip = pose_xflip_euler * torch.FloatTensor([1,-1,-1,-1,1,1]).to(pose_xflip.device)  # flip rot y & z, trans x
                losses["pose_xflip_regularizer"] = ((pose_xflip_euler_xflip - pose_euler) ** 2.).mean()
            elif self.rot_rep == 'lookat':
                rot_mat = pose[...,:9].view(*pose.shape[:-1], 3, 3)
                vec_forward = rot_mat[:, :, 2, :]
                vec_forward_trans = torch.cat([vec_forward, pose[...,9:]], -1)
                rot_mat_xflip = pose_xflip[...,:9].view(*pose_xflip.shape[:-1], 3, 3)
                vec_forward_xflip = rot_mat_xflip[:, :, 2, :]
                vec_forward_trans_xflip = torch.cat([vec_forward_xflip, pose_xflip[...,9:]], -1)
                vec_forward_trans_xflip_xflip = vec_forward_trans_xflip * torch.FloatTensor([-1,1,1,-1,1,1]).to(pose_xflip.device)  # flip forward x, trans x
                pose_xflip_regularizer = ((vec_forward_trans_xflip_xflip - vec_forward_trans)[...,0] ** 2.)  # compute reg on forward x is sufficient
                if epoch in self.pose_zflip_rect_loss_epochs and self.pose_zflip_no_other_losses:
                    pose_xflip_regularizer = pose_xflip_regularizer.mean(1) * is_zori_better
                losses["pose_xflip_regularizer"] = pose_xflip_regularizer.mean()

        ## mesh regularizers
        if (self.mesh_regularization_mode is None) or (self.mesh_regularization_mode == 'none'):
            pass
        else:
            if self.mesh_regularization_mode == 'frame':
                reg_mesh = meshes_pred
            elif self.mesh_regularization_mode == 'seq':
                reg_mesh = self.renderer.get_deformed_mesh(seq_shape)
            else:
                raise NotImplementedError
            mesh_laplacian_method = self.cfgs.get('mesh_laplacian_method', 'uniform')
            losses["mesh_laplacian_smoothing"] = custom_loss.mesh_laplacian_smoothing(reg_mesh, method=mesh_laplacian_method)
            losses["mesh_normal_consistency"] = pytorch3d.loss.mesh_normal_consistency(reg_mesh)

        ## arap between prior shape and sequence shapes
        if self.enable_prior and (epoch in self.seqshape_epochs) and (self.cfgs.get('arap_prior_seq_regularizer_weight', 0.0) > 0):
            edges = meshes_pred[0].edges_packed().detach()
            loss = arap.arap_loss(prior_shape.expand(batch_size,num_frames,-1,-1), seq_shape, edges)
            if epoch in self.pose_zflip_rect_loss_epochs and self.pose_zflip_no_other_losses:
                loss = loss.mean(1) * is_zori_better
            losses["arap_prior_seq_regularizer"] = loss.mean()

        ## arap between sequence shape and frame shapes
        if (epoch in self.arti_epochs) and (num_frames > 1) and (self.cfgs.get('arap_seq_frame_regularizer_weight', 0.0) > 0):
            edges = meshes_pred[0].edges_packed().detach()
            loss = arap.arap_loss(seq_shape, frame_shape, edges)
            if epoch in self.pose_zflip_rect_loss_epochs and self.pose_zflip_no_other_losses:
                loss = loss.mean(1) * is_zori_better
            losses["arap_seq_frame_regularizer"] = loss.mean()

        return losses, aux

    def forward(self, batch, epoch, logger=None, total_iter=None, save_results=False, save_dir=None, logger_prefix=''):
        input_image, mask_gt, mask_dt, mask_valid, flow_gt, bbox, bg_image, seq_idx, frame_idx = (*map(lambda x: validate_tensor_to_device(x, self.device), batch),)
        seq_idx = seq_idx.squeeze(1)  # Bx1 -> B
        global_frame_id, crop_x0, crop_y0, crop_w, crop_h, full_w, full_h, sharpness = bbox.unbind(2)  # BxFx7
        bbox = torch.stack([crop_x0, crop_y0, crop_w, crop_h], 2)
        mask_gt = (mask_gt[:, :, 0, :, :] > 0.9).float()  # BxFxHxW
        mask_dt = mask_dt / self.in_image_size
        batch_size, num_frames, _, h0, w0 = input_image.shape  # BxFxCxHxW
        h = w = self.out_image_size
        aux_viz = {}

        if self.cfgs.get('flow_loss_weight', 0.) > 0.:
            render_flow = True
        else:
            render_flow = False

        ## GT image
        image_gt = input_image
        if self.out_image_size != self.in_image_size:
            image_gt = expandBF(torch.nn.functional.interpolate(collapseBF(image_gt), size=[h, w], mode='bilinear'), b=batch_size, f=num_frames)
        
        ## 1st pose hypothesis with original predictions
        prior_shape_verts_delta = self.predict_prior_paramters()
        seq_shape_verts_delta, arti_params, pose, texture = self.predict_frame_paramters(input_image, epoch)

        ## train pose only alternately during alternate_pose_epochs
        self.train_pose_only = False
        if epoch in self.alternate_pose_epochs:
            if (total_iter // self.alternate_pose_iters) % 2 == 0:
                self.train_pose_only = True

        ## train pose only for N iters when pose xflip rectification is intially enabled
        if len(self.pose_xflip_reg_epochs) > 0 and epoch in self.pose_xflip_reg_epochs and self.pose_xflip_reg_epochs[0] > 0:
            no_xflip_loss_iters = self.pose_xflip_reg_epochs[0] * len(self.trainer.train_loader)
            if (total_iter - no_xflip_loss_iters) < 10000:
                self.train_pose_only = True

        ## train pose only for N iters when pose zflip rectification is intially enabled
        if len(self.pose_zflip_rect_loss_epochs) > 0 and epoch in self.pose_zflip_rect_loss_epochs and self.pose_zflip_rect_loss_epochs[0] > 0:
            no_zflip_loss_iters = self.pose_zflip_rect_loss_epochs[0] * len(self.trainer.train_loader)
            if (total_iter - no_zflip_loss_iters) < 10000:
                self.train_pose_only = True

        ## disable gradients of other components when training pose only
        if self.train_pose_only:
            texture = texture.detach()
            seq_shape_verts_delta = seq_shape_verts_delta.detach()
            if arti_params is not None:
                arti_params = arti_params.detach()
            if self.enable_prior:
                self.backward_prior = False  # disable backward step to avoid accumulating zero grad with weight decay
                prior_shape_verts_delta = prior_shape_verts_delta.detach()

        ## compose shape
        prior_shape, seq_shape, frame_shape, shape_aux = self.compose_shape(prior_shape_verts_delta, seq_shape_verts_delta, arti_params)

        ## rendering
        image_pred, mask_pred, flow_pred, meshes_pred = self.render(pose, texture, frame_shape, background_mode=self.background_mode, image_gt=image_gt, bg_image=bg_image, render_flow=render_flow)

        ## compute reconstruction losses
        losses_zori = self.compute_reconstruction_losses(image_pred, image_gt, mask_pred, mask_gt, mask_dt, mask_valid, flow_pred, flow_gt, background_mode=self.background_mode, reduce=False)

        final_losses = {}
        ## 2nd pose hypothesis mirrored along z-axis
        if epoch in self.pose_zflip_epochs:
            pose_zflip = self.zflip_pose(pose)
            image_pred_zflip, mask_pred_zflip, flow_pred_zflip, meshes_pred_zflip = self.render(pose_zflip, texture, frame_shape, background_mode=self.background_mode, image_gt=image_gt, bg_image=bg_image, render_flow=render_flow)
            losses_zflip = self.compute_reconstruction_losses(image_pred_zflip, image_gt, mask_pred_zflip, mask_gt, mask_dt, mask_valid, flow_pred_zflip, flow_gt, background_mode=self.background_mode, reduce=False)

            ## select the better pose
            total_losses = []
            for losses in [losses_zori, losses_zflip]:
                total_loss = 0
                for name, loss in losses.items():
                    if name in ['mask_loss', 'flow_loss']:
                        loss_weight = self.cfgs.get(f"{name}_weight", 0.)  # look up loss weight in config file
                        if loss_weight > 0:
                            total_loss += loss.mean(1) * loss_weight
                total_losses += [total_loss]

            is_zflip_better = (total_losses[1] < total_losses[0]).float().detach()
            is_zori_better = 1 - is_zflip_better

            ## use the loss from the better pose
            for name in losses_zori.keys():
                ## disable other losses (keep only rect loss) if zflip is better
                if epoch in self.pose_zflip_rect_loss_epochs and self.pose_zflip_no_other_losses:
                    loss = losses_zori[name].mean(1) * is_zori_better
                else:
                    loss = losses_zflip[name].mean(1) * is_zflip_better + losses_zori[name].mean(1) * is_zori_better
                final_losses[name] = loss.mean()

            final_pose = pose_zflip * is_zflip_better.view(-1,1,1) + pose * is_zori_better.view(-1,1,1)
            pose_zflip_rect_loss = ((pose - final_pose.detach())[:,:,8]**2.).mean(1) * is_zflip_better  # only the z-axis lookat
            final_losses['pose_zflip_rect_loss'] = pose_zflip_rect_loss.mean()

        else:
            is_zori_better = torch.ones(batch_size).to(input_image.device)
            for name, loss in losses_zori.items():
                final_losses[name] = loss.mean()

        ## regularizers
        regularizers, aux = self.compute_regularizers(texture, pose, meshes_pred, seq_shape, frame_shape, input_image, epoch, prior_shape=prior_shape, is_zori_better=is_zori_better)
        final_losses.update(regularizers)
        aux_viz.update(aux)

        total_loss = 0
        for name, loss in final_losses.items():
            if self.train_pose_only:
                if name not in ['mask_loss', 'flow_loss', 'pose_xflip_regularizer', 'pose_zflip_rect_loss']:
                    continue
            if epoch not in self.tex_epochs:
                if name in ['rgb_loss', 'perceptual_loss']:
                    continue
            if epoch not in self.pose_zflip_rect_loss_epochs:
                if name in ['pose_zflip_rect_loss']:
                    continue

            loss_weight = self.cfgs.get(f"{name}_weight", 0.)  # look up loss weight in config file

            if (name in ['mesh_laplacian_smoothing', 'mesh_normal_consistency']) and epoch >= self.mesh_reg_decay_epoch:
                decay_rate = self.mesh_reg_decay_rate ** (epoch - self.mesh_reg_decay_epoch)
                loss_weight = max(loss_weight * decay_rate, self.cfgs.get(f"{name}_min_weight", 0.))
            
            if loss_weight > 0.:
                total_loss += loss * loss_weight
        self.total_accum_loss += total_loss  # reset to 0 in backward step

        if torch.isnan(self.total_accum_loss):
            print("NaN in loss...")
            import pdb; pdb.set_trace()

        metrics = {'loss': total_loss, **final_losses}

        ## use results from better hypothesis
        if epoch in self.pose_zflip_epochs:
            ## put a small red marker on top left corner of the rendered image with z-flipped pose
            image_pred_zflip[:,:,0:1,:8,:8] = 1.
            image_pred_zflip[:,:,1:3,:8,:8] = 0.
            image_pred = image_pred_zflip * is_zflip_better.view(-1,1,1,1,1) + image_pred * is_zori_better.view(-1,1,1,1,1)
            mask_pred = mask_pred_zflip * is_zflip_better.view(-1,1,1,1) + mask_pred * is_zori_better.view(-1,1,1,1)
            if flow_pred is not None:
                flow_pred = flow_pred_zflip * is_zflip_better.view(-1,1,1,1,1) + flow_pred * is_zori_better.view(-1,1,1,1,1)
            pose_original = pose
            pose = final_pose

        ## log visuals
        if logger is not None:
            b0 = max(min(batch_size, 16//num_frames), 1)
            def log_image(name, image):
                logger.add_image(logger_prefix+'image/'+name, misc.image_grid(collapseBF(image[:b0,:]).detach().cpu().clamp(0,1)), total_iter)
            
            log_image('image_gt', input_image)
            log_image('image_pred', image_pred)
            log_image('mask_gt', mask_gt.unsqueeze(2).repeat(1,1,3,1,1))
            log_image('mask_pred', mask_pred.unsqueeze(2).repeat(1,1,3,1,1))
            log_image('texture_pred', texture)

            if flow_gt is not None:
                flow_gt_viz = torch.nn.functional.pad(flow_gt, pad=[0, 0, 0, 0, 0, 1])  # add a dummy channel for visualization
                flow_gt_viz = flow_gt_viz + 0.5  # -0.5~1.5
                flow_gt_viz = torch.nn.functional.pad(flow_gt_viz, pad=[0, 0, 0, 0, 0, 0, 0, 1])  # add a dummy frame for visualization

                ## draw marker on large flow frames
                large_flow_marker_mask = torch.zeros_like(flow_gt_viz)
                large_flow_marker_mask[:,:,:,:8,:8] = 1.
                large_flow = torch.cat([self.large_flow, self.large_flow[:,:1] *0.], 1)
                large_flow_marker_mask = large_flow_marker_mask * large_flow[:,:,None,None,None]
                red = torch.FloatTensor([1,0,0]).view(1,1,3,1,1).to(flow_gt_viz.device)
                flow_gt_viz = large_flow_marker_mask * red + (1-large_flow_marker_mask) * flow_gt_viz

                log_image('flow_gt', flow_gt_viz)
            
            if flow_pred is not None:
                flow_pred_viz = torch.nn.functional.pad(flow_pred, pad=[0, 0, 0, 0, 0, 1])  # add a dummy channel for visualization
                flow_pred_viz = flow_pred_viz + 0.5  # -0.5~1.5
                flow_pred_viz = torch.nn.functional.pad(flow_pred_viz, pad=[0, 0, 0, 0, 0, 0, 0, 1])  # add a dummy frame for visualization
                log_image('flow_pred', flow_pred_viz)

            ## draw a red seam on the texture visualizing the symmetry plane
            seam_mask = self.renderer.tex_map_seam_mask.unsqueeze(0)
            seam_mask = nn.functional.interpolate(seam_mask, texture.shape[3:], mode='bilinear').unsqueeze(0)
            red = torch.FloatTensor([1,0,0]).view(1,1,3,1,1).to(texture.device)
            tex_im_with_seam = seam_mask * red + (1-seam_mask) * texture

            ## render rotation animation
            cam_dist = self.cfgs.get('cam_pos_z_offset', 10.) - pose[...,-1].mean().cpu().numpy()  # place the object at center
            logger.add_video(logger_prefix+'animation/rotation', self.render_rotation_frames(frame_shape[:1,:1], tex_im=tex_im_with_seam[:1,:1], num_frames=36, cam_dist=cam_dist).detach().cpu().unsqueeze(0), total_iter, fps=2)

            ## render sequence base shape rotation
            if arti_params is not None:
                logger.add_video(logger_prefix+'animation/seqshape_rotation', self.render_rotation_frames(seq_shape[:1,:1], tex_im=tex_im_with_seam[:1,:1], num_frames=36, cam_dist=cam_dist).detach().cpu().unsqueeze(0), total_iter, fps=2)

            ## render deformation animation within a sequence
            if num_frames > 1:
                logger.add_video(logger_prefix+'animation/deformation', misc.video_grid(self.render_deformation_frames(frame_shape[:1], tex_im=tex_im_with_seam[:1], cam_dist=cam_dist)).detach().cpu().unsqueeze(0), total_iter, fps=2)

            ## render with triangle texture maps
            if self.renderer.face_triangle_tex_map is not None:
                face_triangle_tex_map = self.renderer.face_triangle_tex_map.repeat(batch_size, num_frames, 1,1,1).to(frame_shape.device)
                # meshes_pred.textures = self.renderer.get_textures(face_triangle_tex_map)  # cuda illegal memory access error
                new_mesh = self.renderer.get_deformed_mesh(frame_shape[:b0], pose=pose[:b0])
                new_mesh.textures = self.renderer.get_textures(face_triangle_tex_map[:b0])
                face_triangle_rendered = self.renderer.image_renderer(meshes_world=new_mesh, cameras=self.renderer.cameras).detach().cpu().clamp(0,1).permute(0, 3, 1, 2)
                face_triangle_rendered = expandBF(face_triangle_rendered, b=b0, f=num_frames)
                logger.add_image(logger_prefix+'image/face_triangle_rendered', misc.image_grid(collapseBF(face_triangle_rendered)), total_iter)

                if self.enable_prior:
                    viz_shape = prior_shape.expand(1,1,-1,-1)
                    viz_shape_type = 'prior_'
                else:
                    viz_shape = frame_shape[:1,:1]
                    viz_shape_type = 'frame_'
                logger.add_video(logger_prefix+'animation/'+viz_shape_type+'face_triangle_rotation', self.render_rotation_frames(viz_shape, face_triangle_tex_map[:1,:1], num_frames=36, cam_dist=cam_dist).unsqueeze(0), total_iter, fps=2)

            ## log pose distributions
            if self.rot_rep == 'euler_angle':
                for i, name in enumerate(['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']):
                    logger.add_histogram(logger_prefix+'pose/'+name, pose[...,i], total_iter)
            elif self.rot_rep == 'quaternion':
                for i, name in enumerate(['qt_0', 'qt_1', 'qt_2', 'qt_3', 'trans_x', 'trans_y', 'trans_z']):
                    logger.add_histogram(logger_prefix+'pose/'+name, pose[...,i], total_iter)
                rot_euler = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(pose.detach().cpu()[...,:4]), convention='XYZ')
                for i, name in enumerate(['rot_x', 'rot_y', 'rot_z']):
                    logger.add_histogram(logger_prefix+'pose/'+name, rot_euler[...,i], total_iter)
            elif self.rot_rep == 'lookat':
                for i, name in enumerate(['fwd_x', 'fwd_y', 'fwd_z']):
                    logger.add_histogram(logger_prefix+'pose/'+name, pose[...,6+i], total_iter)
                for i, name in enumerate(['trans_x', 'trans_y', 'trans_z']):
                    logger.add_histogram(logger_prefix+'pose/'+name, pose[...,9+i], total_iter)
                if epoch in self.pose_zflip_epochs:
                    for i, name in enumerate(['fwd_x', 'fwd_y', 'fwd_z']):
                        logger.add_histogram(logger_prefix+'pose_original/'+name, pose_original[...,6+i], total_iter)

            if epoch in self.pose_zflip_epochs:
                logger.add_scalar(logger_prefix+'rot_zflip/is_zflip_better', is_zflip_better.mean(), total_iter)

            if epoch in self.pose_xflip_reg_epochs:
                pose_xflip = aux_viz['pose_xflip']
                if self.rot_rep == 'euler_angle':
                    for i, name in enumerate(['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']):
                        logger.add_histogram(logger_prefix+'pose_xflip/'+name, pose_xflip[...,i], total_iter)
                elif self.rot_rep == 'quaternion':
                    for i, name in enumerate(['qt_0', 'qt_1', 'qt_2', 'qt_3', 'trans_x', 'trans_y', 'trans_z']):
                        logger.add_histogram(logger_prefix+'pose_xflip/'+name, pose_xflip[...,i], total_iter)
                    rot_euler = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(pose_xflip.detach().cpu()[...,:4]), convention='XYZ')
                    for i, name in enumerate(['rot_x', 'rot_y', 'rot_z']):
                        logger.add_histogram(logger_prefix+'pose_xflip/'+name, rot_euler[...,i], total_iter)
                elif self.rot_rep == 'lookat':
                    for i, name in enumerate(['fwd_x', 'fwd_y', 'fwd_z']):
                        logger.add_histogram(logger_prefix+'pose_xflip/'+name, pose_xflip[...,6+i], total_iter)
                    for i, name in enumerate(['trans_x', 'trans_y', 'trans_z']):
                        logger.add_histogram(logger_prefix+'pose_xflip/'+name, pose_xflip[...,9+i], total_iter)

        if save_results:
            save_dirs = []
            fnames = []
            for i, si in enumerate(seq_idx):
                save_dirs += [os.path.join(save_dir, f'{si:05d}')] * num_frames
                fnames += [f'{fidx:05d}_{int(float(fid)):07d}' for (fidx, fid) in zip(frame_idx[i], global_frame_id[i])]
            
            def write_results(suffix, data, type, collapse=True):
                if collapse:
                    data = collapseBF(data)
                data = data.detach().cpu()
                if type == 'image':
                    misc.save_images(save_dirs, data.clamp(0,1).numpy(), suffix=suffix, fnames=fnames)
                elif type == 'txt':
                    misc.save_txt(save_dirs, data.numpy(), suffix=suffix, fnames=fnames)
                elif type == 'obj':
                    misc.save_obj(save_dirs, data, suffix=suffix, fnames=fnames)
                else:
                    raise NotImplementedError

            write_results('input_image', image_gt, type='image')
            write_results('recon_image', image_pred, type='image')
            write_results('mask_gt', mask_gt.unsqueeze(2).repeat(1,1,3,1,1), type='image')
            write_results('mask_pred', mask_pred.unsqueeze(2).repeat(1,1,3,1,1), type='image')
            write_results('texture_pred', texture, type='image')
            if flow_gt is not None:
                flow_gt_viz = torch.cat([flow_gt, torch.zeros_like(flow_gt[:,:,:1])], 2) + 0.5  # -0.5~1.5
                flow_gt_viz = torch.nn.functional.pad(flow_gt_viz, pad=[0, 0, 0, 0, 0, 0, 0, 1])
                write_results('flow_gt', flow_gt_viz, type='image')
            if flow_pred is not None:
                flow_pred_viz = torch.cat([flow_pred, torch.zeros_like(flow_pred[:,:,:1])], 2) + 0.5  # -0.5~1.5
                flow_pred_viz = torch.nn.functional.pad(flow_pred_viz, pad=[0, 0, 0, 0, 0, 0, 0, 1])
                write_results('flow_pred', flow_pred_viz, type='image')

            mesh = self.renderer.get_deformed_mesh(frame_shape, pose=None)
            mesh.textures = self.renderer.get_textures(texture)
            write_results('mesh', mesh, type='obj', collapse=False)
            side_view = self.render_rotation_frame(mesh.detach(), num_frames=360, i=85, cam_dist=15., image_size=self.out_image_size).permute(0,3,1,2)
            write_results('side_view', side_view, type='image', collapse=False)
            write_results('prior_shape', prior_shape.expand(batch_size, num_frames,-1,-1), type='txt')
            write_results('seq_shape', seq_shape, type='txt')
            write_results('frame_shape', frame_shape, type='txt')
            if arti_params is not None:
                write_results('arti_params', arti_params, type='txt')
            write_results('pose', pose, type='txt')

            ## compute scores
            scores = misc.compute_all_metrics(mask_gt, mask_pred, image_gt, image_pred, mask_threshold=0.1)  # mask_mse, mask_iou, image_mse, flow_mse
            scores = (*map(lambda x: x if x is not None else torch.zeros_like(scores[0]), scores),)  # convert nan to 0, assuming mask_mse exists
            scores = torch.stack(scores, -1).detach().cpu()
            write_results('metrics', scores, type='txt')
            self.all_scores = torch.cat([self.all_scores, scores.view(-1, scores.shape[-1])], 0)

        return metrics

    def save_scores(self, path):
        header = 'mask_mse, \
                  mask_iou, \
                  image_mse'
        mean = self.all_scores.mean(0)
        std = self.all_scores.std(0)
        header = header + '\nMean: ' + ',\t'.join(['%.8f'%x for x in mean])
        header = header + '\nStd: ' + ',\t'.join(['%.8f'%x for x in std])
        misc.save_scores(path, self.all_scores, header=header)
        print(header)

    def render_rotation_frame(self, mesh, num_frames, i, cam_dist=None, image_size=256):
        # at = torch.from_numpy(np.array([[0, 0, 0]], dtype=np.float32)).to(self.device)
        mesh_centroid = mesh.verts_padded().mean(1, keepdim=True)
        centered_mesh = mesh.update_padded(mesh.verts_padded() - mesh_centroid)
        at = torch.from_numpy(np.array([[0, 0, 0]], dtype=np.float32)).to(self.device)
        cam = pytorch3d.renderer.FoVPerspectiveCameras(device=self.device, fov=self.cfgs['fov'])
        angle = 2 * np.pi / num_frames * i
        if cam_dist is None:
            cam_dist = self.cfgs.get('cam_pos_z_offset', 10.)
        pos = np.array([[np.sin(angle) * cam_dist, 0, np.cos(angle) * cam_dist]], dtype=np.float32)
        update_camera_pose(cam, torch.from_numpy(pos).to(self.device), at)
        # settings, blend_params = get_soft_rasterizer_settings(image_size=256)
        settings, blend_params = get_soft_rasterizer_settings(image_size=image_size, sigma=0, gamma=1e-5, faces_per_pixel=1)
        return self.renderer.image_renderer(meshes_world=centered_mesh, cameras=cam, raster_settings=settings, blend_params=blend_params).clamp(0,1)  # BxHxWxC

    def render_rotation_frames(self, shape, tex_im, num_frames=36, cam_dist=None):
        frames = []
        mesh = self.renderer.get_deformed_mesh(shape, pose=None)
        mesh.textures = self.renderer.get_textures(tex_im)
        for i in range(num_frames):
            # frame = self.render_rotation_frame(mesh[i % b], num_frames, i)
            frame = self.render_rotation_frame(mesh, num_frames, i, cam_dist=cam_dist)
            frame = misc.image_grid(frame.permute(0, 3, 1, 2))
            frames += [frame]
        return torch.stack(frames, 0)  # TxCxHxW

    def render_deformation_frames(self, shape, tex_im, cam_dist=None):
        b, f, _, _ = shape.shape
        mesh = self.renderer.get_deformed_mesh(shape, pose=None)
        mesh.textures = self.renderer.get_textures(tex_im)
        cam = pytorch3d.renderer.FoVPerspectiveCameras(device=self.device, fov=self.cfgs['fov'])
        at = torch.from_numpy(np.array([[0, 0, 0]], dtype=np.float32)).to(self.device)
        angle = 30 / 180 * np.pi
        pos = np.array([[np.sin(angle) * cam_dist, 0, np.cos(angle) * cam_dist]], dtype=np.float32)
        update_camera_pose(cam, torch.from_numpy(pos).to(self.device), at)
        frames = self.renderer.image_renderer(meshes_world=mesh, cameras=cam).clamp(0,1).permute(0, 3, 1, 2)
        return expandBF(frames, b=b, f=f)  # BxFxCxHxW

    def render_full_frames(self, mesh, cameras):
        settings, blend_params = get_soft_rasterizer_settings(image_size=256, sigma=0, gamma=1e-5, faces_per_pixel=1)
        return self.renderer.image_renderer(meshes_world=mesh, cameras=cameras, raster_settings=settings, blend_params=blend_params).clamp(0,1).permute(0, 3, 1, 2)
