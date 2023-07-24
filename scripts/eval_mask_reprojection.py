import sys
import os
import os.path as osp
from glob import glob
import argparse
import numpy as np
import cv2
import torch

import pytorch3d
import pytorch3d.loss
import pytorch3d.renderer
import pytorch3d.structures

sys.path.insert(0, '..')
from video3d.utils.sphere import get_symmetric_ico_sphere
from video3d.utils.skinning import skinning


def update_camera_pose(cameras, position, at):
    cameras.R = pytorch3d.renderer.look_at_rotation(position, at).to(cameras.device)
    cameras.T = -torch.bmm(cameras.R.transpose(1, 2), position[:, :, None])[:, :, 0]


def get_soft_rasterizer_settings(image_size, sigma=0, gamma=1e-6, faces_per_pixel=30):
    blend_params = pytorch3d.renderer.BlendParams(sigma=sigma, gamma=gamma)
    settings = pytorch3d.renderer.RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=faces_per_pixel,
    )
    return settings, blend_params


def get_light(amb=1., diff=0., spec=0., dir=(0,1,0)):
    return pytorch3d.renderer.DirectionalLights(ambient_color=((amb,)*3,),
                                                  diffuse_color=((diff,)*3,),
                                                  specular_color=((spec,)*3,),
                                                  direction=(dir,))


def create_image_renderer(image_size, fov=25, device='cpu'):
    settings, blend_params = get_soft_rasterizer_settings(image_size)
    lights = get_light().to(device)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=fov)
    return pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(cameras=cameras, raster_settings=settings),
        shader=pytorch3d.renderer.SoftPhongShader(lights=lights, cameras=cameras, blend_params=blend_params, device=device)
    )


def transform_verts(verts, pose, rot_rep='lookat'):
    f, _ = pose.shape
    if rot_rep == 'euler_angle':
        rot_mat = pytorch3d.transforms.euler_angles_to_matrix(pose[...,:3].view(-1,3), convention='XYZ')
        tsf = pytorch3d.transforms.Rotate(rot_mat, device=pose.device)
    elif rot_rep == 'quaternion':
        rot_mat = pytorch3d.transforms.quaternion_to_matrix(pose[...,:4].view(-1,4))
        tsf = pytorch3d.transforms.Rotate(rot_mat, device=pose.device)
    elif rot_rep == 'lookat':
        rot_mat = pose[...,:9].view(-1,3,3)
        tsf = pytorch3d.transforms.Rotate(rot_mat, device=pose.device)
    else:
        raise NotImplementedError
    tsf = tsf.compose(pytorch3d.transforms.Translate(pose[...,-3:].view(-1,3), device=pose.device))
    new_verts = tsf.transform_points(verts)
    return new_verts


def load_file(file_path):
    if file_path.endswith('.txt'):
        return torch.FloatTensor(np.loadtxt(file_path, delimiter=', '))
    elif file_path.endswith('.png'):
        return torch.FloatTensor(cv2.imread(file_path)).flip(2) / 255.
    else:
        raise NotImplementedError


def load_seq_files(seq_dir, sfx):
    return torch.stack([load_file(f) for f in sorted(glob(osp.join(seq_dir, '*'+sfx)))], 0)


def main(args):
    root_dir = 'results/bird/final_bird'
    res_dir = osp.join(root_dir, 'test_results_016')
    eval_dir = osp.join(root_dir, 'test_results_016_mask_reproj')
    frame_deltas = [5, 10, 20, 30, 60, 90]
    allow_deforms = [True, False]
    runs = [(delta, deform) for delta in frame_deltas for deform in allow_deforms]

    device = 'cuda:0'
    image_size = 128
    fov = 25
    cam_pos_z_offset = 10
    cat_type = 'bird'  # bird, horse
    skinning_temperature = 0.1
    num_body_bones = 6
    if cat_type == 'bird':
        num_legs = 0
        num_leg_bones = 0
        body_bones_type = 'max_distance'
        static_root_bones = True
    elif cat_type == 'horse':
        num_legs = 4
        num_leg_bones = 4
        body_bones_type = 'z_minmax_y+'
        static_root_bones = False
    else:
        raise NotImplementedError

    meshes, mesh_aux = get_symmetric_ico_sphere(subdiv=3, return_tex_uv=True, return_face_triangle_tex_map=True, device=device)
    renderer = create_image_renderer(image_size, fov=fov, device=device)
    camera = pytorch3d.renderer.FoVPerspectiveCameras(fov=fov).to(device)
    cam_pos = torch.FloatTensor([[0, 0, cam_pos_z_offset]]).to(device)
    cam_at = torch.FloatTensor([[0, 0, 0]]).to(device)
    update_camera_pose(camera, position=cam_pos, at=cam_at)

    for frame_delta, allow_deform in runs:
        out_folder = 'mask_reproj_%02d' %frame_delta
        out_folder = out_folder + '_deform' if allow_deform else out_folder
        out_dir = osp.join(eval_dir, out_folder)
        os.makedirs(out_dir)

        all_scores = torch.Tensor()
        seq_folder_list = [d for d in sorted(os.listdir(res_dir)) if osp.isdir(osp.join(res_dir, d))]
        for seq_folder in seq_folder_list:
            sfxs = ['prior_shape.txt', 'seq_shape.txt', 'frame_shape.txt', 'pose.txt', 'arti_params.txt', 'mask_gt.png']
            (prior_shape, seq_shape, frame_shape, pose, arti_params, mask_gt) = (load_seq_files(osp.join(res_dir, seq_folder), sfx).to(device) for sfx in sfxs)
            mask_gt = mask_gt[..., 0]

            if len(seq_shape) <= frame_delta:
                continue

            if allow_deform:
                frame_shape, mesh_aux = skinning(seq_shape=seq_shape[:-frame_delta].unsqueeze(0), arti_params=arti_params[frame_delta:].unsqueeze(0), n_bones=num_body_bones, n_legs=num_legs, n_leg_bones=num_leg_bones, body_bones_type=body_bones_type, output_posed_bones=False, temperature=skinning_temperature, static_root_bones=static_root_bones)
                frame_shape = frame_shape[0]  # 1xT -> T
                posed_shape = transform_verts(frame_shape, pose[frame_delta:])
            else:
                posed_shape = transform_verts(frame_shape[:-frame_delta], pose[frame_delta:])
            mask_gt = mask_gt[frame_delta:]

            num_frames = posed_shape.size(0)
            frame_mesh = meshes.extend(num_frames)
            frame_mesh = frame_mesh.update_padded(posed_shape)

            rendered = []
            batch_size = 100
            for start_idx in range(0, num_frames, batch_size):
                end_idx = min(start_idx+batch_size, num_frames)
                mesh_slice = frame_mesh[start_idx:end_idx]
                verts = mesh_slice.verts_padded()
                color = 0.5
                mesh_slice.textures = pytorch3d.renderer.TexturesVertex(verts_features=verts*0+color)
                rendered += [renderer(meshes_world=mesh_slice, cameras=camera)]
            rendered = torch.cat(rendered, 0)
            mask_pred = rendered[...,3]
            mask_pred = (mask_pred > 0.1).float()
            mask_intersection = mask_gt * mask_pred
            mask_union = mask_gt + mask_pred - mask_intersection
            mask_iou = mask_intersection.view(num_frames, -1).sum(1) / mask_union.view(num_frames, -1).sum(1)

            scores = mask_iou.unsqueeze(1).cpu()
            mean = scores.mean(0)
            std = scores.std(0)
            header = 'mask_mse'
            header = header + '\nMean: ' + ',\t'.join(['%.8f'%x for x in mean])
            header = header + '\nStd: ' + ',\t'.join(['%.8f'%x for x in std])
            score_fpath = osp.join(out_dir, seq_folder+'_'+out_folder+'.txt')
            np.savetxt(score_fpath, scores, fmt='%.8f', delimiter=',\t', header=header)
            all_scores = torch.cat([all_scores, scores], 0)
            print("S%s: %.4f +- %.4f" %(seq_folder, mean[0], std[0]))

        mean = all_scores.mean(0)
        std = all_scores.std(0)
        header = 'mask_mse'
        header = header + '\nMean: ' + ',\t'.join(['%.8f'%x for x in mean])
        header = header + '\nStd: ' + ',\t'.join(['%.8f'%x for x in std])
        score_fpath = osp.join(out_dir, 'all_scores.txt')
        np.savetxt(score_fpath, all_scores, fmt='%.8f', delimiter=',\t', header=header)
        print('Saving scores to %s' %score_fpath)
        print("All: %.4f +- %.4f" %(mean[0], std[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    with torch.no_grad():
        main(args)
