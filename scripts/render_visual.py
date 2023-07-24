import sys
import os
import os.path as osp
import argparse
import copy
from glob import glob
import numpy as np
import cv2
import torch
from einops.einops import rearrange
from moviepy.editor import ImageSequenceClip

import pytorch3d
import pytorch3d.loss
import pytorch3d.renderer
import pytorch3d.structures
import pytorch3d.io

sys.path.insert(0, '..')
from video3d.utils.sphere import get_symmetric_ico_sphere
from video3d.utils.skinning import skinning


def load_file(file_path):
    if file_path.endswith('.txt'):
        return torch.FloatTensor(np.loadtxt(file_path, delimiter=', '))
    elif file_path.endswith('.png'):
        return torch.FloatTensor(cv2.imread(file_path)).flip(2) / 255.
    else:
        raise NotImplementedError


def load_seq_files(seq_dir, sfx, frame_stride=1):
    return torch.stack([load_file(f) for f in sorted(glob(osp.join(seq_dir, '*'+sfx)))[::frame_stride]], 0)


def get_frame_ids(seq_dir, frame_stride=1):
    return [osp.basename(f).split('_')[0] for f in sorted(glob(osp.join(seq_dir, '*_pose.txt')))[::frame_stride]]


def get_save_path(out_fold, fid, ext, prefix='', suffix=''):
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''
    out_path = osp.join(out_fold, prefix+fid+suffix+ext)
    os.makedirs(out_fold, exist_ok=True)
    return out_path


def save_meshes(out_fold, meshes, shape=None, tex_ims=None, tex_aux=None, prefix='', suffix='', fname_list=None):
    if shape is not None:
        num_frames = shape.size(0)
        meshes = meshes.extend(num_frames)
        meshes = meshes.update_padded(shape)
    for i, mesh in enumerate(meshes):
        tex_im = tex_ims[i].permute(1, 2, 0) if tex_ims is not None else None
        verts_uvs = tex_aux['verts_tex_uv'] if tex_aux is not None else None
        faces_uvs = tex_aux['face_tex_ids'] if tex_aux is not None else None
        if fname_list is not None:
            fid = fname_list[i]
        else:
            fid = '%07d'%i
        out_path = get_save_path(out_fold, fid, '.obj', prefix=prefix, suffix=suffix)
        pytorch3d.io.save_obj(
            out_path, mesh.verts_padded()[0], mesh.faces_padded()[0], texture_map=tex_im,
            verts_uvs=verts_uvs, faces_uvs=faces_uvs)


def save_images(out_fold, imgs, prefix='', suffix='', ext='.png', fname_list=None):
    imgs = imgs.permute(0,2,3,1).cpu().numpy()
    for i, img in enumerate(imgs):
        img = np.concatenate([np.flip(img[...,:3], -1), img[...,3:]], -1)  # RGBA to BGRA
        if 'depth' in suffix:
            im_out = np.uint16(img*65535.)
        else:
            im_out = np.uint8(img*255.)
        if fname_list is not None:
            fid = fname_list[i]
        else:
            fid = '%07d'%i
        out_path = get_save_path(out_fold, fid, ext, prefix=prefix, suffix=suffix)
        cv2.imwrite(out_path, im_out)


def update_camera_pose(cameras, position, at):
    cameras.R = pytorch3d.renderer.look_at_rotation(position, at).to(cameras.device)
    cameras.T = -torch.bmm(cameras.R.transpose(1, 2), position[:, :, None])[:, :, 0]


def get_soft_rasterizer_settings(image_size, sigma=0, gamma=1e-6, faces_per_pixel=30):
    blend_params = pytorch3d.renderer.BlendParams(sigma=sigma, gamma=gamma)
    settings = pytorch3d.renderer.RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=faces_per_pixel,
        cull_backfaces=True
    )
    return settings, blend_params


def get_light(amb=1., diff=0., spec=0., dir=(0,1,0)):
    return pytorch3d.renderer.DirectionalLights(ambient_color=((amb,)*3,),
                                                  diffuse_color=((diff,)*3,),
                                                  specular_color=((spec,)*3,),
                                                  direction=(dir,))


def create_soft_image_renderer(image_size, fov=25, device='cpu'):
    settings, blend_params = get_soft_rasterizer_settings(image_size)
    lights = get_light().to(device)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=fov)
    return pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(cameras=cameras, raster_settings=settings),
        shader=pytorch3d.renderer.SoftPhongShader(lights=lights, cameras=cameras, blend_params=blend_params, device=device)
    )


def create_hard_image_renderer(image_size, fov=25, device='cpu'):
    settings = pytorch3d.renderer.RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1, cull_backfaces=True)
    lights = get_light().to(device)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=fov)
    return pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(cameras=cameras, raster_settings=settings),
        shader=pytorch3d.renderer.HardPhongShader(lights=lights, cameras=cameras, device=device)
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


def estimate_bone_rotation(b):
    """
    (0, 0, 1) = matmul(R^(-1), b)

    assumes x, y is a symmetry plane

    returns R
    """
    b = b / torch.norm(b, dim=-1, keepdim=True)

    n = torch.FloatTensor([[1, 0, 0]]).to(b.device)
    n = n.expand_as(b)
    v = torch.cross(b, n, dim=-1)

    R = torch.stack([n, v, b], dim=-1).transpose(-2, -1)

    return R


def estimate_vector_rotation(vector_a, vector_b):
    """
    vector_a = matmul(R, vector_b)

    returns R

    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    """
    vector_a = vector_a / torch.norm(vector_a, dim=-1, keepdim=True)
    vector_b = vector_b / torch.norm(vector_b, dim=-1, keepdim=True)

    v = torch.cross(vector_a, vector_b, dim=-1)
    c = torch.sum(vector_a * vector_b, dim=-1)

    skew = torch.stack([
        torch.stack([torch.zeros_like(v[..., 0]), -v[..., 2], v[..., 1]], dim=-1),
        torch.stack([v[..., 2], torch.zeros_like(v[..., 0]), -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0], torch.zeros_like(v[..., 0])], dim=-1)],
        dim=-1)

    R = torch.eye(3, device=vector_a.device)[None] + skew + torch.matmul(skew, skew) / (1  + c[..., None, None])

    return R


def make_color_map_image(img_width, img_height, saturation=255):
    # """
    # Creates a color wheel based image of given width and height
    # Args:
    #     img_width (int):
    #     img_height (int):
    #
    # Returns:
    #     opencv image (numpy array): color wheel based image
    #
    # https://stackoverflow.com/questions/65609247/create-color-wheel-pattern-image-in-python
    # """
    # hue = np.fromfunction(lambda i, j: (np.arctan2(i-img_height/2, img_width/2-j) + np.pi)*(180/np.pi)/2,
    #                       (img_height, img_width), dtype=np.float)
    # saturation = np.ones((img_height, img_width)) * saturation
    # value = np.ones((img_height, img_width)) * 255
    # hsl = np.dstack((hue, saturation, value))
    # color_map = cv2.cvtColor(np.array(hsl, dtype=np.uint8), cv2.COLOR_HSV2RGB)

    x = torch.linspace(0,1,img_width).repeat(img_height,1)
    x = torch.cat([x[:, img_width//2:], x[:, :img_width//2]], 1)
    val = np.uint8(x*255)
    color_map = cv2.applyColorMap(val, cv2.COLORMAP_HSV)[:,:,::-1]
    return color_map


def add_mesh_color(mesh, color):
    verts = mesh.verts_padded()
    color = torch.FloatTensor(color).to(verts.device).view(1,1,3) / 255
    mesh.textures = pytorch3d.renderer.TexturesVertex(verts_features=verts*0+color)
    return mesh


## discontinuous version - used in training the ablation model
def desymmetrize_tex(mesh_aux):
    mesh_aux = copy.deepcopy(mesh_aux)
    tex_uv_seam1 = mesh_aux['verts_tex_uv'][:mesh_aux['num_verts_seam']].clone()
    tex_uv_seam1[:,0] = tex_uv_seam1[:,0] /2 + 0.5
    tex_uv_side1 = mesh_aux['verts_tex_uv'][mesh_aux['num_verts_seam']:mesh_aux['num_verts_seam']+mesh_aux['num_verts_one_side']].clone()
    tex_uv_side1[:,0] = tex_uv_side1[:,0] /2 + 0.5
    tex_uv_seam2 = mesh_aux['verts_tex_uv'][:mesh_aux['num_verts_seam']].clone()
    tex_uv_seam2[:,0] = tex_uv_seam2[:,0] /2
    tex_uv_side2 = mesh_aux['verts_tex_uv'][mesh_aux['num_verts_seam']+mesh_aux['num_verts_one_side']:].clone()
    tex_uv_side2[:,0] = tex_uv_side2[:,0] /2
    mesh_aux['verts_tex_uv'] = torch.cat([tex_uv_seam1, tex_uv_side1, tex_uv_side2, tex_uv_seam2], 0)

    num_faces = mesh_aux['face_tex_ids'].size(0)
    face_tex_ids1 = mesh_aux['face_tex_ids'][:num_faces//2].clone()
    face_tex_ids2 = mesh_aux['face_tex_ids'][num_faces//2:].clone()
    face_tex_ids2[face_tex_ids2 < mesh_aux['num_verts_seam']] += mesh_aux['num_verts_seam'] + 2*mesh_aux['num_verts_one_side']
    mesh_aux['face_tex_ids'] = torch.cat([face_tex_ids1, face_tex_ids2], 0)
    return mesh_aux


## continuous version - for visualizing uv map rendering
def desymmetrize_tex_cont(mesh_aux):
    mesh_aux = copy.deepcopy(mesh_aux)
    tex_uv_seam1 = mesh_aux['verts_tex_uv'][:mesh_aux['num_verts_seam']].clone()
    tex_uv_seam1[:,0] = -tex_uv_seam1[:,0] /2 + 0.5
    tex_uv_side1 = mesh_aux['verts_tex_uv'][mesh_aux['num_verts_seam']:mesh_aux['num_verts_seam']+mesh_aux['num_verts_one_side']].clone()
    tex_uv_side1[:,0] = -tex_uv_side1[:,0] /2 + 0.5
    tex_uv_seam2 = mesh_aux['verts_tex_uv'][:mesh_aux['num_verts_seam']].clone()
    tex_uv_seam2[:,0] = tex_uv_seam2[:,0] /2 + 0.5
    tex_uv_side2 = mesh_aux['verts_tex_uv'][mesh_aux['num_verts_seam']+mesh_aux['num_verts_one_side']:].clone()
    tex_uv_side2[:,0] = tex_uv_side2[:,0] /2 + 0.5
    mesh_aux['verts_tex_uv'] = torch.cat([tex_uv_seam1, tex_uv_side1, tex_uv_side2, tex_uv_seam2], 0)

    num_faces = mesh_aux['face_tex_ids'].size(0)
    face_tex_ids1 = mesh_aux['face_tex_ids'][:num_faces//2].clone()
    face_tex_ids2 = mesh_aux['face_tex_ids'][num_faces//2:].clone()
    face_tex_ids2[face_tex_ids2 < mesh_aux['num_verts_seam']] += mesh_aux['num_verts_seam'] + 2*mesh_aux['num_verts_one_side']
    mesh_aux['face_tex_ids'] = torch.cat([face_tex_ids1, face_tex_ids2], 0)
    return mesh_aux


def create_sphere(position, scale, device, color=[139, 149, 173]):
    mesh = pytorch3d.utils.ico_sphere(2).to(device)
    mesh = mesh.extend(position.shape[0])

    # scale and offset
    mesh = mesh.update_padded(mesh.verts_padded() * scale + position[:, None])

    mesh = add_mesh_color(mesh, color)

    return mesh


def create_elipsoid(bone, scale=0.05, color=[139, 149, 173], generic_rotation_estim=True):
    length = torch.norm(bone[:, 0] - bone[:, 1], dim=-1)

    mesh = pytorch3d.utils.ico_sphere(2).to(bone.device)
    mesh = mesh.extend(bone.shape[0])
    # scale x, y
    verts = mesh.verts_padded() * torch.FloatTensor([scale, scale, 1]).to(bone.device)
    # stretch along z axis, set the start to origin
    verts[:, :, 2] = verts[:, :, 2] * length[:, None] * 0.5 + length[:, None] * 0.5

    bone_vector = bone[:, 1] - bone[:, 0]
    z_vector = torch.FloatTensor([[0, 0, 1]]).to(bone.device)
    z_vector = z_vector.expand_as(bone_vector)
    if generic_rotation_estim:
        rot = estimate_vector_rotation(z_vector, bone_vector)
    else:
        rot = estimate_bone_rotation(bone_vector)
    tsf = pytorch3d.transforms.Rotate(rot, device=bone.device)
    tsf = tsf.compose(pytorch3d.transforms.Translate(bone[:, 0], device=bone.device))
    verts = tsf.transform_points(verts)

    mesh = mesh.update_padded(verts)

    mesh = add_mesh_color(mesh, color)

    return mesh


def create_bones_scene(bones, joint_color=[66, 91, 140], bone_color=[119, 144, 189], show_end_point=False):
    meshes = []
    for bone_i in range(bones.shape[1]):
        # points
        meshes += [create_sphere(bones[:, bone_i, 0], 0.1, bones.device, color=joint_color)]
        if show_end_point:
            meshes += [create_sphere(bones[:, bone_i, 1], 0.1, bones.device, color=joint_color)]

        # connecting ellipsoid
        meshes += [create_elipsoid(bones[:, bone_i], color=bone_color)]

    current_batch_size = bones.shape[0]
    meshes = [pytorch3d.structures.join_meshes_as_scene([m[i] for m in meshes]) for i in range(current_batch_size)]
    mesh = pytorch3d.structures.join_meshes_as_batch(meshes)

    return mesh


def transform_bones(bones, pose, rot_rep='euler_angle'):
    f, n, a, d = bones.shape
    bones = rearrange(bones, 'f n a d -> f (n a) d')
    bones = transform_verts(bones, pose, rot_rep=rot_rep)
    bones = rearrange(bones, 'f (n a) d -> f n a d', f=f, n=n, a=a, d=d)
    return bones


def render_bones(renderer, camera, bones, pose=None, batch_size=32, rot_rep='lookat'):
    num_frames = bones.size(0)

    light_dir = (0,1,1)
    lights = get_light(amb=0.8, diff=0.4, spec=0., dir=light_dir).to(bones.device)

    rendered = []
    for start_idx in range(0, num_frames, batch_size):
        end_idx = min(start_idx+batch_size, num_frames)
        bones_slice = bones[start_idx:end_idx]

        mesh = create_bones_scene(bones_slice)

        if pose is not None:
            pose_slice = pose[start_idx:end_idx]
            mesh = mesh.update_padded(transform_verts(mesh.verts_padded(), pose_slice, rot_rep=rot_rep))

        rendered += [renderer(meshes_world=mesh, cameras=camera, lights=lights).clamp(0,1).cpu()]
    rendered = torch.cat(rendered, 0)
    return rendered


def render_image(renderer, camera, mesh, shape, tex_im=None, tex_aux=None, batch_size=32, tex_mode='tex'):
    num_frames = shape.size(0)
    frame_mesh = mesh.extend(num_frames)
    frame_mesh = frame_mesh.update_padded(shape)

    rendered = []
    for start_idx in range(0, num_frames, batch_size):
        end_idx = min(start_idx+batch_size, num_frames)
        mesh_slice = frame_mesh[start_idx:end_idx]

        slice_size = len(mesh_slice)
        verts = mesh_slice.verts_padded()
        if tex_mode == 'tex':
            light_dir = (0,0,1)
            lights = get_light(amb=1., diff=0., spec=0., dir=light_dir).to(shape.device)
            mesh_slice.textures = pytorch3d.renderer.TexturesUV(maps=tex_im[start_idx:end_idx].permute(0, 2, 3, 1),  # texture maps are BxHxWx3
                                                     faces_uvs=tex_aux['face_tex_ids'].repeat(slice_size, 1, 1),
                                                     verts_uvs=tex_aux['verts_tex_uv'].repeat(slice_size, 1, 1))
        elif tex_mode == 'tex_with_light':
            light_dir = (0,1,1)
            lights = get_light(amb=0.8, diff=0.2, spec=0., dir=light_dir).to(shape.device)
            mesh_slice.textures = pytorch3d.renderer.TexturesUV(maps=tex_im[start_idx:end_idx].permute(0, 2, 3, 1),  # texture maps are BxHxWx3
                                                     faces_uvs=tex_aux['face_tex_ids'].repeat(slice_size, 1, 1),
                                                     verts_uvs=tex_aux['verts_tex_uv'].repeat(slice_size, 1, 1))
        elif tex_mode == 'blue':
            light_dir = (0,1,1)
            lights = get_light(amb=0.8, diff=0.4, spec=0., dir=light_dir).to(shape.device)
            color = torch.FloatTensor([156, 199, 234]).to(shape.device).view(1,1,3) / 255
            mesh_slice.textures = pytorch3d.renderer.TexturesVertex(verts_features=verts*0+color)
        elif tex_mode == 'gray':
            light_dir = (0,0,1)
            lights = get_light(amb=0.2, diff=0.7, spec=0., dir=light_dir).to(shape.device)
            color = 1.
            mesh_slice.textures = pytorch3d.renderer.TexturesVertex(verts_features=verts*0+color)

        rendered += [renderer(meshes_world=mesh_slice, cameras=camera, lights=lights).clamp(0,1).cpu()]
    rendered = torch.cat(rendered, 0)
    return rendered


def main(args):
    root_dir = '/scratch/shared/beegfs/szwu/projects/video3d/video3d_dev/results/cvpr22/bird/cvpr22_bird_release/final_bird'
    res_folder = 'test_results_016'
    out_root_dir = osp.join(root_dir, res_folder+'_visuals')

    cat_type = 'bird'  # bird, horse
    # init_pose = None
    init_pose = torch.FloatTensor([0, 180, 0, 0, 0, 0]) / 180 * np.pi  # if canonical shape is facing backward, use this to rotate it to facing front. Also update articulation parameters accordingly
    render_mode = 'visual'  # views, visual, animate, rotation, animate_video, party_parrot, shape_hierachy
    frame_stride = 10
    fps = 25
    disable_sym_tex = False
    fov = 25
    cam_pos_z_offset = 12
    temperature = 0.1
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

    device = 'cuda:0'
    image_size = 256
    render_batch_size = 16
    res_dir = osp.join(root_dir, res_folder)
    out_dir = osp.join(out_root_dir, render_mode)
    seq_ids = None  ## specify a list of sequence indices to render

    tex_modes = {'input_view': ['tex', 'blue', 'gray']}
    if render_mode == 'recon':
        pass
    elif render_mode == 'views':
        tex_modes['other_views'] = ['tex', 'gray']
    elif render_mode == 'visual':
        tex_modes['input_view_bones'] = ['tex']
        tex_modes['other_views'] = ['tex', 'gray']
        tex_modes['animate'] = ['tex', 'gray']
    elif render_mode == 'animate':
        tex_modes['animate'] = ['tex', 'gray']
    elif render_mode == 'rotation':
        tex_modes['rotation'] = ['tex', 'gray']
    elif render_mode == 'animate_video':
        tex_modes['animate_video'] = ['tex']
    elif render_mode == 'party_parrot':
        tex_modes['party_parrot'] = ['tex']
    elif render_mode == 'shape_hierachy':
        tex_modes = {'input_view': [], 'prior': ['gray'], 'seq': ['gray'], 'frame': ['gray', 'tex']}

    meshes, mesh_aux = get_symmetric_ico_sphere(subdiv=3, return_tex_uv=True, return_face_triangle_tex_map=True, device=device)
    renderer = create_soft_image_renderer(image_size, fov=fov, device=device)
    camera = pytorch3d.renderer.FoVPerspectiveCameras(fov=fov).to(device)
    cam_pos = torch.FloatTensor([[0, 0, cam_pos_z_offset]]).to(device)
    cam_at = torch.FloatTensor([[0, 0, 0]]).to(device)
    update_camera_pose(camera, position=cam_pos, at=cam_at)
    color_wheel_tex_im = torch.FloatTensor(make_color_map_image(image_size, image_size) /255.).flip(1).to(device).permute(2,0,1)[None,:,:,:]

    # hack to turn off texture symmetry
    if disable_sym_tex:
        mesh_aux = desymmetrize_tex(mesh_aux)

    seq_folder_list = [d for d in sorted(os.listdir(res_dir)) if osp.isdir(osp.join(res_dir, d))]
    if seq_ids is not None:
        seq_folder_list = seq_folder_list[seq_ids]

    for i, seq_folder in enumerate(seq_folder_list):
        print(f'rendering sequence {seq_folder}')

        sfxs = ['prior_shape.txt', 'seq_shape.txt', 'frame_shape.txt', 'pose.txt', 'arti_params.txt', 'mask_gt.png', 'input_image.png', 'texture_pred.png']
        (prior_shape, seq_shape, frame_shape, pose, arti_params, mask_gt, input_image, tex_im) = (load_seq_files(osp.join(res_dir, seq_folder), sfx, frame_stride=frame_stride).to(device) for sfx in sfxs)
        fname_list = get_frame_ids(osp.join(res_dir, seq_folder), frame_stride=frame_stride)
        mask_gt = mask_gt[..., 0]
        input_image = input_image.permute(0,3,1,2)
        tex_im = tex_im.permute(0,3,1,2)
        num_frames = pose.size(0)
        out_seq_dir = osp.join(out_dir, f'{seq_folder}')

        if 'input_view' in tex_modes:
            posed_shape = transform_verts(frame_shape, pose, rot_rep='lookat')
            for tex_mode in tex_modes['input_view']:
                rendered = render_image(renderer, camera, meshes, posed_shape, tex_im=tex_im, tex_aux=mesh_aux, batch_size=render_batch_size, tex_mode=tex_mode).permute(0,3,1,2)
                save_images(out_seq_dir, rendered, suffix='input_view_'+tex_mode, ext='.png', fname_list=fname_list)

            ## color wheel texture map
            desym_mesh_aux = desymmetrize_tex_cont(mesh_aux)
            rendered = render_image(renderer, camera, meshes, posed_shape, tex_im=color_wheel_tex_im.repeat(num_frames,1,1,1), tex_aux=desym_mesh_aux, batch_size=render_batch_size, tex_mode='tex').permute(0,3,1,2)
            save_images(out_seq_dir, rendered, suffix='input_view_colormap', ext='.png', fname_list=fname_list)
            save_images(out_seq_dir, input_image, suffix='input_image', ext='.png', fname_list=fname_list)

        if 'input_view_bones' in tex_modes:
            articulated_shape, aux = skinning(seq_shape=seq_shape.unsqueeze(0), arti_params=arti_params.unsqueeze(0), n_bones=num_body_bones, n_legs=num_legs, n_leg_bones=num_leg_bones, body_bones_type=body_bones_type, output_posed_bones=True, temperature=temperature, static_root_bones=static_root_bones)
            articulated_bones = aux['posed_bones'].squeeze(0)
            posed_articulated_bones = transform_bones(articulated_bones, pose, rot_rep='lookat')
            rendered_bones = render_bones(renderer, camera, posed_articulated_bones, batch_size=render_batch_size).permute(0,3,1,2)
            posed_shape = transform_verts(articulated_shape.squeeze(0), pose, rot_rep='lookat')
            for tex_mode in tex_modes['input_view_bones']:
                rendered = render_image(renderer, camera, meshes, posed_shape, tex_im=tex_im, tex_aux=mesh_aux, batch_size=render_batch_size, tex_mode=tex_mode).permute(0,3,1,2)
                rendered = rendered_bones[:,3:]*0.8*rendered_bones + (1 - rendered_bones[:,3:]*0.8)*rendered
                save_images(out_seq_dir, rendered, suffix='input_view_bones_'+tex_mode, ext='.png', fname_list=fname_list)

        if 'other_views' in tex_modes:
            other_poses = [(torch.FloatTensor([30, -90, 0, 0, 0, 0]) / 180 * np.pi,)]
            other_poses += [(torch.FloatTensor([30, -45, 0, 0, 0, 0]) / 180 * np.pi,)]
            other_poses += [(torch.FloatTensor([30, 0, 0, 0, 0, 0]) / 180 * np.pi,)]
            other_poses += [(torch.FloatTensor([30, 45, 0, 0, 0, 0]) / 180 * np.pi,)]
            other_poses += [(torch.FloatTensor([30, 90, 0, 0, 0, 0]) / 180 * np.pi,)]
            other_poses += [(torch.FloatTensor([100, 0, 0, 0, 0, 0]) / 180 * np.pi,)]
            other_poses += [(torch.FloatTensor([75, -45, 0, 0, 0, 0]) / 180 * np.pi,)]
            other_poses += [(torch.FloatTensor([75, -30, 0, 0, 0, 0]) / 180 * np.pi,
                             torch.FloatTensor([-45, 0, 0, 0, 0, 0]) / 180 * np.pi)]
            other_poses += [(torch.FloatTensor([120, 60, 0, 0, 0, 0]) / 180 * np.pi,
                             torch.FloatTensor([-45, 0, 0, 0, 0, 0]) / 180 * np.pi)]
            for j, poses in enumerate(other_poses):
                posed_shape = frame_shape
                if init_pose is not None:
                    poses = (init_pose,) + poses
                for p in poses:
                    p = p.to(device).repeat(num_frames,1)
                    posed_shape = transform_verts(posed_shape, p, rot_rep='euler_angle')
                for tex_mode in tex_modes['other_views']:
                    rendered = render_image(renderer, camera, meshes, posed_shape, tex_im=tex_im, tex_aux=mesh_aux, batch_size=render_batch_size, tex_mode=tex_mode).permute(0,3,1,2)
                    save_images(out_seq_dir, rendered, suffix='view_%02d_'%j+tex_mode, ext='.png', fname_list=fname_list)

        if 'other_views_bones' in tex_modes:
            other_poses = [(torch.FloatTensor([30, -90, 0, 0, 0, 0]) / 180 * np.pi,)]
            other_poses += [(torch.FloatTensor([30, -45, 0, 0, 0, 0]) / 180 * np.pi,)]
            other_poses += [(torch.FloatTensor([30, 0, 0, 0, 0, 0]) / 180 * np.pi,)]
            other_poses += [(torch.FloatTensor([30, 45, 0, 0, 0, 0]) / 180 * np.pi,)]
            other_poses += [(torch.FloatTensor([30, 90, 0, 0, 0, 0]) / 180 * np.pi,)]
            other_poses += [(torch.FloatTensor([100, 0, 0, 0, 0, 0]) / 180 * np.pi,)]
            other_poses += [(torch.FloatTensor([75, -45, 0, 0, 0, 0]) / 180 * np.pi,)]
            other_poses += [(torch.FloatTensor([75, -30, 0, 0, 0, 0]) / 180 * np.pi,
                             torch.FloatTensor([-45, 0, 0, 0, 0, 0]) / 180 * np.pi)]
            other_poses += [(torch.FloatTensor([120, 60, 0, 0, 0, 0]) / 180 * np.pi,
                             torch.FloatTensor([-45, 0, 0, 0, 0, 0]) / 180 * np.pi)]
            articulated_shape, aux = skinning(seq_shape=seq_shape.unsqueeze(0), n_bones=num_body_bones, arti_params=arti_params.unsqueeze(0), output_posed_bones=True, temperature=temperature)
            articulated_bones = aux['posed_bones'].squeeze(0)
            for j, poses in enumerate(other_poses):
                posed_shape = articulated_shape.squeeze(0)
                posed_articulated_bones = articulated_bones
                if init_pose is not None:
                    poses = (init_pose,) + poses
                for p in poses:
                    p = p.to(device).repeat(num_frames,1)
                    posed_shape = transform_verts(posed_shape, p, rot_rep='euler_angle')
                    posed_articulated_bones = transform_bones(posed_articulated_bones, p, rot_rep='euler_angle')
                rendered_bones = render_bones(renderer, camera, posed_articulated_bones, batch_size=render_batch_size).permute(0,3,1,2)
                for tex_mode in tex_modes['other_views']:
                    rendered = render_image(renderer, camera, meshes, posed_shape, tex_im=tex_im, tex_aux=mesh_aux, batch_size=render_batch_size, tex_mode=tex_mode).permute(0,3,1,2)
                    rendered = rendered_bones[:,3:]*0.8*rendered_bones + (1 - rendered_bones[:,3:]*0.8)*rendered
                    save_images(out_seq_dir, rendered, suffix='view_bones_%02d_'%j+tex_mode, ext='.png', fname_list=fname_list)

        if 'rotation' in tex_modes:
            posed_shape = transform_verts(frame_shape, pose, rot_rep='lookat')
            rots = np.linspace(0, 360, 75)
            for j, rot in enumerate(rots):
                rot_pose = torch.FloatTensor([0, rot, 0, 0, 0, 0]).to(device).repeat(num_frames,1) / 180 * np.pi
                rot_shape = transform_verts(posed_shape, rot_pose, rot_rep='euler_angle')
                for tex_mode in tex_modes['rotation']:
                    rendered = render_image(renderer, camera, meshes, rot_shape, tex_im=tex_im, tex_aux=mesh_aux, batch_size=render_batch_size, tex_mode=tex_mode).permute(0,3,1,2)
                    save_images(out_seq_dir, rendered, suffix='rot_%03d_'%j+tex_mode, ext='.png', fname_list=fname_list)
            for fname in fname_list:
                for tex_mode in tex_modes['rotation']:
                    im_list = sorted(glob(osp.join(out_seq_dir, fname+'_rot_*'+tex_mode+'*')))
                    clip = ImageSequenceClip(im_list, fps=fps)
                    clip.write_videofile(osp.join(out_seq_dir, fname+'_rot_'+tex_mode+'.mp4'))

        ## animation
        if 'animate' in tex_modes:
            all_arti_params = [torch.FloatTensor([[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]] + [[0,0,0]]*16) / 180* np.pi]  # canonical
            all_arti_params += [torch.FloatTensor([[0,0,0], [30,60,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]] + [[0,0,0]]*16) / 180* np.pi]  # head top right
            all_arti_params += [torch.FloatTensor([[0,0,0], [-30,-60,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]] + [[0,0,0]]*16) / 180* np.pi]  # head top right
            for j, arti_params in enumerate(all_arti_params):
                arti_params = arti_params.to(device).repeat(num_frames,1,1)
                animated_shape, aux = skinning(seq_shape=seq_shape.unsqueeze(0), n_bones=num_body_bones, arti_params=arti_params.unsqueeze(0), output_posed_bones=True, temperature=temperature)
                articulated_bones = aux['posed_bones'].squeeze(0)
                animated_shape = animated_shape[0]
                if init_pose is not None:
                    articulated_bones = transform_bones(articulated_bones, init_pose.to(device).repeat(num_frames,1), rot_rep='euler_angle')
                    animated_shape = transform_verts(animated_shape, init_pose.to(device).repeat(num_frames,1), rot_rep='euler_angle')
                rotations = [-60, 0, 60]
                for r, rot in enumerate(rotations):
                    p = torch.FloatTensor([60, rot, 0, 0, 0, 0]).to(device).repeat(num_frames,1) / 180 * np.pi
                    posed_shape = transform_verts(animated_shape, p, rot_rep='euler_angle')
                    posed_articulated_bones = transform_bones(articulated_bones, p, rot_rep='euler_angle')
                    rendered_bones = render_bones(renderer, camera, posed_articulated_bones, batch_size=render_batch_size).permute(0,3,1,2)
                    for tex_mode in tex_modes['animate']:
                        rendered = render_image(renderer, camera, meshes, posed_shape, tex_im=tex_im, tex_aux=mesh_aux, batch_size=render_batch_size, tex_mode=tex_mode).permute(0,3,1,2)
                        rendered = rendered_bones[:,3:]*0.8*rendered_bones + (1 - rendered_bones[:,3:]*0.8)*rendered
                        save_images(out_seq_dir, rendered, suffix='ani_%02d_view_%02d_'%(j,r)+tex_mode, ext='.png', fname_list=fname_list)

        ## animation video
        if 'animate_video' in tex_modes:
            arti_params_seq = []
            for rx in np.concatenate([np.zeros(10), np.linspace(0, 60, 15), np.linspace(60, -60, 30), np.linspace(-60, 0, 15)]):
                arti_params_seq += [torch.FloatTensor([[-30,0,0], [rx,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]) / 180* np.pi]
            for ry in np.concatenate([np.zeros(10), np.linspace(0, 60, 15), np.linspace(60, -60, 30), np.linspace(-60, 0, 15)]):
                arti_params_seq += [torch.FloatTensor([[-30,0,0], [0,ry,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]) / 180* np.pi]
            for rz in np.concatenate([np.zeros(10), np.linspace(0, 60, 15), np.linspace(60, -60, 30), np.linspace(-60, 0, 15)]):
                arti_params_seq += [torch.FloatTensor([[-30,0,0], [0,0,rz], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]) / 180* np.pi]
            for j, arti_params in enumerate(arti_params_seq):
                arti_params = arti_params.to(device).repeat(num_frames,1,1)
                num_body_bones = arti_params.size(1)
                animated_shape, aux = skinning(seq_shape=seq_shape.unsqueeze(0), n_bones=num_body_bones, arti_params=arti_params.unsqueeze(0))
                animated_shape = animated_shape[0]
                if init_pose is not None:
                    animated_shape = transform_verts(animated_shape, init_pose.to(device).repeat(num_frames,1), rot_rep='euler_angle')
                p = torch.FloatTensor([30, 0, 0, 0, 0, 0]).to(device).repeat(num_frames,1) / 180 * np.pi
                posed_shape = transform_verts(animated_shape, p, rot_rep='euler_angle')
                for tex_mode in tex_modes['animate_video']:
                    rendered = render_image(renderer, camera, meshes, posed_shape, tex_im=tex_im, tex_aux=mesh_aux, batch_size=render_batch_size, tex_mode=tex_mode).permute(0,3,1,2)
                    save_images(out_seq_dir, rendered, suffix='ani_%03d_'%j+tex_mode, ext='.png', fname_list=fname_list)
            for fname in fname_list:
                for tex_mode in tex_modes['animate_video']:
                    im_list = sorted(glob(osp.join(out_seq_dir, fname+'_ani_*'+tex_mode+'*')))
                    clip = ImageSequenceClip(im_list, fps=fps)
                    clip.write_videofile(osp.join(out_seq_dir, fname+'_ani_'+tex_mode+'.mp4'))

        ## party parrot animation
        if 'party_parrot' in tex_modes:
            num_rot_frames = 25

            ## left right
            rots = np.linspace(-90, 270, num_rot_frames) / 180* np.pi
            for j, rot in enumerate(rots):
                rot_x = np.cos(rot*2) *30
                rot_y = np.cos(rot) *60
                arti_params = torch.FloatTensor([[-30,0,0], [rot_x,rot_y,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]) / 180* np.pi

                arti_params = arti_params.to(device).repeat(num_frames,1,1)
                num_body_bones = arti_params.size(1)
                animated_shape, aux = skinning(seq_shape=seq_shape.unsqueeze(0), n_bones=num_body_bones, arti_params=arti_params.unsqueeze(0), output_posed_bones=True, temperature=temperature)
                animated_shape = animated_shape[0]
                if init_pose is not None:
                    animated_shape = transform_verts(animated_shape, init_pose.to(device).repeat(num_frames,1), rot_rep='euler_angle')
                p = torch.FloatTensor([30, 0, 0, 0, 0, 0]).to(device).repeat(num_frames,1) / 180 * np.pi
                posed_shape = transform_verts(animated_shape, p, rot_rep='euler_angle')

                # render bones
                articulated_bones = aux['posed_bones'].squeeze(0)
                if init_pose is not None:
                    articulated_bones = transform_bones(articulated_bones, init_pose.to(device).repeat(num_frames,1), rot_rep='euler_angle')
                posed_articulated_bones = transform_bones(articulated_bones, p, rot_rep='euler_angle')
                rendered_bones = render_bones(renderer, camera, posed_articulated_bones, batch_size=render_batch_size).permute(0,3,1,2)
                bone_alpha = rendered_bones[:,3:] * 0.5

                for tex_mode in tex_modes['party_parrot']:
                    rendered = render_image(renderer, camera, meshes, posed_shape, tex_im=tex_im, tex_aux=mesh_aux, batch_size=render_batch_size, tex_mode=tex_mode).permute(0,3,1,2)
                    rendered_composed = rendered_bones*bone_alpha + (1-bone_alpha)*rendered
                    save_images(out_seq_dir, rendered_composed, prefix='', suffix='party_%03d_'%j+tex_mode+'_bones', ext='.png', fname_list=fname_list)

            for fname in fname_list:
                for tex_mode in tex_modes['party_parrot']:
                    im_list = sorted(glob(osp.join(out_seq_dir, fname+'_party_*'+tex_mode+'_bones*')))
                    clip = ImageSequenceClip(im_list, fps=fps)
                    clip.write_videofile(osp.join(out_seq_dir, fname+'_party_'+tex_mode+'_bones.mp4'))

        ## 3-level hierarchical shape
        if render_mode == 'shape_hierachy':
            side_pose = torch.FloatTensor([0, 90, 0, 0, 0, 0]).to(device).repeat(num_frames,1) / 180 * np.pi

            if init_pose is not None:
                prior_shape = transform_verts(prior_shape, init_pose.to(device).repeat(num_frames,1), rot_rep='euler_angle')
            posed_shape = transform_verts(prior_shape, side_pose, rot_rep='euler_angle')
            for tex_mode in tex_modes['prior']:
                rendered = render_image(renderer, camera, meshes, posed_shape, tex_im=tex_im, tex_aux=mesh_aux, batch_size=render_batch_size, tex_mode=tex_mode).permute(0,3,1,2)
                save_images(out_seq_dir, rendered, suffix='prior_' + tex_mode, ext='.png', fname_list=fname_list)

            if init_pose is not None:
                seq_shape = transform_verts(seq_shape, init_pose.to(device).repeat(num_frames,1), rot_rep='euler_angle')
            posed_shape = transform_verts(seq_shape, side_pose, rot_rep='euler_angle')
            for tex_mode in tex_modes['seq']:
                rendered = render_image(renderer, camera, meshes, posed_shape, tex_im=tex_im, tex_aux=mesh_aux, batch_size=render_batch_size, tex_mode=tex_mode).permute(0,3,1,2)
                save_images(out_seq_dir, rendered, suffix='seq_' + tex_mode, ext='.png', fname_list=fname_list)

            if init_pose is not None:
                frame_shape = transform_verts(frame_shape, init_pose.to(device).repeat(num_frames,1), rot_rep='euler_angle')
            posed_shape = transform_verts(frame_shape, side_pose, rot_rep='euler_angle')
            for tex_mode in tex_modes['frame']:
                rendered = render_image(renderer, camera, meshes, posed_shape, tex_im=tex_im, tex_aux=mesh_aux, batch_size=render_batch_size, tex_mode=tex_mode).permute(0,3,1,2)
                save_images(out_seq_dir, rendered, suffix='frame_' + tex_mode, ext='.png', fname_list=fname_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    with torch.no_grad():
        main(args)
