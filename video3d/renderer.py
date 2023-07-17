import numpy as np
import torch
import torch.nn as nn
import pytorch3d
import pytorch3d.loss
import pytorch3d.renderer
import pytorch3d.structures
import pytorch3d.io
import pytorch3d.transforms
from .utils import sphere
from einops import rearrange


def update_camera_pose(cameras, position, at):
    cameras.R = pytorch3d.renderer.look_at_rotation(position, at).to(cameras.device)
    cameras.T = -torch.bmm(cameras.R.transpose(1, 2), position[:, :, None])[:, :, 0]


def get_soft_rasterizer_settings(image_size, sigma=1e-6, gamma=1e-6, faces_per_pixel=30):
    blend_params = pytorch3d.renderer.BlendParams(sigma=sigma, gamma=gamma)
    settings = pytorch3d.renderer.RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=faces_per_pixel,
    )
    return settings, blend_params


class Renderer(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        self.device = cfgs.get('device', 'cpu')
        self.image_size = cfgs.get('out_image_size', 64)
        self.fov = cfgs.get('fov', 25)
        self.rot_rep = cfgs.get('rot_rep', 'euler_angle')
        self.cam_pos_z_offset = cfgs.get('cam_pos_z_offset', 10.)
        cam_pos = torch.FloatTensor([[0, 0, self.cam_pos_z_offset]]).to(self.device)
        cam_at = torch.FloatTensor([[0, 0, 0]]).to(self.device)
        self.cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=self.fov).to(self.device)
        update_camera_pose(self.cameras, position=cam_pos, at=cam_at)
        self.image_renderer = self._create_image_renderer()

        self.ico_sphere_subdiv = cfgs.get('ico_sphere_subdiv', 2)
        self.meshes, aux = sphere.get_symmetric_ico_sphere(subdiv=self.ico_sphere_subdiv, return_tex_uv=True, return_face_triangle_tex_map=True, device=self.device)
        self.init_verts = self.meshes.verts_padded()
        self.tex_faces_uv = aux['face_tex_ids'].unsqueeze(0)
        self.tex_verts_uv = aux['verts_tex_uv'].unsqueeze(0)
        self.face_triangle_tex_map = aux['face_triangle_tex_map'].permute(2,0,1)
        self.tex_map_seam_mask = aux['seam_mask'].permute(2,0,1)
        self.num_verts_total = self.init_verts.size(1)
        self.num_verts_seam = aux['num_verts_seam']
        self.num_verts_one_side = aux['num_verts_one_side']

        # hack to turn off texture symmetry
        if cfgs.get('disable_sym_tex', False):
            tex_uv_seam1 = self.tex_verts_uv[:,:aux['num_verts_seam']].clone()
            tex_uv_seam1[:,:,0] = tex_uv_seam1[:,:,0] /2 + 0.5
            tex_uv_side1 = self.tex_verts_uv[:,aux['num_verts_seam']:aux['num_verts_seam']+aux['num_verts_one_side']].clone()
            tex_uv_side1[:,:,0] = tex_uv_side1[:,:,0] /2 + 0.5
            tex_uv_seam2 = self.tex_verts_uv[:,:aux['num_verts_seam']].clone()
            tex_uv_seam2[:,:,0] = tex_uv_seam2[:,:,0] /2
            tex_uv_side2 = self.tex_verts_uv[:,aux['num_verts_seam']+aux['num_verts_one_side']:].clone()
            tex_uv_side2[:,:,0] = tex_uv_side2[:,:,0] /2
            self.tex_verts_uv = torch.cat([tex_uv_seam1, tex_uv_side1, tex_uv_side2, tex_uv_seam2], 1)

            num_faces = self.tex_faces_uv.shape[1]
            face_tex_ids1 = self.tex_faces_uv[:, :num_faces//2].clone()
            face_tex_ids2 = self.tex_faces_uv[:, num_faces//2:].clone()
            face_tex_ids2[face_tex_ids2 < aux['num_verts_seam']] += aux['num_verts_seam'] + 2*aux['num_verts_one_side']
            self.tex_faces_uv = torch.cat([face_tex_ids1, face_tex_ids2], 1)
            self.face_triangle_tex_map = torch.cat([self.face_triangle_tex_map, self.face_triangle_tex_map.flip(2)], 2)
            self.tex_map_seam_mask = torch.cat([self.tex_map_seam_mask, self.tex_map_seam_mask.flip(2)], 2)

    def _create_silhouette_renderer(self):
        settings, blend_params = get_soft_rasterizer_settings(self.image_size)
        return pytorch3d.renderer.MeshRenderer(
            rasterizer=pytorch3d.renderer.MeshRasterizer(cameras=self.cameras, raster_settings=settings),
            shader=pytorch3d.renderer.SoftSilhouetteShader(cameras=self.cameras, blend_params=blend_params)
        )

    def _create_image_renderer(self):
        settings, blend_params = get_soft_rasterizer_settings(self.image_size)
        lights = pytorch3d.renderer.DirectionalLights(device=self.device,
                                                      ambient_color=((1., 1., 1.),),
                                                      diffuse_color=((0., 0., 0.),),
                                                      specular_color=((0., 0., 0.),),
                                                      direction=((0, 1, 0),))
        return pytorch3d.renderer.MeshRenderer(
            rasterizer=pytorch3d.renderer.MeshRasterizer(cameras=self.cameras, raster_settings=settings),
            shader=pytorch3d.renderer.SoftPhongShader(device=self.device, lights=lights, cameras=self.cameras, blend_params=blend_params)
        )

    def transform_verts(self, verts, pose):
        b, f, _ = pose.shape
        if self.rot_rep == 'euler_angle':
            rot_mat = pytorch3d.transforms.euler_angles_to_matrix(pose[...,:3].view(-1,3), convention='XYZ')
            tsf = pytorch3d.transforms.Rotate(rot_mat, device=pose.device)
        elif self.rot_rep == 'quaternion':
            rot_mat = pytorch3d.transforms.quaternion_to_matrix(pose[...,:4].view(-1,4))
            tsf = pytorch3d.transforms.Rotate(rot_mat, device=pose.device)
        elif self.rot_rep == 'lookat':
            rot_mat = pose[...,:9].view(-1,3,3)
            tsf = pytorch3d.transforms.Rotate(rot_mat, device=pose.device)
        else:
            raise NotImplementedError
        tsf = tsf.compose(pytorch3d.transforms.Translate(pose[...,-3:].view(-1,3), device=pose.device))
        new_verts = tsf.transform_points(rearrange(verts, 'b f ... -> (b f) ...'))
        return rearrange(new_verts, '(b f) ... -> b f ...', b=b, f=f)

    def symmetrize_shape(self, shape):
        verts_seam = shape[:,:,:self.num_verts_seam] * torch.FloatTensor([0,1,1]).to(shape.device)
        verts_one_side = shape[:,:,self.num_verts_seam:self.num_verts_seam+self.num_verts_one_side] * torch.FloatTensor([1,1,1]).to(shape.device)
        verts_other_side = verts_one_side * torch.FloatTensor([-1,1,1]).to(shape.device)
        shape = torch.cat([verts_seam, verts_one_side, verts_other_side], 2)
        return shape

    def get_deformed_mesh(self, shape, pose=None, return_verts=False):
        b, f, _, _ = shape.shape
        if pose is not None:
            shape = self.transform_verts(shape, pose)
        mesh = self.meshes.extend(b*f)
        mesh = mesh.update_padded(rearrange(shape, 'b f ... -> (b f) ...'))
        if return_verts:
            return shape, mesh
        else:
            return mesh

    def get_textures(self, tex_im):
        b, f, c, h, w = tex_im.shape
        textures = pytorch3d.renderer.TexturesUV(maps=rearrange(tex_im, 'b f c h w -> (b f) h w c'),  # texture maps are BxHxWx3
                                                 faces_uvs=self.tex_faces_uv.repeat(b*f, 1, 1),
                                                 verts_uvs=self.tex_verts_uv.repeat(b*f, 1, 1))
        return textures

    def render_flow(self, meshes, shape, pose, deformed_shape=None):
        b, f, _, _ = shape.shape
        if f < 2:
            return None
        if deformed_shape is None:
            deformed_shape, meshes = self.get_deformed_mesh(shape, pose=pose, return_verts=True)
        
        im_size_WH = torch.FloatTensor([self.image_size, self.image_size]).to(shape.device)  # (w,h)
        verts_2d = self.cameras.transform_points_screen(rearrange(deformed_shape, 'b f ... -> (b f) ...'), im_size_WH.view(1,2).repeat(b*f,1), eps=1e-7)
        verts_2d = rearrange(verts_2d, '(b f) ... -> b f ...', b=b, f=f)
        verts_flow = verts_2d[:, 1:, :, :2] - verts_2d[:, :-1, :, :2]  # Bx(F-1)xVx(x,y)
        verts_flow = verts_flow / im_size_WH.view(1, 1, 1, 2) * 0.5 + 0.5  # 0~1
        flow_tex = torch.nn.functional.pad(verts_flow, pad=[0, 1, 0, 0, 0, 1])  # BxFxVx3
        meshes.textures = pytorch3d.renderer.TexturesVertex(verts_features=flow_tex.view(b*f, -1, 3))
        flow = self.image_renderer(meshes_world=meshes, cameras=self.cameras)
        flow = rearrange(flow, '(b f) ... -> b f ...', b=b, f=f)[:, :-1]  # Bx(F-1)xHxWx3
        flow_mask = (flow[..., 3:] > 0.01).float()
        flow = (flow[..., :2] - 0.5) * 2 * flow_mask  # Bx(F-1)xHxWx2, -1~1
        return flow

    def forward(self, pose, texture, shape, render_flow=True):
        b, f, _ = pose.shape
        deformed_shape, mesh = self.get_deformed_mesh(shape, pose=pose, return_verts=True)
        if render_flow:
            flow = self.render_flow(mesh, shape, pose, deformed_shape=deformed_shape)  # Bx(F-1)xHxWx2
        else:
            flow = None
        mesh.textures = self.get_textures(texture)
        image = self.image_renderer(meshes_world=mesh, cameras=self.cameras)
        image = rearrange(image, '(b f) ... -> b f ...', b=b, f=f)
        return image, flow, mesh
