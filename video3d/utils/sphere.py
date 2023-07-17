import io
import numpy as np
import cv2
from PIL import Image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import pytorch3d
import pytorch3d.renderer
import pytorch3d.structures
import pytorch3d.io
import pytorch3d.transforms
import pytorch3d.utils


## https://stackoverflow.com/a/58641662/11471407
def fig_to_img(fig, dpi=200, im_size=(512,512)):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img = np.array(Image.open(buf).convert('RGB').resize(im_size)) / 255.
    return img


def get_ico_sphere(subdiv=1):
    return pytorch3d.utils.ico_sphere(level=subdiv)


def get_symmetric_ico_sphere(subdiv=1, return_tex_uv=True, return_face_triangle_tex_map=True, device='cpu'):
    sph_mesh = get_ico_sphere(subdiv=subdiv)
    sph_verts = sph_mesh.verts_padded()[0]
    sph_faces = sph_mesh.faces_padded()[0]

    ## rotate the default mesh s.t. the seam is exactly on yz-plane
    rot_z = np.arctan(0.5000/0.3090)  # computed from vertices in ico_sphere
    tfs = pytorch3d.transforms.RotateAxisAngle(rot_z, 'Z', degrees=False)
    rotated_verts = tfs.transform_points(sph_verts)

    ## identify vertices on each side and on the seam
    verts_id_seam = []
    verts_id_one_side = []
    verts_id_other_side = []
    for i, v in enumerate(rotated_verts):
        ## on the seam, x=0
        if v[0].abs() < 0.001:  # threshold 0.001
            verts_id_seam += [i]
            rotated_verts[i][0] = 0.  # force it to be 0

        ## right side, x>0
        elif v[0] > 0:
            verts_id_one_side += [i]

        ## left side, x<0
        else:
            verts_id_other_side += [i]

    ## create a new set of symmetric vertices
    new_vid = 0
    vid_old_to_new = {}
    verts_seam = []
    for vid in verts_id_seam:
        verts_seam += [rotated_verts[vid]]
        vid_old_to_new[vid] = new_vid
        new_vid += 1
    verts_seam = torch.stack(verts_seam, 0)

    verts_one_side = []
    for vid in verts_id_one_side:
        verts_one_side += [rotated_verts[vid]]
        vid_old_to_new[vid] = new_vid
        new_vid += 1
    verts_one_side = torch.stack(verts_one_side, 0)

    verts_other_side = []
    for vid in verts_id_one_side:
        verts_other_side += [rotated_verts[vid] * torch.FloatTensor([-1,1,1])]  # flip x
        new_vid += 1
    verts_other_side = torch.stack(verts_other_side, 0)

    new_verts = torch.cat([verts_seam, verts_one_side, verts_other_side], 0)

    ## create a new set of symmetric faces
    faces_one_side = []
    faces_other_side = []
    for old_face in sph_faces:
        new_face1 = []  # one side
        new_face2 = []  # the other side
        for vi in old_face:
            vi = vi.item()
            if vi in verts_id_seam:
                new_face1 += [vid_old_to_new[vi]]
                new_face2 += [vid_old_to_new[vi]]
            elif vi in verts_id_one_side:
                new_face1 += [vid_old_to_new[vi]]
                new_face2 += [vid_old_to_new[vi]+len(verts_id_one_side)]  # assuming the symmetric vertices are appended right after the original ones
            else:
                break

        if len(new_face1) == 3:  # no vert on the other side
            faces_one_side += [new_face1]
            faces_other_side += [new_face2[::-1]]  # reverse face orientation
    new_faces = faces_one_side + faces_other_side
    new_faces = torch.LongTensor(new_faces)
    sym_sph_mesh = pytorch3d.structures.Meshes(verts=[new_verts], faces=[new_faces])

    aux = {}
    aux['num_verts_seam'] = len(verts_seam)
    aux['num_verts_one_side'] = len(verts_one_side)

    ## create texture map uv
    if return_tex_uv:
        verts_tex_uv = torch.stack([-new_verts[:,2], new_verts[:,1]], 1)  # -z,y
        verts_tex_uv = verts_tex_uv / ((verts_tex_uv**2).sum(1,keepdim=True)**0.5).clamp(min=1e-8)
        magnitude = new_verts[:,:1].abs().acos()  # set magnitude to angle deviation from vertical axis, for more even texture mapping
        magnitude = magnitude / magnitude.max() *0.95  # max 0.95
        verts_tex_uv = verts_tex_uv * magnitude
        verts_tex_uv = verts_tex_uv /2 + 0.5  # rescale to 0~1
        face_tex_ids = new_faces
        aux['verts_tex_uv'] = verts_tex_uv.to(device)
        aux['face_tex_ids'] = face_tex_ids.to(device)

    ## create face color map
    if return_face_triangle_tex_map:
        dpi = 200
        im_size = (512, 512)
        fig = plt.figure(figsize=(8,8), dpi=dpi, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        num_colors = 10
        cmap = plt.get_cmap('tab10', num_colors)
        num_faces = len(face_tex_ids)
        face_tex_ids_one_side = face_tex_ids[:num_faces//2]  # assuming symmetric faces are appended right after the original ones
        for i, face in enumerate(face_tex_ids_one_side):
            vert_uv = verts_tex_uv[face]  # 3x2
            # color = cmap(i%num_colors)
            color = cmap(np.random.randint(num_colors))
            t = plt.Polygon(vert_uv, facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(t)
        ## draw arrow
        ax.arrow(0.85, 0.5, -0.7, 0., length_includes_head=True, width=0.03, head_width=0.15, overhang=0.2, color='white')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        face_triangle_tex_map = torch.FloatTensor(fig_to_img(fig, dpi, im_size))
        plt.close()

        ## draw seam
        fig = plt.figure(figsize=(8,8), dpi=dpi, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        for i, face in enumerate(face_tex_ids_one_side):
            vert_uv = verts_tex_uv[face]  # 3x2
            vert_on_seam = ((vert_uv-0.5)**2).sum(1)**0.5 > 0.47
            if vert_on_seam.sum() == 2:
                ax.plot(*vert_uv[vert_on_seam].t(), color='black', linewidth=10)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        seam_mask = torch.FloatTensor(fig_to_img(fig, dpi, im_size))
        plt.close()
        seam_mask = (seam_mask[:,:,:1] < 0.1).float()

        red = torch.FloatTensor([1,0,0]).view(1,1,3)
        face_triangle_tex_map = seam_mask * red + (1-seam_mask) * face_triangle_tex_map
        aux['face_triangle_tex_map'] = face_triangle_tex_map.to(device)
        aux['seam_mask'] = seam_mask.to(device)

    return sym_sph_mesh.to(device), aux
