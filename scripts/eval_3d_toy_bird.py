import os
import os.path as osp
from glob import glob
import numpy as np
import trimesh
from scipy.spatial import cKDTree as KDTree


result_folders = {
    'ours_release_016': 'results/toy_birds/our_final/final_test_toy_birds_016',

    'ours_ab_nosym_010': 'results/toy_birds/our_ablations/ab_nosym_010',
    'ours_ab_nozflip_010': 'results/toy_birds/our_ablations/ab_nozflip_010',
    'ours_ab_noflow_010': 'results/toy_birds/our_ablations/ab_noflow_010',
    'ours_ab_novideo_020': 'results/toy_birds/our_ablations/ab_novideo_020',
    'ours_ab_noprior_008': 'results/toy_birds/our_ablations/ab_noprior_008',

    'cmr_finetune': 'results/toy_birds/other_methods/cmr_finetune',
    'cmr_no_kp': 'results/toy_birds/other_methods/cmr_no_kp',
    'ucmr_finetune': 'results/toy_birds/other_methods/ucmr_finetune',
    'ucmr_no_temp': 'results/toy_birds/other_methods/ucmr_no_temp',
    'umr_finetune': 'results/toy_birds/other_methods/umr_finetune',
    'umr_no_scops': 'results/toy_birds/other_methods/umr_no_scops',
    'umr_scratch_scops': 'results/toy_birds/other_methods/umr_scratch_scops',
    'vmr_finetune': 'results/toy_birds/other_methods/vmr_finetune',
    'vmr_no_kp_10k': 'results/toy_birds/other_methods/vmr_no_kp_10k',
    'vmr_no_kp_30k': 'results/toy_birds/other_methods/vmr_no_kp_30k',
}

method = 'ours_release_016'
result_folder = result_folders[method]
save_scaled_scans = False
if method.startswith('ours_'):
    suffix = '.obj'
    result_list = sorted(glob(osp.join(result_folder, '*', '*'+suffix)))
else:
    suffix = '.npz'
    result_list = sorted(glob(osp.join(result_folder, '*'+suffix)))

out_folder = osp.join('result/eval/toy_birds', method)
os.makedirs(out_folder)

## alignment parameters
target_volume = 900  # roughly corresponds to a witdth of 10cm
num_points_align = 1000
max_iterations = 100
cost_threshold = 0
num_points_chamfer = 30000
num_scans = 23
scan_folder = 'data/3d_toy_birds/3d_scans'
scan_list = [osp.join(scan_folder, f'bird{(i+1):03d}_scan.obj') for i in range(num_scans)]


def sample_surface_point(mesh, num_points, even=False):
    if even:
        sample_points, indexes = trimesh.sample.sample_surface_even(mesh, count=num_points)
        while len(sample_points) < num_points:
            more_sample_points, indexes = trimesh.sample.sample_surface_even(mesh, count=num_points)
            sample_points = np.concatenate([sample_points, more_sample_points], axis=0)
    else:
        sample_points, indexes = trimesh.sample.sample_surface(mesh, count=num_points)
    return sample_points[:num_points]


def write_result(fpath, mat, cost, cfdist):
    scale, shear, angles, trans, persp = trimesh.transformations.decompose_matrix(mat)
    with open(fpath, 'w') as f:
        f.write(f'chamfer distance: total: {cfdist[0]}, to recon: {cfdist[1]}, to scan: {cfdist[2]}\n')
        f.write(f'cost: {cost}\n')
        f.write(f'transformation matrix:\n{transform_mat}\n')
        f.write(f'scale:{scale} \nshear: {shear}\nrotation angle: {angles}\ntranslation: {trans}\nperspective transform: {persp}')


def load_mesh(fpath):
    if fpath.endswith('.obj'):
        mesh = trimesh.load_mesh(fpath)
    elif fpath.endswith('.npz'):
        mesh_npz = np.load(fpath)
        verts = mesh_npz['verts']
        faces = mesh_npz['faces']
        faces = np.concatenate((faces, faces[:, list(reversed(range(faces.shape[-1])))]), axis=0)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return mesh


# https://github.com/facebookresearch/DeepSDF/blob/main/deep_sdf/metrics/chamfer.py
def compute_trimesh_chamfer(gt_points, gen_mesh, num_mesh_samples=30000):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.
    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)
    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)
    """
    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]

    # only need numpy array of points
    gt_points_np = gt_points.vertices

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer, gen_to_gt_chamfer


def calculate_scale(mesh, target_volume, method='volume'):
    if method == 'bounding_box':
        width, height, length = mesh.extents
        bounding_box_volume = (width * height * length)
        scale = (target_volume / bounding_box_volume)**(1/3)
    elif method == 'volume':
        voxel_length = mesh.extents.min() /100
        voxel = mesh.voxelized(voxel_length).fill()
        voxel_volume = voxel.volume
        scale = (target_volume / voxel_volume)**(1/3)
    return scale


all_cfdists = []
for i, result_fpath in enumerate(result_list):
    ## load scan
    if method.startswith('ours_'):
        bird_id = i // 15
        scene_id = (i % 15) // 5
        view_id = i % 5
        fname = f'bird{bird_id+1:03d}_scene{scene_id+1:02d}_view{view_id+1:02d}_im1_01.obj'
    else:
        fname = osp.basename(result_fpath)
        bird_id = int(fname[4:7]) -1
    mesh_scan = load_mesh(scan_list[bird_id])
    pc_scan = trimesh.load(scan_list[bird_id].replace('.obj', '.ply'))

    ## load result
    mesh_result = load_mesh(result_fpath)

    ## rough canonical pose alignment (simply from visual inspection, dosen't need to be accurate)
    if method.startswith('ours_ab_nosym_'):
        rot_mat = None
    elif method.startswith('ours_ab_noprior_'):
        rot_mat1 = trimesh.transformations.euler_matrix(0, np.pi, 0, 'sxyz')
        rot_mat2 = trimesh.transformations.euler_matrix(-30/180*np.pi, 0, 0, 'sxyz')
        rot_mat = trimesh.transformations.concatenate_matrices(rot_mat2, rot_mat1)
    elif method.startswith('ours_ab_novideo_'):
        rot_mat = trimesh.transformations.euler_matrix(0, np.pi, 0, 'sxyz')
    elif method.startswith('ours_ab_nozflip_'):
        rot_mat = trimesh.transformations.euler_matrix(30/180*np.pi, 0, 0, 'sxyz')
    elif method.startswith('ours_ab_noflow_'):
        rot_mat = trimesh.transformations.euler_matrix(0, np.pi, 0, 'sxyz')
    elif method.startswith('ours_'):
        rot_mat = trimesh.transformations.euler_matrix(0, np.pi, 0, 'sxyz')
    elif method.startswith('ucmr_no_temp'):
        rot_mat = trimesh.transformations.euler_matrix(np.pi/2, 0, 0, 'sxyz')
    elif method.startswith('cmr_') or method.startswith('ucmr_') or method.startswith('vmr_'):
        rot_mat = trimesh.transformations.euler_matrix(-np.pi/2, 0, 0, 'sxyz')
    elif method.startswith('umr_scratch_scops'):
        rot_mat1 = trimesh.transformations.euler_matrix(0, -np.pi/2, 0, 'sxyz')
        rot_mat2 = trimesh.transformations.euler_matrix(0, 0, np.pi/2, 'sxyz')
        rot_mat = trimesh.transformations.concatenate_matrices(rot_mat2, rot_mat1)
    elif method.startswith('umr_no_scops'):
        rot_mat1 = trimesh.transformations.euler_matrix(0, np.pi/2, 0, 'sxyz')
        rot_mat2 = trimesh.transformations.euler_matrix(0, 0, -np.pi/2, 'sxyz')
        rot_mat = trimesh.transformations.concatenate_matrices(rot_mat2, rot_mat1)
    elif method.startswith('umr_'):
        rot_mat1 = trimesh.transformations.euler_matrix(0, -np.pi/2, 0, 'sxyz')
        rot_mat2 = trimesh.transformations.euler_matrix(0, 0, np.pi/2, 'sxyz')
        rot_mat = trimesh.transformations.concatenate_matrices(rot_mat2, rot_mat1)
    else:
        raise NotImplementedError('unknown method')

    if rot_mat is not None:
        mesh_result.apply_transform(rot_mat)
    scale = calculate_scale(mesh_result, target_volume, method='volume')
    scale_mat = trimesh.transformations.scale_matrix(scale)
    mesh_result.apply_transform(scale_mat)

    ## icp alignment
    sample_points = sample_surface_point(mesh_result, num_points_align, even=True)
    transform_mat, pts_transformed, cost = trimesh.registration.icp(sample_points, mesh_scan, threshold=cost_threshold, max_iterations=max_iterations, reflection=False, translation=True, scale=False)

    ## export aligned mesh
    mesh_result.apply_transform(transform_mat)
    mesh_result.export(osp.join(out_folder, fname.replace(suffix, '_aligned.obj')))

    ## compute chamfer distances
    gt_to_gen_chamfer, gen_to_gt_chamfer = compute_trimesh_chamfer(pc_scan, mesh_result, num_points_chamfer)
    bidir_chamfer = (gt_to_gen_chamfer + gen_to_gt_chamfer) / 2.
    cfdist = [bidir_chamfer, gt_to_gen_chamfer, gen_to_gt_chamfer]
    all_cfdists += [cfdist]

    print(f'{fname} - align_cost: {cost:.6f}, chamfer_dist: {bidir_chamfer:.6f}')
    write_result(osp.join(out_folder, fname.replace(suffix, '_scores.txt')), transform_mat, cost, cfdist)

all_cfdists = np.array(all_cfdists)
score_mean = all_cfdists.mean(0)
score_std = all_cfdists.std(0)
header = f'mean Chamfer distance: {score_mean[0]:.6f}+-{score_std[0]:.6f}\tto recon: {score_mean[1]:.6f}+-{score_std[1]:.6f}\tto scan: {score_mean[2]:.6f}+-{score_std[2]:.6f}'
np.savetxt(osp.join(out_folder, 'all_scores.txt'), np.array(all_cfdists), fmt='%.6f', delimiter='\t', header=header)
print(method + ' - ' + header)
