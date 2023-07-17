import math
import torch
import torch.nn as nn
import pytorch3d
from . import geometry
from einops import rearrange
import itertools


def _joints_to_bones(joints, bones_idxs):
    bones = []
    for a, b in bones_idxs:
        bones += [torch.stack([joints[:, :, a], joints[:, :, b]], dim=2)]
    bones = torch.stack(bones, dim=2)
    return bones


def _estimate_vertices_to_bones(bones_pred, seq_shape_pred, temperature=1):
    vertices_to_bones = []
    for i in range(bones_pred.shape[2]):
        vertices_to_bones += [geometry.line_segment_distance(bones_pred[:, :, i, 0], bones_pred[:, :, i, 1], seq_shape_pred)]
    vertices_to_bones = nn.functional.softmax(1 / torch.stack(vertices_to_bones) / temperature, dim=0)
    return vertices_to_bones


def _estimate_bones(seq_shape, n_bones, resample=False, n_leg_bones=0, n_legs=4, body_bones_type='max_distance'):
    # preprocess shape
    if resample:
        b, _, n, _ = seq_shape.shape
        seq_shape = geometry.sample_farthest_points(rearrange(seq_shape, 'b f n d -> (b f) d n'), n // 4)
        seq_shape = rearrange(seq_shape, '(b f) d n -> b f n d', b=b)

    # find two farthest points
    # x is the symmetry plane, ignore it
    dists = torch.norm(seq_shape[:, :, None, :, 1:] - seq_shape[:, :, :, None, 1:], dim=-1)
    d = dists.shape[-1]
    indices_flat = rearrange(dists, 'b f d1 d2 -> b f (d1 d2)').argmax(2)
    indices = torch.cat([(indices_flat // d)[..., None], (indices_flat % d)[..., None]], dim=2)
    indices_gather = indices[..., None].repeat(1, 1, 1, 3)
    points = seq_shape.gather(2, indices_gather)
    # fix the points order along z axis
    z_coordinate = points[:, :, :, 2]
    front = z_coordinate < 0
    point_a = rearrange(points[~front], '(b f) d -> b f d', b=seq_shape.shape[0])
    point_b = rearrange(points[front], '(b f) d -> b f d', b=seq_shape.shape[0])

    if body_bones_type == 'max_distance':
        # find two farthest points
        # x is the symmetry plane, ignore it
        dists = torch.norm(seq_shape[:, :, None, :, 1:] - seq_shape[:, :, :, None, 1:], dim=-1)
        d = dists.shape[-1]
        indices_flat = rearrange(dists, 'b f d1 d2 -> b f (d1 d2)').argmax(2)
        indices = torch.cat([(indices_flat // d)[..., None], (indices_flat % d)[..., None]], dim=2)
        indices_gather = indices[..., None].repeat(1, 1, 1, 3)
        points = seq_shape.gather(2, indices_gather)
        # fix the points order along z axis
        z_coordinate = points[:, :, :, 2]
        front = z_coordinate < 0
        point_a = rearrange(points[~front], '(b f) d -> b f d', b=seq_shape.shape[0])
        point_b = rearrange(points[front], '(b f) d -> b f d', b=seq_shape.shape[0])
    
    elif body_bones_type == 'z_minmax':
        indices = seq_shape[:, :, :, 2].argmax(2)
        indices_gather = indices[..., None, None].repeat(1, 1, 1, 3)
        point_a = seq_shape.gather(2, indices_gather).squeeze(2)
        indices = seq_shape[:, :, :, 2].argmin(2)
        indices_gather = indices[..., None, None].repeat(1, 1, 1, 3)
        point_b = seq_shape.gather(2, indices_gather).squeeze(2)
    
    elif body_bones_type == 'z_minmax_y+':
        mid_point = seq_shape.mean(2)
        seq_shape_pos_y_mask = (seq_shape[:, :, :, 1] > mid_point[:, :, None, 1]).float()  # y higher than midpoint
        seq_shape_z = seq_shape[:, :, :, 2] * seq_shape_pos_y_mask + (-1e6) * (1 - seq_shape_pos_y_mask)
        indices = seq_shape_z.argmax(2)
        indices_gather = indices[..., None, None].repeat(1, 1, 1, 3)
        point_a = seq_shape.gather(2, indices_gather).squeeze(2)
        seq_shape_z = seq_shape[:, :, :, 2] * seq_shape_pos_y_mask + 1e6 * (1 - seq_shape_pos_y_mask)
        indices = seq_shape_z.argmin(2)
        indices_gather = indices[..., None, None].repeat(1, 1, 1, 3)
        point_b = seq_shape.gather(2, indices_gather).squeeze(2)
    
    else:
        raise ValueError(f'Wrong body_bones_type: {body_bones_type}')

    # place points on the symmetry axis
    point_a[..., 0] = 0
    point_b[..., 0] = 0

    mid_point = seq_shape.mean(2)
    # place points on the symmetry axis
    mid_point[..., 0] = 0

    assert n_bones % 2 == 0
    n_joints = n_bones + 1
    blend = torch.linspace(0., 1., math.ceil(n_joints / 2), device=point_a.device)[None, None, :, None]
    joints_a = point_a[:, :, None] * (1 - blend) + mid_point[:, :, None] * blend
    # point_a to mid_point
    joints_b = point_b[:, :, None] * blend + mid_point[:, :, None] * (1 - blend)
    # mid_point to point_b
    joints = torch.cat([joints_a[:, :, :-1], joints_b], 2)

    # build bones and kinematic chain starting from leaf bones
    half_n_bones = n_bones // 2
    bones_to_joints = []
    kinematic_chain = []
    bone_idx = 0
    # bones from point_a to mid_point
    dependent_bones = []
    for i in range(half_n_bones):
        bones_to_joints += [(i + 1, i)]
        kinematic_chain = [(bone_idx, dependent_bones)] + kinematic_chain
        dependent_bones = dependent_bones + [bone_idx]
        bone_idx += 1
    # bones from mid_point to n_bones
    dependent_bones = []
    for i in range(n_bones - 1, half_n_bones - 1, -1):
        bones_to_joints += [(i, i + 1)]
        kinematic_chain = [(bone_idx, dependent_bones)] + kinematic_chain
        dependent_bones = dependent_bones + [bone_idx]
        bone_idx += 1

    bones_pred = _joints_to_bones(joints, bones_to_joints)

    if n_leg_bones > 0:
        assert n_legs == 4
        # attach four legs
        # y, z is symetry plain
        # y axis is up
        #
        # top down view:
        #
        #          z
        #          ⌃
        #      3   |   0
        # x <------|--------
        #      2   |   1
        #          |
        #
        # find a point with the lowest y in each quadrant
        quadrant0 = torch.logical_and(seq_shape[..., 0] > 0, seq_shape[..., 2] > 0)
        quadrant1 = torch.logical_and(seq_shape[..., 0] > 0, seq_shape[..., 2] < 0)
        quadrant2 = torch.logical_and(seq_shape[..., 0] < 0, seq_shape[..., 2] < 0)
        quadrant3 = torch.logical_and(seq_shape[..., 0] < 0, seq_shape[..., 2] > 0)

        def build_kinematic_chain(n_bones, start_bone_idx):
            # build bones and kinematic chain starting from leaf bone (body joint)
            bones_to_joints = []
            kinematic_chain = []
            bone_idx = start_bone_idx
            # bones from point_a to mid_point
            dependent_bones = []
            for i in range(n_bones):
                bones_to_joints += [(i + 1, i)]
                kinematic_chain = [(bone_idx, dependent_bones)] + kinematic_chain
                dependent_bones = dependent_bones + [bone_idx]
                bone_idx += 1
            return bones_to_joints, kinematic_chain

        def find_leg_in_quadrant(quadrant, n_bones):
            all_joints = torch.zeros([seq_shape.shape[0], seq_shape.shape[1], n_bones + 1, 3], dtype=seq_shape.dtype, device=seq_shape.device)
            for b in range(seq_shape.shape[0]):
                for f in range(seq_shape.shape[1]):
                    # find a point with the lowest y
                    quadrant_points = seq_shape[b, f][quadrant[b, f]]
                    idx = torch.argmin(quadrant_points[:, 1])
                    foot = quadrant_points[idx]

                    # find closest point on the body bone structure
                    idx = torch.argmin(torch.norm(bones_pred[b, f, :, 0] - foot[None], dim=1))
                    body_joint = bones_pred[b, f, idx, 0]

                    # create bone structure from the foot to the body joint
                    blend = torch.linspace(0., 1., n_bones + 1, device=seq_shape.device)[:, None]
                    joints = foot[None] * (1 - blend) + body_joint[None] * blend
                    all_joints[b, f] = joints
            return all_joints

        quadrants = [quadrant0, quadrant1, quadrant2, quadrant3]
        start_bone_idx = n_bones
        all_leg_bones = []
        all_leg_kinematic_chain = []
        for quadrant in quadrants:
            leg_joints = find_leg_in_quadrant(quadrant, n_leg_bones)
            leg_bones_to_joints, leg_kinematic_chain = build_kinematic_chain(n_leg_bones, start_bone_idx=start_bone_idx)
            start_bone_idx += n_leg_bones
            leg_bones = _joints_to_bones(leg_joints, leg_bones_to_joints)
            all_leg_bones += [leg_bones]
            all_leg_kinematic_chain += [leg_kinematic_chain]

        all_bones = [bones_pred] + all_leg_bones
        all_bones = torch.cat(all_bones, dim=2)
        all_kinematic_chain = list(itertools.chain(kinematic_chain, *all_leg_kinematic_chain))
    else:
        all_bones = bones_pred
        all_kinematic_chain = kinematic_chain

    return all_bones.detach(), all_kinematic_chain


def _estimate_bone_rotation(b):
    """
    (0, 0, 1) = matmul(b, R^(-1))

    returns R
    """
    up = torch.FloatTensor([0,1,0]).to(b.device).view(1,3)
    vec_forward = nn.functional.normalize(b, p=2, dim=-1)  # x right, y up, z forward
    vec_right = up.expand_as(vec_forward).cross(vec_forward, dim=-1)
    vec_right = nn.functional.normalize(vec_right, p=2, dim=-1)
    vec_up = vec_forward.cross(vec_right, dim=-1)
    vec_up = nn.functional.normalize(vec_up, p=2, dim=-1)
    rot_mat = torch.stack([vec_right, vec_up, vec_forward], 1)
    return rot_mat


def children_to_parents(kinematic_tree):
    """
    converts list [(bone1, [children1, ...]), (bone2, [children1, ...]), ...] to [(bone1, [parent1, ...]), ....]
    """
    parents = []
    for bone_id, _ in kinematic_tree:
        # establish a kinematic chain with current bone as the leaf bone
        parents_ids = [parent_id for parent_id, children in kinematic_tree if bone_id in children]
        parents += [(bone_id, parents_ids)]
    return parents


def skinning(seq_shape, arti_params, n_bones, n_legs=0, n_leg_bones=0, body_bones_type='max_distance', output_posed_bones=False, temperature=1, static_root_bones=False):
    b, f, _, _ = seq_shape.shape

    bones_estim, kinematic_tree = _estimate_bones(
        seq_shape.detach(), n_bones, n_leg_bones=n_leg_bones, n_legs=n_legs, body_bones_type=body_bones_type)

    bones_pred = bones_estim

    # associate vertices to bones
    vertices_to_bones = _estimate_vertices_to_bones(bones_pred, seq_shape, temperature=temperature)

    rots_pred = arti_params[:, :, :n_bones + n_leg_bones * n_legs]

    if static_root_bones:
        bone_parents = children_to_parents(kinematic_tree)
        root_bones = [bone for bone, parents in bone_parents if not parents]
        for root_bone in root_bones:
            rots_pred[:, :, root_bone] = rots_pred[:, :, root_bone] * 0

    # Rotate vertices based on bone assignments
    frame_shape_pred = []
    if output_posed_bones:
        posed_bones = bones_pred.clone()
    # Go through each bone
    for bone_id, _ in kinematic_tree:
        # establish a kinematic chain with current bone as the leaf bone
        parents_ids = [parent_id for parent_id, children in kinematic_tree if bone_id in children]
        chain_ids = parents_ids + [bone_id]
        # chain from leaf to parents
        chain_ids = chain_ids[::-1]

        device = seq_shape.device

        # go through the kinematic chain from leaf to root and compose transformation
        tsf = None
        for i in chain_ids:
            # establish transformation
            rest_joint = bones_pred[:, :, i, 0]
            rest_bone_vector = bones_pred[:, :, i, 1] - bones_pred[:, :, i, 0]
            rest_bone_rot = _estimate_bone_rotation(rest_bone_vector.view(-1, 3))

            # transform to the bone local frame
            tsf_translate = pytorch3d.transforms.Translate(-rest_joint.view(-1, 3), device=device)
            if tsf is None:
                tsf = tsf_translate
            else:
                tsf = tsf.compose(tsf_translate)
            tsf = tsf.compose(pytorch3d.transforms.Rotate(rest_bone_rot.transpose(-2, -1), device=device))

            # rotate the mesh in the bone local frame
            rot_pred = rots_pred[:, :, i]
            rot_pred_mat = pytorch3d.transforms.euler_angles_to_matrix(rot_pred.view(-1, 3), convention='XYZ')
            tsf = tsf.compose(pytorch3d.transforms.Rotate(rot_pred_mat, device=device))

            # transform to the world frame
            tsf = tsf.compose(pytorch3d.transforms.Rotate(rest_bone_rot, device=device))
            tsf = tsf.compose(pytorch3d.transforms.Translate(rest_joint.view(-1, 3), device=device))

        # transform vertices
        seq_shape_bone = tsf.transform_points(rearrange(seq_shape, 'b f ... -> (b f) ...'))
        seq_shape_bone = rearrange(seq_shape_bone, '(b f) ... -> b f ...', b=b, f=f)

        if output_posed_bones:
            posed_bones[:, :, bone_id] = rearrange(tsf.transform_points(rearrange(posed_bones[:, :, bone_id], 'b f ... -> (b f) ...')), '(b f) ... -> b f ...', b=b, f=f)

        # transform mesh with weights
        frame_shape_pred += [vertices_to_bones[bone_id, ..., None] * seq_shape_bone]

    frame_shape_pred = sum(frame_shape_pred)

    aux = {}
    aux['bones_pred'] = bones_pred
    aux['vertices_to_bones'] = vertices_to_bones
    if output_posed_bones:
        aux['posed_bones'] = posed_bones

    return frame_shape_pred, aux
