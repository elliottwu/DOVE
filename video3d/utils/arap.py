import pytorch3d
import torch
from einops import rearrange
from torch._C import device


def edges_to_sparse_incidence(edges, num_vertices):
    num_edges = edges.shape[0]
    row_indexes = torch.arange(num_edges, dtype=torch.long, device=edges.device).repeat_interleave(2)
    col_indexes = edges.reshape(-1)
    indexes = torch.stack([row_indexes, col_indexes])
    values = torch.FloatTensor([1, -1]).to(edges.device).repeat(num_edges)
    return torch.sparse.FloatTensor(indexes, values, torch.Size([num_edges, num_vertices]))


def compute_svd_rotation(vertices_rest_pose, vertices_deformed_pose, incidence_mat):
    """
    Adapted from:
    https://github.com/kzhou23/shape_pose_disent/blob/a8017c405892c98f52fa9775327172633290b1d8/arap.py#L76

    vertices_rest_pose: B x V x D
    vertices_deformed_pose: B x V x D
    incidence_mat: E x V
    
    """
    batch_size, num_vertices, dimensions = vertices_rest_pose.shape
    vertices = torch.cat((vertices_rest_pose, vertices_deformed_pose), dim=0)
    # 2B x V x D -> V x (D x 2B)
    vertices = rearrange(vertices, 'a v d -> v (d a)')
    # E x V . V x (D x 2B) - > E x (D x 2B)
    edges = torch.sparse.mm(incidence_mat, vertices)
    edges = rearrange(edges, 'e (d a) -> a e d', d=dimensions)                 
    rest_edges, deformed_edges = torch.split(edges, batch_size, dim=0)

    edges_outer = torch.matmul(rest_edges[:, :, :, None], deformed_edges[:, :, None, :])
    edges_outer = rearrange(edges_outer, 'b e d1 d2 -> e (b d1 d2)')

    abs_incidence_mat = incidence_mat.clone()
    abs_incidence_mat._values()[:] = torch.abs(abs_incidence_mat._values())
    
    # transposed S
    S = torch.sparse.mm(abs_incidence_mat.t(), edges_outer)
    S = rearrange(S, 'v (b d1 d2) -> b v d2 d1', v=num_vertices, b=batch_size, d1=dimensions, d2=dimensions)
    
    # SVD on gpu is extremely slow! https://github.com/pytorch/pytorch/pull/48436
    device = S.device
    U, _, V = torch.svd(S.cpu())
    U = U.to(device)
    V = V.to(device)

    det_sign = torch.det(torch.matmul(U, V.transpose(-2, -1)))
    U = torch.cat([U[..., :-1], U[..., -1:] * det_sign[..., None, None]], axis=-1)

    rotations = torch.matmul(U, V.transpose(-2, -1))

    return rotations


def compute_rotation(vertices_rest_pose, vertices_deformed_pose, edges):
  """
  vertices_rest_pose: B x V x D
  vertices_deformed_pose: B x V x D
  edges: E x 2
  """
  num_vertices = vertices_rest_pose.shape[1]
  incidence_mat = edges_to_sparse_incidence(edges, num_vertices)
  rot = compute_svd_rotation(vertices_rest_pose, vertices_deformed_pose, incidence_mat)
  rot = pytorch3d.transforms.matrix_to_quaternion(rot)
  return rot


def quaternion_normalize(quaternion, eps=1e-12):
  """
  Adapted from tensorflow_graphics

  Normalizes a quaternion.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    quaternion:  A tensor of shape `[A1, ..., An, 4]`, where the last dimension
      represents a quaternion.
    eps: A lower bound value for the norm that defaults to 1e-12.
    name: A name for this op that defaults to "quaternion_normalize".

  Returns:
    A N-D tensor of shape `[?, ..., ?, 1]` where the quaternion elements have
    been normalized.

  Raises:
    ValueError: If the shape of `quaternion` is not supported.
  """
  return l2_normalize(quaternion, dim=-1, epsilon=eps)


def l2_normalize(x, dim=-1, epsilon=1e-12):
    square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
    x_inv_norm = torch.rsqrt(torch.clamp(square_sum, min=epsilon))
    return x * x_inv_norm
    

def arap_energy(vertices_rest_pose,
           vertices_deformed_pose,
           quaternions,
           edges,
           vertex_weight=None,
           edge_weight=None,
           conformal_energy=True,
           aggregate_loss=True):
  """
  Adapted from tensorflow_graphics

  Estimates an As Conformal As Possible (ACAP) fitting energy.
  For a given mesh in rest pose, this function evaluates a variant of the ACAP
  [1] fitting energy for a batch of deformed meshes. The vertex weights and edge
  weights are defined on the rest pose.
  The method implemented here is similar to [2], but with an added free variable
    capturing a scale factor per vertex.
  [1]: Yusuke Yoshiyasu, Wan-Chun Ma, Eiichi Yoshida, and Fumio Kanehiro.
  "As-Conformal-As-Possible Surface Registration." Computer Graphics Forum. Vol.
  33. No. 5. 2014.</br>
  [2]: Olga Sorkine, and Marc Alexa.
  "As-rigid-as-possible surface modeling". Symposium on Geometry Processing.
  Vol. 4. 2007.
  Note:
    In the description of the arguments, V corresponds to
      the number of vertices in the mesh, and E to the number of edges in this
      mesh.
  Note:
    In the following, A1 to An are optional batch dimensions.
  Args:
    vertices_rest_pose: A tensor of shape `[V, 3]` containing the position of
      all the vertices of the mesh in rest pose.
    vertices_deformed_pose: A tensor of shape `[A1, ..., An, V, 3]` containing
      the position of all the vertices of the mesh in deformed pose.
    quaternions: A tensor of shape `[A1, ..., An, V, 4]` defining a rigid
      transformation to apply to each vertex of the rest pose. See Section 2
      from [1] for further details.
    edges: A tensor of shape `[E, 2]` defining indices of vertices that are
      connected by an edge.
    vertex_weight: An optional tensor of shape `[V]` defining the weight
      associated with each vertex. Defaults to a tensor of ones.
    edge_weight: A tensor of shape `[E]` defining the weight of edges. Common
      choices for these weights include uniform weighting, and cotangent
      weights. Defaults to a tensor of ones.
    conformal_energy: A `bool` indicating whether each vertex is associated with
      a scale factor or not. If this parameter is True, scaling information must
      be encoded in the norm of `quaternions`. If this parameter is False, this
      function implements the energy described in [2].
    aggregate_loss: A `bool` defining whether the returned loss should be an
      aggregate measure. When True, the mean squared error is returned. When
      False, returns two losses for every edge of the mesh.
    name: A name for this op. Defaults to "as_conformal_as_possible_energy".
  Returns:
    When aggregate_loss is `True`, returns a tensor of shape `[A1, ..., An]`
    containing the ACAP energies. When aggregate_loss is `False`, returns a
    tensor of shape `[A1, ..., An, 2*E]` containing each term of the summation
    described in the equation 7 of [2].
  Raises:
    ValueError: if the shape of `vertices_rest_pose`, `vertices_deformed_pose`,
    `quaternions`, `edges`, `vertex_weight`, or `edge_weight` is not supported.
  """
  # with tf.compat.v1.name_scope(name, "as_conformal_as_possible_energy", [
  #     vertices_rest_pose, vertices_deformed_pose, quaternions, edges,
  #     conformal_energy, vertex_weight, edge_weight
  # ]):
  # vertices_rest_pose = tf.convert_to_tensor(value=vertices_rest_pose)
  # vertices_deformed_pose = tf.convert_to_tensor(value=vertices_deformed_pose)
  # quaternions = tf.convert_to_tensor(value=quaternions)
  # edges = tf.convert_to_tensor(value=edges)
  # if vertex_weight is not None:
  #   vertex_weight = tf.convert_to_tensor(value=vertex_weight)
  # if edge_weight is not None:
  #   edge_weight = tf.convert_to_tensor(value=edge_weight)

  # shape.check_static(
  #     tensor=vertices_rest_pose,
  #     tensor_name="vertices_rest_pose",
  #     has_rank=2,
  #     has_dim_equals=(-1, 3))
  # shape.check_static(
  #     tensor=vertices_deformed_pose,
  #     tensor_name="vertices_deformed_pose",
  #     has_rank_greater_than=1,
  #     has_dim_equals=(-1, 3))
  # shape.check_static(
  #     tensor=quaternions,
  #     tensor_name="quaternions",
  #     has_rank_greater_than=1,
  #     has_dim_equals=(-1, 4))
  # shape.compare_batch_dimensions(
  #     tensors=(vertices_deformed_pose, quaternions),
  #     last_axes=(-3, -3),
  #     broadcast_compatible=False)
  # shape.check_static(
  #     tensor=edges, tensor_name="edges", has_rank=2, has_dim_equals=(-1, 2))
  # tensors_with_vertices = [vertices_rest_pose,
  #                           vertices_deformed_pose,
  #                           quaternions]
  # names_with_vertices = ["vertices_rest_pose",
  #                         "vertices_deformed_pose",
  #                         "quaternions"]
  # axes_with_vertices = [-2, -2, -2]
  # if vertex_weight is not None:
  #   shape.check_static(
  #       tensor=vertex_weight, tensor_name="vertex_weight", has_rank=1)
  #   tensors_with_vertices.append(vertex_weight)
  #   names_with_vertices.append("vertex_weight")
  #   axes_with_vertices.append(0)
  # shape.compare_dimensions(
  #     tensors=tensors_with_vertices,
  #     axes=axes_with_vertices,
  #     tensor_names=names_with_vertices)
  # if edge_weight is not None:
  #   shape.check_static(
  #       tensor=edge_weight, tensor_name="edge_weight", has_rank=1)
  #   shape.compare_dimensions(
  #       tensors=(edges, edge_weight),
  #       axes=(0, 0),
  #       tensor_names=("edges", "edge_weight"))

  if not conformal_energy:
    quaternions = quaternion_normalize(quaternions)
  # Extracts the indices of vertices.
  indices_i, indices_j = torch.unbind(edges, dim=-1)
  # Extracts the vertices we need per term.
  vertices_i_rest = vertices_rest_pose[..., indices_i, :]
  vertices_j_rest = vertices_rest_pose[..., indices_j, :]
  vertices_i_deformed = vertices_deformed_pose[..., indices_i, :]
  vertices_j_deformed = vertices_deformed_pose[..., indices_j, :]
  # Extracts the weights we need per term.
  weights_shape = vertices_i_rest.shape[-2]
  if vertex_weight is not None:
    weight_i = vertex_weight[indices_i]
    weight_j = vertex_weight[indices_j]
  else:
    weight_i = weight_j = torch.ones(weights_shape, dtype=vertices_rest_pose.dtype, device=vertices_rest_pose.device)
  weight_i = weight_i[..., None]
  weight_j = weight_j[..., None]
  if edge_weight is not None:
    weight_ij = edge_weight
  else:
    weight_ij = torch.ones(weights_shape, dtype=vertices_rest_pose.dtype, device=vertices_rest_pose.device)
  weight_ij = weight_ij[..., None]
  # Extracts the rotation we need per term.
  quaternion_i = quaternions[..., indices_i, :]
  quaternion_j = quaternions[..., indices_j, :]
  # Computes the energy.
  deformed_ij = vertices_i_deformed - vertices_j_deformed
  rotated_rest_ij = pytorch3d.transforms.quaternion_apply(quaternion_i, (vertices_i_rest - vertices_j_rest))
  energy_ij = weight_i * weight_ij * (deformed_ij - rotated_rest_ij)
  deformed_ji = vertices_j_deformed - vertices_i_deformed
  rotated_rest_ji = pytorch3d.transforms.quaternion_apply(quaternion_j, (vertices_j_rest - vertices_i_rest))
  energy_ji = weight_j * weight_ij * (deformed_ji - rotated_rest_ji)
  energy_ij_squared = torch.sum(energy_ij ** 2, dim=-1)
  energy_ji_squared = torch.sum(energy_ji ** 2, dim=-1)
  if aggregate_loss:
    average_energy_ij = torch.mean(energy_ij_squared, dim=-1)
    average_energy_ji = torch.mean(energy_ji_squared, dim=-1)
    return (average_energy_ij + average_energy_ji) / 2.0
  return torch.cat((energy_ij_squared, energy_ji_squared), dim=-1)


def arap_loss(vertices_rest_pose, vertices_deformed_pose, edges):
    # squash batch dimensions
    vertices_rest_pose_shape = list(vertices_rest_pose.shape)
    vertices_deformed_pose_shape = list(vertices_deformed_pose.shape)
    vertices_rest_pose = vertices_rest_pose.reshape([-1] + vertices_rest_pose_shape[-2:])
    vertices_deformed_pose = vertices_deformed_pose.reshape([-1] + vertices_deformed_pose_shape[-2:])
    
    # try:
    quaternions = compute_rotation(vertices_rest_pose, vertices_deformed_pose, edges)
    # except RuntimeError:
    #   print('SVD did not converge')
    # batch_size = vertices_rest_pose.shape[0]
    # num_vertices = vertices_rest_pose.shape[-2]
    # quaternions = pytorch3d.transforms.matrix_to_quaternion(pytorch3d.transforms.euler_angles_to_matrix(torch.zeros([batch_size, num_vertices, 3], device=vertices_rest_pose.device), 'XYZ'))
    
    quaternions = quaternions.detach()

    energy = arap_energy(
      vertices_rest_pose,
      vertices_deformed_pose,
      quaternions,
      edges,
      aggregate_loss=True,
      conformal_energy=False)
    return energy.reshape(vertices_rest_pose_shape[:-2])
