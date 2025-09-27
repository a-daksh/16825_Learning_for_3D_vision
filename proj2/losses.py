import torch
import torch.nn.functional as F
# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	voxel_src=voxel_src.clamp(1e-6, 1-1e-6)
	critertia = F.binary_cross_entropy
	loss = critertia(voxel_src, voxel_tgt) + critertia(1-voxel_src, 1-voxel_tgt)
	# implement some loss for binary voxel grids
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  [1, 5000, 3]
	# implement chamfer loss from scratch

	dist = (point_cloud_src[:, :, None, :] - point_cloud_tgt[:, None, :, :]).pow(2).sum(-1)
	min_src_tgt = dist.min(dim=2).values
	min_tgt_src = dist.min(dim=1).values

	loss_chamfer = (min_src_tgt.sum(dim=1) + min_tgt_src.sum(dim=1)).mean()	
	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
	# return loss_laplacian
	pass