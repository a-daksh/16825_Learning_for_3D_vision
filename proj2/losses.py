import torch
import torch.nn.functional as F
# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# Clamp to avoid numerical instability with log(0) or log(1)
	voxel_src_clamped = torch.clamp(voxel_src, 1e-7, 1 - 1e-7)
	critertia = F.binary_cross_entropy
	loss = critertia(voxel_src_clamped, voxel_tgt)
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
	# implement laplacian smoothening loss
	verts=mesh_src.verts_list()[0]
	k=6
	
	dist=verts.unsqueeze(1) - verts.unsqueeze(0)
	dist=(dist.pow(2).sum(dim=2)).pow(0.5)
	
	nearest_neighs=(torch.topk(dist, k,dim=1,largest=False).indices)[:,1:]   # N X K-1
	neighbors=verts[nearest_neighs]
	centre=neighbors.mean(1)

	temp=((verts-centre)**2).sum(1)
	loss_laplacian=(temp**0.5).mean(0)

	return loss_laplacian