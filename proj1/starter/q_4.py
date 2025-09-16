"""
Usage:
    python -m starter.q_4 
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer


def transform_cow(
    cow_path="data/cow_with_axis.obj",
    image_size=256,
    R_relative=[[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
    T_relative=[-3, 0, 3],
    device=None,
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)
    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()
    R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative

    renderer = get_mesh_renderer(image_size)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()

if __name__ == "__main__":
    plt.imsave("my_images/transform1.jpg",transform_cow(R_relative=[[0, 1, 0], [-1, 0, 0], [0, 0, 1]], T_relative=[0,0,0]))
    plt.imsave("my_images/transform2.jpg",transform_cow(R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], T_relative=[0,0,2]))
    plt.imsave("my_images/transform3.jpg",transform_cow(R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], T_relative=[0.5, -0.5, 0]))
    plt.imsave("my_images/transform4.jpg",transform_cow(R_relative=[[0, 0, 1], [0, 1, 0], [-1, 0, 0]], T_relative=[-3, 0, 3]))
