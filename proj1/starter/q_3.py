"""
Usage:
    python -m starter.q_3
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import imageio

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def retexture_cow(
    cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None
):
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

    color_1, color_2= torch.tensor([1,0,0]), torch.tensor([0,1,0])
    z_min, z_max=torch.min(vertices[:,:,2]), torch.max(vertices[:,:,2])

    alpha=(vertices[0,:,2]-z_min)/(z_max-z_min)
    colors=alpha.unsqueeze(1)*color_2+ (1-alpha).unsqueeze(1)*color_1
    textures=colors.unsqueeze(0)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    
    images=[]
    for azimuth in range(0,360,10):
        R,T=pytorch3d.renderer.cameras.look_at_view_transform(3, 0, azimuth)

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )

        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

        rend = renderer(mesh, cameras=cameras, lights=lights)
        # Had to do the unit thing because i was getting an error from mimsave 
        # Google told me that mimsave does not handle float directly  
        images.append((rend.cpu().numpy()[0,..., :3] * 255).astype('uint8'))

    return images


if __name__ == "__main__":

    images = retexture_cow()  # List of images [(H, W, 3)]
    duration = 2000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave("my_images/cow_retextured.gif", images, duration=duration,loop=0)
