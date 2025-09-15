"""
Sample code to render a cow.

Usage:
    python -m starter.render_mesh --image_size 256 --output_path images/cow_render.jpg
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import imageio

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def render_cow(
    cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None, mode='static'
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    
    if mode=='static':
        # Prepare the camera:
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 3]]), fov=60, device=device
        )

        # Place a point light in front of the cow.
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        # The .cpu moves the tensor to GPU (if needed).
        return rend
    
    if mode=='dynamic':
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="images/cow_render.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--render_style", type=str, default='static', help="Decided what type of rendering you want; choose between static or dynamic")
    args = parser.parse_args()

    if args.render_style=='static':
        image = render_cow(cow_path=args.cow_path, image_size=args.image_size, mode='static')
        plt.imsave(args.output_path, image)
    elif args.render_style=='dynamic':
        images = render_cow(cow_path=args.cow_path, image_size=args.image_size, mode='dynamic')  # List of images [(H, W, 3)]
        duration = 2000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
        imageio.mimsave(args.output_path, images, duration=duration,loop=0)
    else:
        print("Invalid render style - killing program")
