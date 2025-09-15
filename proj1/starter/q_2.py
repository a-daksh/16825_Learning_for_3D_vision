"""
Usage:
    python -m starter.q_2 --image_size 256 --render_style dynamic --shape tetrahedron
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import imageio

from starter.utils import get_device, get_mesh_renderer


def render_shape(shape, image_size=256, device=None, mode='static'):
    if device is None:
        device = get_device()

    renderer = get_mesh_renderer(image_size=image_size)
    
    if shape=='tetrahedron':
        vertices=torch.tensor([
            [0,0,0],
            [2,0,0],
            [1,0,2],
            [1,2,1]], dtype=torch.float32)
        faces=torch.tensor([
            [0,1,2],
            [2,3,0],
            [2,1,3],
            [1,0,3]], dtype=torch.int64)
        vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
        faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

        colors = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            ], dtype=torch.float32).unsqueeze(0) 
        
    elif shape=='cube':
        vertices=torch.tensor([
            [0,0,0],
            [1,0,0],
            [1,0,1],
            [0,0,1],
            [0,1,0],
            [1,1,0],
            [1,1,1],
            [0,1,1]], dtype=torch.float32)
        faces=torch.tensor([
            [0,4,5],
            [0,1,5],
            [2,1,5],
            [2,6,5],
            [3,2,6],
            [3,7,6],
            [3,0,4],
            [3,7,4],
            [7,4,5],
            [7,6,5],
            [3,0,1],
            [3,2,1]], dtype=torch.int64)
        vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
        faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

        colors = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            ], dtype=torch.float32).unsqueeze(0) 
    
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(colors),
    )
    mesh = mesh.to(device)
    
    if mode=='static':
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, ]]), fov=60, device=device
        )

        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        return rend
    
    if mode=='dynamic':
        images=[]
        for azimuth in range(0,360,10):
            R,T=pytorch3d.renderer.cameras.look_at_view_transform(5, -10, azimuth)

            cameras = pytorch3d.renderer.FoVPerspectiveCameras(
                R=R, T=T, fov=60, device=device
            )

            lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

            rend = renderer(mesh, cameras=cameras, lights=lights)
            images.append((rend.cpu().numpy()[0,..., :3] * 255).astype('uint8'))

        return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--render_style", type=str, default='static', help="Decided what type of rendering you want; choose between static or dynamic")
    parser.add_argument("--shape", type=str, help="Choose shape between tetrahedron and cube")
    args = parser.parse_args()
    if args.shape!='tetrahedron' and args.shape!='cube':
        raise ValueError("Invalid shape")
    
    if args.render_style=='static':
        image = render_shape(shape=args.shape, image_size=args.image_size, mode='static')
        plt.imsave(f"images/{args.shape}.png", image)
    elif args.render_style=='dynamic':
        images = render_shape(shape=args.shape, image_size=args.image_size, mode='dynamic')  # List of images [(H, W, 3)]
        duration = 2000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
        imageio.mimsave(f"images/{args.shape}.gif", images, duration=duration,loop=0)
    else:
        raise ValueError("Invalid render style")
