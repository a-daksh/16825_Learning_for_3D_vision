import argparse
import pickle

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
import imageio

from starter.utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image

def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def render_point_cloud(
    verts, rgb,
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )

    points=verts.unsqueeze(0).to(device)
    features=rgb.unsqueeze(0).to(device)
    point_cloud = pytorch3d.structures.Pointclouds(points=points, features=features)

    images=[]
    for azimuth in range(0,360,10):
        R, T = pytorch3d.renderer.look_at_view_transform(6, 0, azimuth)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

        rend = renderer(point_cloud, cameras=cameras)
        images.append((rend.cpu().numpy()[0,..., :3] * 255).astype('uint8'))

    return images

def render_plant(which):
    data=load_rgbd_data()
    data={key: torch.from_numpy(value) if "cameras" not in key else value for key, value in data.items()}

    if which == 'first':    
        points, rgb = unproject_depth_image(image=data["rgb1"], mask=data["mask1"], depth=data["depth1"], camera=data["cameras1"] )
    elif which == 'second':
        points, rgb = unproject_depth_image(image=data["rgb2"], mask=data["mask2"], depth=data["depth2"], camera=data["cameras2"] )
    elif which == 'union':
        points1, rgb1 = unproject_depth_image(image=data["rgb1"], mask=data["mask1"], depth=data["depth1"], camera=data["cameras1"] )
        points2, rgb2 = unproject_depth_image(image=data["rgb2"], mask=data["mask2"], depth=data["depth2"], camera=data["cameras2"] )
        points=torch.cat([points1,points2])
        rgb=torch.cat([rgb1,rgb2])

    images = render_point_cloud(points, rgb)
    duration = 2000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(f"my_images/plant_{which}_render.gif", images, duration=duration,loop=0)

def render_torus(image_size=256, num_samples=200, device=None, R=2,r=0.5):
    """
    Renders a trus using parametric sampling. Parametric equation given at
    https://en.wikipedia.org/wiki/Torus#Geometry
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2* np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = (R + r*torch.sin(Theta)) * torch.cos(Phi)
    y = (R + r*torch.sin(Theta)) * torch.sin(Phi)
    z = r*torch.cos(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    renderer = get_points_renderer(image_size=image_size, device=device)
    
    images=[]
    for azimuth in range(0,360,10):
        R, T = pytorch3d.renderer.look_at_view_transform(6, 0, azimuth)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

        rend = renderer(torus_point_cloud, cameras=cameras)
        images.append((rend.cpu().numpy()[0,..., :3] * 255).astype('uint8'))
    
    duration = 2000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(f"my_images/torus.gif", images, duration=duration,loop=0)

def render_spiral(image_size=256, num_samples=100, device=None, R=5):
    """
    Renders a spiral using parametric sampling.
    """

    if device is None:
        device = get_device()

    a = 1.0 
    b = 0.1
    c = 0.5

    theta = torch.linspace(0, 10 * np.pi, 500) 

    r = a * torch.exp(b * theta)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    z = c * theta

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    torus_point_cloud = pytorch3d.structures.Pointclouds(points=[points], features=[color]).to(device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    
    images=[]
    for azimuth in range(0,360,10):
        R, T = pytorch3d.renderer.look_at_view_transform(10, 0, azimuth)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

        rend = renderer(torus_point_cloud, cameras=cameras)
        images.append((rend.cpu().numpy()[0,..., :3] * 255).astype('uint8'))
    
    duration = 2000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(f"my_images/spiral.gif", images, duration=duration,loop=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", type=str, choices=["point_cloud", "parametric", "implicit"])
    parser.add_argument("--which_plant", type=str, choices=["first", "second", "union"])
    parser.add_argument("--which_parameter", type=str, choices=["torus", "spiral"])
    args=parser.parse_args()
    
    if args.render=='point_cloud':
        render_plant(args.which_plant)
    if args.render=='parametric':
        if args.which_parameter=='torus':
            render_torus()
        elif args.which_parameter=='spiral':
            render_spiral()
        else:
            render_torus()
            render_spiral()
