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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--which", type=str, choices=["first", "second", "union"],
    )
    args=parser.parse_args()
    render_plant(args.which)

