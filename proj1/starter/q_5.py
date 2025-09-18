"""
python -m starter.q_5 --render parametric --which_parameter torus
python -m starter.q_5 --render parametric --which_parameter octahedron
python -m starter.q_5 --render implicit --which_implicit torus
python -m starter.q_5 --render implicit --which_implicit octahedron
"""
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

# 5.1
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

# 5.2
def render_torus(image_size=256, num_samples=200, device=None, R=2,r=0.5):
    """
    Renders a trus using parametric sampling. Parametric equation given at
    https://en.wikipedia.org/wiki/Torus#Geometry
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2* np.pi, num_samples)
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
    imageio.mimsave(f"my_images/torus_parametric.gif", images, duration=duration,loop=0)

def render_oct(image_size=256, num_samples=100, device=None, R=5):
    """
    Renders a octahedron using parametric sampling.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, np.pi, num_samples)
    Phi, Theta = torch.meshgrid(phi, theta)

    r_oct = 1.0 / (torch.abs(torch.sin(Theta) * torch.cos(Phi)) + 
                torch.abs(torch.sin(Theta) * torch.sin(Phi)) + 
                torch.abs(torch.cos(Theta)))

    x = r_oct * torch.sin(Theta) * torch.cos(Phi)
    y = r_oct * torch.sin(Theta) * torch.sin(Phi)
    z = r_oct * torch.cos(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    torus_point_cloud = pytorch3d.structures.Pointclouds(points=[points], features=[color]).to(device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    
    images=[]
    for azimuth in range(0,360,10):
        R, T = pytorch3d.renderer.look_at_view_transform(5, 10, azimuth)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

        rend = renderer(torus_point_cloud, cameras=cameras)
        images.append((rend.cpu().numpy()[0,..., :3] * 255).astype('uint8'))
    
    duration = 2000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(f"my_images/octahedron_parametric.gif", images, duration=duration,loop=0)

# 5.3
def render_torus_mesh(image_size=256, voxel_size=64, device=None, R=1, r=0.5):
    if device is None:
        device = get_device()
    
    min_value = -2.1
    max_value = 2.1
    
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = (torch.sqrt(X**2 + Y**2) - R)**2 + Z**2 - r**2

    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    
    # Vertex coordinates are indexed by array position, so we need to renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value

    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    
    images=[]
    for azim in range(0,360,10):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=5, elev=0, azim=azim)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    
        rend = renderer(mesh, cameras=cameras, lights=lights)
        images.append((rend.cpu().numpy()[0,..., :3] * 255).astype('uint8'))

    duration = 2000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(f"my_images/torus_implicit.gif", images, duration=duration,loop=0)
    
def render_oct_mesh(image_size=256, voxel_size=64, device=None, R=1, r=0.5):
    if device is None:
        device = get_device()
    
    min_value = -2.1
    max_value = 2.1
    
    X, Y, Z = torch.meshgrid(torch.linspace(-2, 2, 100), 
                         torch.linspace(-2, 2, 100), 
                         torch.linspace(-2, 2, 100))
    voxels = torch.abs(X) + torch.abs(Y) + torch.abs(Z) <= 1
    
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    
    # Vertex coordinates are indexed by array position, so we need to renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value

    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    
    images=[]
    for azim in range(0,360,10):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=6, elev=10, azim=azim)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    
        rend = renderer(mesh, cameras=cameras, lights=lights)
        images.append((rend.cpu().numpy()[0,..., :3] * 255).astype('uint8'))

    duration = 2000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(f"my_images/octahedron_implicit.gif", images, duration=duration,loop=0)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", type=str, choices=["point_cloud", "parametric", "implicit"])
    parser.add_argument("--which_plant", type=str, choices=["first", "second", "union"])
    parser.add_argument("--which_parameter", type=str, choices=["torus", "octahedron"])
    parser.add_argument("--which_implicit", type=str, choices=["torus", "octahedron"])
    args=parser.parse_args()
    
    if args.render=='point_cloud':
        render_plant(args.which_plant)
    if args.render=='parametric':
        if args.which_parameter=='torus':
            render_torus()
        elif args.which_parameter=='octahedron':
            render_oct()
        else:
            render_torus()
            render_oct()
    if args.render=='implicit':
        if args.which_implicit=='torus':
            render_torus_mesh()
        elif args.which_implicit=='octahedron':
            render_oct_mesh()
        else:
            render_torus_mesh()
            render_oct_mesh()