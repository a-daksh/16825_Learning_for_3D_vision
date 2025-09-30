import torch
import mcubes
import pytorch3d
import imageio

from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)

def render_voxels(name,voxels, image_size=256, device=None):
    if device is None:
        raise AssertionError("Specify Device !!!!!!!")
    
    voxels=voxels.cpu().detach().squeeze(0)
    vertices, faces = mcubes.marching_cubes(voxels.numpy(), isovalue=0.5)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))

    center = (vertices.min(0).values + vertices.max(0).values) / 2
    vertices = (vertices - center) * 2
    
    textures = pytorch3d.renderer.TexturesVertex(torch.ones_like(vertices).unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    
    images=[]
    for azim in range(0,360,10):
        R, T = pytorch3d.renderer.look_at_view_transform(dist=3*vertices.max(), elev=0, azim=azim)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    
        rend = renderer(mesh, cameras=cameras, lights=lights)
        images.append((rend.cpu().numpy()[0,..., :3] * 255).astype('uint8'))

    duration = 2000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(f"my_images/{name}.gif", images, duration=duration,loop=0)
 
def render_point_cloud(name, verts, image_size=256, rgb=(1, 1, 1),radius=0.01, device=None):
    """
    Renders a point cloud.
    """
    if device is None:
        raise AssertionError("Specify Device !!!!!!!")

    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=rgb),
    )
    
    points=verts.to(device)
    features=torch.tensor(rgb).expand(1, points.shape[1],-1).to(device)
    point_cloud = pytorch3d.structures.Pointclouds(points=points, features=features.float())

    images=[]
    for azimuth in range(0,360,10):
        R, T = pytorch3d.renderer.look_at_view_transform(3*verts.max(), 0, azimuth)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

        rend = renderer(point_cloud, cameras=cameras)
        images.append((rend.detach().cpu().numpy()[0,..., :3] * 255).astype('uint8'))

    duration = 2000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(f"my_images/{name}.gif", images, duration=duration,loop=0)
 
def render_mesh(name, mesh, image_size=256, color=[0.7, 0.7, 1], device=None):
    if device is None:
        raise AssertionError("Specify Device !!!!!!!")

    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )

    # set per-vertex color on the mesh
    verts = mesh.verts_padded().to(device)                            # (1, V, 3)
    verts_rgb = torch.ones_like(verts) * torch.tensor(color, device=device)
    mesh.textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb)

    images = []
    for azimuth in range(0, 360, 10):
        R, T = pytorch3d.renderer.look_at_view_transform(3*verts.max(), 0, azimuth, device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        images.append((rend.detach().cpu().numpy()[0, ..., :3] * 255).astype('uint8'))

    duration = 2000 // 15
    imageio.mimsave(f"my_images/{name}.gif", images, duration=duration, loop=0)
