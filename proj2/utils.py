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
        AssertionError("Specify Device !!!!!!!")
    
    voxels=voxels.cpu().detach().squeeze(0)
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
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
 