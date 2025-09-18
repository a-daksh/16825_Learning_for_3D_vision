import numpy as np
import torch
import pytorch3d
import imageio
from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def create_floor_mesh(size=8, device=None):
    if device is None:
        device = get_device()
    
    vertices = torch.tensor([
        [-size, -1.5, -size],
        [size, -1.5, -size],
        [size, -1.5, size],
        [-size, -1.5, size]
    ], dtype=torch.float32).unsqueeze(0).to(device)
    
    faces = torch.tensor([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=torch.long).unsqueeze(0).to(device)
    
    textures = torch.ones_like(vertices) * torch.tensor([0.9, 0.9, 0.85], dtype=torch.float32).to(device)
    
    return vertices, faces, textures


def render_disco_cow():
    device = get_device()
    renderer = get_mesh_renderer(image_size=512)
    
    cow_vertices, cow_faces = load_cow_mesh("data/cow.obj")
    cow_vertices = cow_vertices.unsqueeze(0).to(device)
    cow_faces = cow_faces.unsqueeze(0).to(device)
    
    floor_vertices, floor_faces, floor_textures = create_floor_mesh(device=device)
    
    all_vertices = torch.cat([cow_vertices, floor_vertices], dim=1)
    all_faces = torch.cat([cow_faces, floor_faces + cow_vertices.shape[1]], dim=1)
    
    images = []
    
    for frame in range(72):
        angle = frame * 5
        time = frame * 0.1
        
        light_r = abs(np.sin(time * 2)) 
        light_g = abs(np.sin(time * 2 + 2))
        light_b = abs(np.sin(time * 2 + 4))
        
        cow_textures = torch.ones_like(cow_vertices) * torch.tensor([0.7, 0.7, 1], dtype=torch.float32).to(device)
        all_textures = torch.cat([cow_textures, floor_textures], dim=1)
        
        mesh = pytorch3d.structures.Meshes(
            verts=all_vertices,
            faces=all_faces,
            textures=pytorch3d.renderer.TexturesVertex(all_textures)
        )
        
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=8, elev=15, azim=angle
        )
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        
        light_x = 3 * np.cos(time * 3)
        light_z = 3 * np.sin(time * 3)
        lights = pytorch3d.renderer.PointLights(
            location=[[light_x, 2, light_z]], 
            ambient_color=[[0.2, 0.2, 0.2]],
            diffuse_color=[[light_r, light_g, light_b]],
            device=device
        )
        
        rend = renderer(mesh, cameras=cameras, lights=lights)
        image = (rend.cpu().numpy()[0, ..., :3] * 255).astype('uint8')
        images.append(image)
    
    return images


if __name__ == "__main__":
    images = render_disco_cow()
    duration = 100
    imageio.mimsave("my_images/disco_cow.gif", images, duration=duration, loop=0)