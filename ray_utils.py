import torch
import numpy as np
import matplotlib.pyplot as plt
import torch

def get_rays_us_linear(N_samples_lateral, sw, c2w):
    # Extract translation and rotation from camera-to-world matrix
    t = c2w[:3, -1]
    R = c2w[:3, :3]

    # Create a range of x-values
    x =  torch.linspace(-N_samples_lateral/2, N_samples_lateral/2, steps=N_samples_lateral) * sw
    y = torch.zeros_like(x)
    z = torch.zeros_like(x)

    # Stack and prepare the origin base
    origin_base = torch.stack([x, y, z], dim=1)

    # Rotate the origin base and add translation
    origin_rotated = torch.mm(origin_base, R.transpose(0, 1))
    rays_o = origin_rotated + t

    # Define direction base and rotate
    dirs_base = torch.tensor([0., 1., 0.], dtype=torch.float32)
    dirs_r = torch.mm(dirs_base.unsqueeze(0), R.transpose(0, 1)).squeeze(0)
    rays_d = dirs_r.repeat(N_samples_lateral, 1)  # Repeat for all rays
    # visualize_rays(rays_o, rays_d, c2w) 

    return rays_o, rays_d



def visualize_rays(rays_o, rays_d, c2w):
    # Create a 3D plot
    rays_d = rays_d.detach().cpu().numpy()
    rays_o = rays_o.detach().cpu().numpy()

    visualizer = np.arange(0, rays_o.shape[0], 1)

    pose = c2w.detach().cpu().numpy()   
    # Extract translation (position) from the last column
    x, y, z = pose[:3, 3]

    # Create basis vectors for the pose
    # The columns of the rotation matrix represent the basis vectors
    basis_vectors = pose[:3, :3]  

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

        
    # Draw the basis vectors
    ax.quiver(x, y, z, *basis_vectors[:, 0], color='r', length=0.01, normalize=True)
    ax.quiver(x, y, z, *basis_vectors[:, 1], color='g', length=0.01, normalize=True)
    ax.quiver(x, y, z, *basis_vectors[:, 2], color='b', length=0.01, normalize=True)

    # For each ray, plot a line from the origin in the direction of the ray
    for i in visualizer:
        # Define the end point of the ray for visualization
        end_point = rays_o[i] + rays_d[i] *0.001  # Scale the direction for visualization

        # Plot the ray
        ax.plot([rays_o[i, 0], end_point[0]], 
                [rays_o[i, 1], end_point[1]], 
                [rays_o[i, 2], end_point[2]], 'r-')

    # Set labels and show the plot
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.title('3D Ray Visualization')
    plt.show()

