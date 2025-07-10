import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

def visualize_bev_points(points, batch_idx=0, camera_idx=None, max_points=10000):
    """
    Visualize BEV points from the output of get_geometry function
    
    Args:
        points: tensor of shape B x N x D x H x W x 3
        batch_idx: which batch to visualize
        camera_idx: which camera to visualize (None means all cameras)
    """
    # Create colormap for different cameras
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan']
    
    plt.figure(figsize=(12, 10))
    
    # Top-down view (BEV)
    plt.subplot(2, 2, 1)
    if camera_idx is None:
        for cam in range(points.shape[1]):
            cam_points = points[batch_idx, cam].reshape(-1, 3)
            # Randomly sample points if too many
            if len(cam_points) > max_points:
                indices = np.random.choice(len(cam_points), max_points, replace=False)
                cam_points = cam_points[indices]
            plt.scatter(cam_points[:, 0].numpy(), cam_points[:, 1].numpy(), 
                       s=1, alpha=0.5, label=f'Camera {cam}', color=colors[cam % len(colors)])
    else:
        cam_points = points[batch_idx, camera_idx].reshape(-1, 3)
        if len(cam_points) > max_points:
            indices = np.random.choice(len(cam_points), max_points, replace=False)
            cam_points = cam_points[indices]
        plt.scatter(cam_points[:, 0].numpy(), cam_points[:, 1].numpy(), 
                   s=1, alpha=0.5, label=f'Camera {camera_idx}', color=colors[camera_idx % len(colors)])
    
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('X (forward)')
    plt.ylabel('Y (left)')
    plt.title('BEV (Top-Down) View')
    plt.legend()
    
    # 3D view
    ax = plt.subplot(2, 2, 2, projection='3d')
    if camera_idx is None:
        for cam in range(points.shape[1]):
            cam_points = points[batch_idx, cam].reshape(-1, 3)
            if len(cam_points) > max_points:
                indices = np.random.choice(len(cam_points), max_points, replace=False)
                cam_points = cam_points[indices]
            ax.scatter(cam_points[:, 0].numpy(), cam_points[:, 1].numpy(), cam_points[:, 2].numpy(), 
                      s=1, alpha=0.5, label=f'Camera {cam}', color=colors[cam % len(colors)])
    else:
        cam_points = points[batch_idx, camera_idx].reshape(-1, 3)
        if len(cam_points) > max_points:
            indices = np.random.choice(len(cam_points), max_points, replace=False)
            cam_points = cam_points[indices]
        ax.scatter(cam_points[:, 0].numpy(), cam_points[:, 1].numpy(), cam_points[:, 2].numpy(), 
                  s=1, alpha=0.5, label=f'Camera {camera_idx}', color=colors[camera_idx % len(colors)])
    
    ax.set_xlabel('X (forward)')
    ax.set_ylabel('Y (left)')
    ax.set_zlabel('Z (up)')
    ax.set_title('3D View')
    ax.legend()
    
    # Front view (X-Z plane)
    plt.subplot(2, 2, 3)
    if camera_idx is None:
        for cam in range(points.shape[1]):
            cam_points = points[batch_idx, cam].reshape(-1, 3)
            if len(cam_points) > max_points:
                indices = np.random.choice(len(cam_points), max_points, replace=False)
                cam_points = cam_points[indices]
            plt.scatter(cam_points[:, 0].numpy(), cam_points[:, 2].numpy(), 
                       s=1, alpha=0.5, label=f'Camera {cam}', color=colors[cam % len(colors)])
    else:
        cam_points = points[batch_idx, camera_idx].reshape(-1, 3)
        if len(cam_points) > max_points:
            indices = np.random.choice(len(cam_points), max_points, replace=False)
            cam_points = cam_points[indices]
        plt.scatter(cam_points[:, 0].numpy(), cam_points[:, 2].numpy(), 
                   s=1, alpha=0.5, label=f'Camera {camera_idx}', color=colors[camera_idx % len(colors)])
    
    plt.grid(True)
    plt.xlabel('X (forward)')
    plt.ylabel('Z (up)')
    plt.title('Front View (X-Z plane)')
    
    # Side view (Y-Z plane)
    plt.subplot(2, 2, 4)
    if camera_idx is None:
        for cam in range(points.shape[1]):
            cam_points = points[batch_idx, cam].reshape(-1, 3)
            if len(cam_points) > max_points:
                indices = np.random.choice(len(cam_points), max_points, replace=False)
                cam_points = cam_points[indices]
            plt.scatter(cam_points[:, 1].numpy(), cam_points[:, 2].numpy(), 
                       s=1, alpha=0.5, label=f'Camera {cam}', color=colors[cam % len(colors)])
    else:
        cam_points = points[batch_idx, camera_idx].reshape(-1, 3)
        if len(cam_points) > max_points:
            indices = np.random.choice(len(cam_points), max_points, replace=False)
            cam_points = cam_points[indices]
        plt.scatter(cam_points[:, 1].numpy(), cam_points[:, 2].numpy(), 
                   s=1, alpha=0.5, label=f'Camera {camera_idx}', color=colors[camera_idx % len(colors)])
    
    plt.grid(True)
    plt.xlabel('Y (left)')
    plt.ylabel('Z (up)')
    plt.title('Side View (Y-Z plane)')
    
    plt.tight_layout()
    plt.savefig('/home/xiangbog/Folder/Research/SkyLink/airv2x/debug/bev_visualization.png', dpi=300)

def visualize_cameras_frustum(points, batch_idx=0):
    """Visualize the frustum of each camera in 3D space"""
    # Create colormap for different cameras
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan']
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Ego vehicle position (origin)
    ax.scatter([0], [0], [0], color='black', s=100, marker='*', label='Ego Vehicle')
    
    # For each camera
    for cam in range(points.shape[1]):
        # Get camera center (the translation vector)
        cam_center = points[batch_idx, cam, 0, 0, 0].numpy()
        
        # Sample points from this camera's frustum for visualization
        D, H, W = points.shape[2:5]
        
        # Sample grid points
        d_indices = np.linspace(0, D-1, 5).astype(int)
        h_indices = np.linspace(0, H-1, 4).astype(int)
        w_indices = np.linspace(0, W-1, 4).astype(int)
        
        # Get frustum corner points
        frustum_points = []
        for d in d_indices:
            for h in h_indices:
                for w in w_indices:
                    frustum_points.append(points[batch_idx, cam, d, h, w].numpy())
        
        frustum_points = np.array(frustum_points)
        
        # Plot the frustum points
        ax.scatter(frustum_points[:, 0], frustum_points[:, 1], frustum_points[:, 2], 
                  color=colors[cam % len(colors)], s=2, alpha=0.3)
        
        # Connect the camera center to the nearest depth plane corners
        nearest_d = 0
        for h in h_indices:
            for w in w_indices:
                corner = points[batch_idx, cam, nearest_d, h, w].numpy()
                ax.plot([cam_center[0], corner[0]], 
                        [cam_center[1], corner[1]], 
                        [cam_center[2], corner[2]], 
                        color=colors[cam % len(colors)], linewidth=0.5, alpha=0.5)
    
    # Set axis labels and title
    ax.set_xlabel('X (forward)')
    ax.set_ylabel('Y (left)')
    ax.set_zlabel('Z (up)')
    ax.set_title('Camera Frustums in 3D Space')
    
    # Set equal aspect ratio
    max_range = np.array([
        ax.get_xlim()[1] - ax.get_xlim()[0],
        ax.get_ylim()[1] - ax.get_ylim()[0],
        ax.get_zlim()[1] - ax.get_zlim()[0]
    ]).max() / 2.0
    
    mid_x = (ax.get_xlim()[1] + ax.get_xlim()[0]) / 2
    mid_y = (ax.get_ylim()[1] + ax.get_ylim()[0]) / 2
    mid_z = (ax.get_zlim()[1] + ax.get_zlim()[0]) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.savefig('/home/xiangbog/Folder/Research/SkyLink/airv2x/debug/camera_frustums.png', dpi=300)

def visualize_density(points, batch_idx=0, camera_idx=None, grid_res=0.5):
    """
    Visualize point density in BEV space
    
    Args:
        points: tensor of shape B x N x D x H x W x 3
        batch_idx: which batch to visualize
        camera_idx: which camera to visualize (None means all cameras)
        grid_res: resolution of grid for density calculation (meters)
    """
    plt.figure(figsize=(14, 6))
    
    # Extract points
    if camera_idx is None:
        # Combine all cameras
        all_points = []
        for cam in range(points.shape[1]):
            all_points.append(points[batch_idx, cam].reshape(-1, 3))
        points_combined = torch.cat(all_points, dim=0).numpy()
    else:
        points_combined = points[batch_idx, camera_idx].reshape(-1, 3).numpy()
    
    # Create grid for density calculation
    x_min, x_max = np.floor(points_combined[:, 0].min()), np.ceil(points_combined[:, 0].max())
    y_min, y_max = np.floor(points_combined[:, 1].min()), np.ceil(points_combined[:, 1].max())
    
    x_grid = np.arange(x_min, x_max + grid_res, grid_res)
    y_grid = np.arange(y_min, y_max + grid_res, grid_res)
    
    # Calculate point density using histogram2d
    hist, x_edges, y_edges = np.histogram2d(
        points_combined[:, 0], points_combined[:, 1],
        bins=[x_grid, y_grid]
    )
    
    # Normalize and apply log for better visualization
    hist = np.log1p(hist.T)  # log(1+x) to handle zeros
    
    # Create colormap
    cmap = plt.cm.viridis
    
    # Plot density map
    plt.subplot(1, 2, 1)
    plt.imshow(hist, cmap=cmap, extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='equal')
    plt.colorbar(label='Log(1+count)')
    plt.xlabel('X (forward)')
    plt.ylabel('Y (left)')
    plt.title('BEV Point Density')
    plt.grid(True, alpha=0.3)
    
    # Plot scatter for comparison
    plt.subplot(1, 2, 2)
    sample_size = min(10000, len(points_combined))
    sample_idx = np.random.choice(len(points_combined), sample_size, replace=False)
    plt.scatter(points_combined[sample_idx, 0], points_combined[sample_idx, 1], 
               s=1, alpha=0.3, c='blue')
    plt.xlabel('X (forward)')
    plt.ylabel('Y (left)')
    plt.title('BEV Points Sample')
    plt.axis('equal')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/xiangbog/Folder/Research/SkyLink/airv2x/debug/density_visualization.png', dpi=300)