import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
from PIL import Image
import torch.nn.functional as F

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

def visualize_bev_feature(
        bev_feat: torch.Tensor,
        batch_idx: int = 0,
        channel_idx: int = None,
        save_path: str = None
    ) -> np.ndarray:
    """
    Visualize a BEV feature map as a heatmap.

    Args:
        bev_feat (torch.Tensor): Input tensor of shape [B, C, H, W].
        batch_idx (int): Index of the sample in the batch to visualize. Default is 0.
        channel_idx (int): If specified, visualize only this channel; otherwise, average across all channels.
        save_path (str): If given, save the heatmap image to this path; otherwise, display it in a window.

    Returns:
        np.ndarray: The resulting BGR heatmap image (dtype uint8).
    """
    # Extract the chosen sample and move to CPU numpy array
    if isinstance(bev_feat, torch.Tensor):
        array = bev_feat[batch_idx].detach().cpu().numpy()  # shape: (C, H, W)
    else:
        array = bev_feat  # assume already a numpy array

    # Select a single channel or average across channels
    if channel_idx is None:
        feature = np.mean(array, axis=0)  # shape: (H, W)
    else:
        feature = array[channel_idx]      # shape: (H, W)

    # Normalize values to [0, 255]
    gray_img = (feature * 255).astype(np.uint8)

    # Save or display the result
    if save_path:
        cv2.imwrite(save_path, gray_img)
    else:
        cv2.imshow("BEV Feature Heatmap", gray_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def visualize_bev_feature_in_once(save_path):
    """
    Visualize multiple BEV features by overlaying them with different colors.

    Args:
        save_path (str): Base path for saving the visualization results.
    """
    imgs = [Image.open(save_path + f"bev_feature_{i}.png").convert('L') for i in range(6)]
    w, h = imgs[0].size  

    # Define 6 RGB colors for different features
    colors = [
        (255,   0,   0),  # Red
        (  0, 255,   0),  # Green
        (  0,   0, 255),  # Blue
        (255, 255,   0),  # Yellow
        (255,   0, 255),  # Magenta
        (  0, 255, 255),  # Cyan
    ]

    # Accumulate colored features
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    for gray, col in zip(imgs, colors):
        arr = np.array(gray, dtype=np.float32) / 255.0  # Normalize
        for c in range(3):
            canvas[..., c] += arr * col[c]

    # Clip values and convert to uint8
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    out = Image.fromarray(canvas)
    out.save(save_path + "bev_overlay.png")

def depth_to_one_hot(x, num_bins=41, min_val=0.0, max_val=256.0):
    """
    Convert depth values to one-hot encoding.

    Args:
        x: Tensor of shape [B, N, 1, fH, fW, C], depth values in [min_val, max_val]
        num_bins: number of one-hot bins (default: 41)
        min_val: minimum depth value
        max_val: maximum depth value

    Returns:
        Tensor of shape [B, N, num_bins, fH, fW, C], one-hot along the depth dimension
    """
    # Remove the singleton depth dim
    depth = x.squeeze(2)           # -> [B, N, fH, fW, C]
    
    # Compute number of bins based on max depth value
    num_bins = int(depth.max()) + 1

    # One-hot encode
    one_hot = F.one_hot(depth, num_classes=num_bins)  # -> [B,N,fH,fW,C,num_bins]
    
    # Move the bin axis into position 2
    one_hot = one_hot.permute(0, 1, 5, 2, 3, 4).float()  # -> [B,N,num_bins,fH,fW,C]
    return one_hot

def visualize_3d_points(geom, x_img, save_path='3d_points.png', threshold=0):
    """
    Visualize depth-encoded image features mapped to 3D space.
    
    Args:
        geom: B x N x D x H x W x 3 tensor, mapping from pixel coordinates to ego vehicle coordinate system
        x_img: B x N x D x fH x fW x 1 tensor, one-hot encoded depth
        save_path: path to save visualization results
        threshold: threshold for considering a depth bin as valid
    """
    # Convert inputs to numpy arrays
    if isinstance(geom, torch.Tensor):
        geom = geom.detach().cpu().numpy()
    if isinstance(x_img, torch.Tensor):
        x_img = x_img.detach().cpu().numpy()
    
    # Process only the first batch
    batch_idx = 0
    
    # Create figure with 3D axes (GUI-less mode)
    fig = plt.figure(figsize=(10, 8))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get dimension information
    B, N, D, H, W, _ = geom.shape
    B, N, D, fH, fW, _ = x_img.shape
    
    # Calculate scaling factors (if feature map size differs from original image)
    scale_h = H / fH
    scale_w = W / fW
    
    # Assign different colors for each camera
    colors = plt.cm.rainbow(np.linspace(0, 1, N))
    
    for cam_idx in range(N):
        # Extract geometry mapping and depth encoding for current camera
        cam_geom = geom[batch_idx, cam_idx]  # D x H x W x 3
        cam_depth = x_img[batch_idx, cam_idx]  # D x fH x fW x 1
        
        # Remove last dimension from cam_depth
        cam_depth = cam_depth.squeeze(-1)  # D x fH x fW
        
        # Find depth bin with maximum value for each pixel
        depth_values = np.max(cam_depth, axis=0)  # fH x fW
        depth_indices = np.argmax(cam_depth, axis=0)  # fH x fW
        
        # Collect 3D points
        points_3d = []
        
        # Process each pixel
        for h in range(fH):
            for w in range(fW):
                if depth_values[h, w] > threshold:
                    d = depth_indices[h, w]
                    
                    # Map to original image coordinates
                    orig_h = min(int(h * scale_h), H - 1)
                    orig_w = min(int(w * scale_w), W - 1)
                    
                    # Get 3D coordinates
                    point_3d = cam_geom[d, orig_h, orig_w]
                    points_3d.append(point_3d)
        
        print(f"Camera {cam_idx+1}: Found {len(points_3d)} valid 3D points")
        
        if points_3d:
            points_3d = np.array(points_3d)
            
            # Plot points in 3D
            ax.scatter(
                points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                c=[colors[cam_idx]], s=1, alpha=0.5, label=f'Camera {cam_idx+1}'
            )
    
    # Set plot properties
    ax.set_title('3D point cloud visualization')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set equal aspect ratio
    x_lim = ax.get_xlim3d()
    y_lim = ax.get_ylim3d()
    z_lim = ax.get_zlim3d()
    x_range = x_lim[1] - x_lim[0]
    y_range = y_lim[1] - y_lim[0]
    max_range = max(x_range, y_range)

    z_mid = 0.5 * (z_lim[0] + z_lim[1])
    ax.set_zlim3d(z_mid - max_range/2, z_mid + max_range/2)
    ax.set_box_aspect([1, 1, 1]) 
    ax.set_proj_type('ortho')
    ax.legend()
    
    # Save figure
    fig.savefig(save_path)
    plt.close(fig)
    
    print(f"3D visualization saved to {save_path}")