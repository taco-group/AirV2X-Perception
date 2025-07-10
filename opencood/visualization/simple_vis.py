# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# Modifier: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
from matplotlib import pyplot as plt
import torch
from matplotlib.colors import ListedColormap

import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev

def visualize(
    pred_box_tensor,
    gt_tensor,
    pcd,
    pc_range,
    save_path,
    method="3d",
    vis_gt_box=True,
    vis_pred_box=True,
    left_hand=False,
    uncertainty=None,
    **kwargs,
):
    """
    Visualize the prediction, ground truth with point cloud together.
    They may be flipped in y axis. Since carla is left hand coordinate, while kitti is right hand.

    Parameters
    ----------
    pred_box_tensor : torch.Tensor
        (N, 8, 3) prediction.

    gt_tensor : torch.Tensor
        (N, 8, 3) groundtruth bbx

    pcd : torch.Tensor
        PointCloud, (N, 4).

    pc_range : list
        [xmin, ymin, zmin, xmax, ymax, zmax]

    save_path : str
        Save the visualization results to given path.

    dataset : BaseDataset
        opencood dataset object.

    method: str, 'bev' or '3d'

    """
    pcd_rsu = kwargs.get("pcd_rsu", None)
    if pcd_rsu is not None:
        pcd_rsu = pcd_rsu.cpu().numpy()
    pcd_drone = kwargs.get("pcd_drone", None)
    if pcd_drone is not None:
        pcd_drone = pcd_drone.cpu().numpy()
        
    plt.figure(
        figsize=[(pc_range[3] - pc_range[0]) / 40, (pc_range[4] - pc_range[1]) / 40]
    )
    pc_range = [int(i) for i in pc_range]
    pcd_np = pcd.cpu().numpy()

    if vis_pred_box:
        pred_box_np = pred_box_tensor.cpu().numpy()
        pred_name = ["pred"] * pred_box_np.shape[0]
        # pred_name = [''] * pred_box_np.shape[0]
        if uncertainty is not None:
            uncertainty_np = uncertainty.cpu().numpy()
            uncertainty_np = np.exp(uncertainty_np)
            d_a_square = 1.6**2 + 3.9**2

            if uncertainty_np.shape[1] == 3:
                uncertainty_np[:, :2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np)
                # yaw angle is in radian, it's the same in g2o SE2's setting.

                pred_name = [
                    f"x_u:{uncertainty_np[i, 0]:.3f} y_u:{uncertainty_np[i, 1]:.3f} a_u:{uncertainty_np[i, 2]:.3f}"
                    for i in range(uncertainty_np.shape[0])
                ]

            elif uncertainty_np.shape[1] == 2:
                uncertainty_np[:, :2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np)  # yaw angle is in radian

                pred_name = [
                    f"x_u:{uncertainty_np[i, 0]:.3f} y_u:{uncertainty_np[i, 1]:3f}"
                    for i in range(uncertainty_np.shape[0])
                ]

            elif uncertainty_np.shape[1] == 7:
                uncertainty_np[:, :2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np)  # yaw angle is in radian

                pred_name = [
                    f"x_u:{uncertainty_np[i, 0]:.3f} y_u:{uncertainty_np[i, 1]:3f} a_u:{uncertainty_np[i, 6]:3f}"
                    for i in range(uncertainty_np.shape[0])
                ]

    if vis_gt_box:
        gt_box_np = gt_tensor.cpu().numpy()
        gt_name = ["gt"] * gt_box_np.shape[0]
        # gt_name = [''] * gt_box_np.shape[0]

    if method == "bev":
        canvas = canvas_bev.Canvas_BEV_heading_right(
            canvas_shape=(
                (pc_range[4] - pc_range[1]) * 10,
                (pc_range[3] - pc_range[0]) * 10,
            ),
            canvas_x_range=(pc_range[0], pc_range[3]),
            canvas_y_range=(pc_range[1], pc_range[4]),
            left_hand=left_hand,
        )

        canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)  # Get Canvas Coords
        canvas.draw_canvas_points(canvas_xy[valid_mask], colors=(111, 179, 126))
        if pcd_rsu is not None:
            canvas_xy_rsu, valid_mask_rsu = canvas.get_canvas_coords(pcd_rsu)
            canvas.draw_canvas_points(canvas_xy_rsu[valid_mask_rsu], colors=(11,49,161))
        if pcd_drone is not None:
            canvas_xy_drone, valid_mask_drone = canvas.get_canvas_coords(pcd_drone)
            canvas.draw_canvas_points(
                canvas_xy_drone[valid_mask_drone], colors=(150,60,0)
            )
        
        if vis_gt_box:
            canvas.draw_boxes(gt_box_np, colors=(0, 255, 0))
            # canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
            # canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name, box_line_thickness=6)
        if vis_pred_box:
            canvas.draw_boxes(pred_box_np, colors=(255, 0, 0))
            # canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name)
            # canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name, box_line_thickness=6)
            if "cavnum" in kwargs:
                canvas.draw_boxes(
                    pred_box_np[: kwargs["cavnum"]],
                    colors=(0, 191, 255),
                    texts=[""] * kwargs["cavnum"],
                )
                # canvas.draw_boxes(pred_box_np[:kwargs['cavnum']], colors=(0,191,255), texts=['']*kwargs['cavnum'], box_line_thickness=6)

    elif method == "3d":
        canvas = canvas_3d.Canvas_3D(left_hand=left_hand)
        canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
        canvas.draw_canvas_points(canvas_xy[valid_mask], colors=(111, 179, 126))
        if pcd_rsu is not None:
            canvas_xy_rsu, valid_mask_rsu = canvas.get_canvas_coords(pcd_rsu)
            canvas.draw_canvas_points(canvas_xy_rsu[valid_mask_rsu], colors=(11,49,161))
        if pcd_drone is not None:
            canvas_xy_drone, valid_mask_drone = canvas.get_canvas_coords(pcd_drone)
            canvas.draw_canvas_points(
                canvas_xy_drone[valid_mask_drone], colors=(150,60,0)
            )
        if vis_pred_box:
            canvas.draw_boxes(pred_box_np, colors=(255, 0, 0))
            # canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name)
        if vis_gt_box:
            canvas.draw_boxes(gt_box_np, colors=(0, 255, 0))
            # canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
    else:
        raise (f"Not Completed for f{method} visualization.")

    plt.axis("off")

    plt.imshow(canvas.canvas)

    plt.tight_layout()
    plt.savefig(save_path, transparent=False, dpi=400)
    plt.clf()
    plt.close()


def visualize_segmentation(seg_tensor, num_classes, basepath, prefix, idx):
    """
    Visualize dynamic segmentation tensor with shape [1, H, W] containing up to 11 classes
    
    Parameters:
    - seg_tensor: Segmentation tensor with shape [1, H, W]
    
    Returns:
    - segmentation: numpy array of the segmentation map
    """
    # Remove batch dimension
    segmentation = seg_tensor.squeeze(0)  # Now has shape [H, W]
    
    # Convert to numpy if it's a torch tensor
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.detach().cpu().numpy()
    
    # Define color map for dynamic segmentation with 11     
    # Create a distinct colormap for 11 classes
    # Using tab20 which has 20 distinct colors
    colors = plt.cm.tab20(np.linspace(0, 1, 20))[:num_classes]
    cmap = ListedColormap(colors)
    
    # Plot the segmentation
    plt.figure(figsize=(10, 8))
    img = plt.imshow(segmentation, cmap=cmap, vmin=0, vmax=num_classes-1)
    plt.title(f"{prefix} Segmentation ({num_classes} classes)")
    plt.axis('off')
    
    # Add a color bar to show class mapping
    cbar = plt.colorbar(img, ticks=np.arange(num_classes))
    cbar.set_label('Class')
    
    # Add class labels if needed
    # cbar.set_ticklabels(['Background', 'Class 1', 'Class 2', ...]) # Add your class names here
    
    plt.tight_layout()
    plt.savefig(f"{basepath}/{prefix}_{idx}.png", dpi=300)
    