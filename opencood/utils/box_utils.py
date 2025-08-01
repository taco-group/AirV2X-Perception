# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Bounding box related utility functions
"""

import copy
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
from pyquaternion import Quaternion

import opencood.utils.common_utils as common_utils
from opencood.utils.transformation_utils import x1_to_x2, x_to_world


def corner_to_center_torch(corner3d, order="lwh"):
    corner3d_ = corner3d.cpu().numpy()
    return torch.from_numpy(corner_to_center(corner3d_, order)).to(corner3d.device)


def corner_to_center(corner3d, order="lwh"):
    """
    Convert 8 corners to x, y, z, dx, dy, dz, yaw.
    yaw in radians

    Parameters
    ----------
    corner3d : np.ndarray
        (N, 8, 3)

    order : str, for output.
        'lwh' or 'hwl'

    Returns
    -------
    box3d : np.ndarray
        (N, 7)
    """
    assert corner3d.ndim == 3
    batch_size = corner3d.shape[0]

    xyz = np.mean(corner3d[:, [0, 3, 5, 6], :], axis=1)
    h = abs(np.mean(corner3d[:, 4:, 2] - corner3d[:, :4, 2], axis=1, keepdims=True))
    l = (
        np.sqrt(
            np.sum(
                (corner3d[:, 0, [0, 1]] - corner3d[:, 3, [0, 1]]) ** 2,
                axis=1,
                keepdims=True,
            )
        )
        + np.sqrt(
            np.sum(
                (corner3d[:, 2, [0, 1]] - corner3d[:, 1, [0, 1]]) ** 2,
                axis=1,
                keepdims=True,
            )
        )
        + np.sqrt(
            np.sum(
                (corner3d[:, 4, [0, 1]] - corner3d[:, 7, [0, 1]]) ** 2,
                axis=1,
                keepdims=True,
            )
        )
        + np.sqrt(
            np.sum(
                (corner3d[:, 5, [0, 1]] - corner3d[:, 6, [0, 1]]) ** 2,
                axis=1,
                keepdims=True,
            )
        )
    ) / 4

    w = (
        np.sqrt(
            np.sum(
                (corner3d[:, 0, [0, 1]] - corner3d[:, 1, [0, 1]]) ** 2,
                axis=1,
                keepdims=True,
            )
        )
        + np.sqrt(
            np.sum(
                (corner3d[:, 2, [0, 1]] - corner3d[:, 3, [0, 1]]) ** 2,
                axis=1,
                keepdims=True,
            )
        )
        + np.sqrt(
            np.sum(
                (corner3d[:, 4, [0, 1]] - corner3d[:, 5, [0, 1]]) ** 2,
                axis=1,
                keepdims=True,
            )
        )
        + np.sqrt(
            np.sum(
                (corner3d[:, 6, [0, 1]] - corner3d[:, 7, [0, 1]]) ** 2,
                axis=1,
                keepdims=True,
            )
        )
    ) / 4

    theta = (
        np.arctan2(
            corner3d[:, 1, 1] - corner3d[:, 2, 1], corner3d[:, 1, 0] - corner3d[:, 2, 0]
        )
        + np.arctan2(
            corner3d[:, 0, 1] - corner3d[:, 3, 1], corner3d[:, 0, 0] - corner3d[:, 3, 0]
        )
        + np.arctan2(
            corner3d[:, 5, 1] - corner3d[:, 6, 1], corner3d[:, 5, 0] - corner3d[:, 6, 0]
        )
        + np.arctan2(
            corner3d[:, 4, 1] - corner3d[:, 7, 1], corner3d[:, 4, 0] - corner3d[:, 7, 0]
        )
    )[:, np.newaxis] / 4

    if order == "lwh":
        return np.concatenate([xyz, l, w, h, theta], axis=1).reshape(batch_size, 7)
    elif order == "hwl":
        return np.concatenate([xyz, h, w, l, theta], axis=1).reshape(batch_size, 7)
    else:
        sys.exit("Unknown order")


def boxes_to_corners2d(boxes3d, order):
    """
      0 -------- 1
      |          |
      |          |
      |          |
      3 -------- 2
    Parameters
    __________
    boxes3d: np.ndarray or torch.Tensor
        (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center.

    order : str
        'lwh' or 'hwl'

    Returns:
        corners2d: np.ndarray or torch.Tensor
        (N, 4, 3), the 4 corners of the bounding box.

    """
    corners3d = boxes_to_corners_3d(boxes3d, order)
    corners2d = corners3d[:, :4, :]
    return corners2d


def boxes2d_to_corners2d(boxes2d, order="lwh"):
    """
      0 -------- 1
      |          |
      |          |
      |          |
      3 -------- 2
    Parameters
    __________
    boxes2d: np.ndarray or torch.Tensor
        (..., 5) [x, y, dx, dy, heading], (x, y) is the box center.

    order : str
        'lwh' or 'hwl'

    Returns:
        corners2d: np.ndarray or torch.Tensor
        (..., 4, 2), the 4 corners of the bounding box.

    """
    assert order == "lwh", "boxes2d_to_corners_2d only supports lwh order for now."
    boxes2d, is_numpy = common_utils.check_numpy_to_torch(boxes2d)
    template = boxes2d.new_tensor(([1, -1], [1, 1], [-1, 1], [-1, -1])) / 2
    input_shape = boxes2d.shape
    boxes2d = boxes2d.view(-1, 5)
    corners2d = boxes2d[:, None, 2:4].repeat(1, 4, 1) * template[None, :, :]
    corners2d = common_utils.rotate_points_along_z_2d(
        corners2d.view(-1, 2), boxes2d[:, 4].repeat_interleave(4)
    ).view(-1, 4, 2)
    corners2d += boxes2d[:, None, 0:2]
    corners2d = corners2d.view(*(input_shape[:-1]), 4, 2)
    return corners2d


def boxes_to_corners_3d(boxes3d, order):
    """
        4 -------- 5
       /|         /|
      7 -------- 6 .
      | |        | |
      . 0 -------- 1
      |/         |/
      3 -------- 2
    Parameters
    __________
    boxes3d: np.ndarray or torch.Tensor
        (N, 7) [x, y, z, l, w, h, heading], or [x, y, z, h, w, l, heading]

               (x, y, z) is the box center.

    order : str
        'lwh' or 'hwl'

    Returns:
        corners3d: np.ndarray or torch.Tensor
        (N, 8, 3), the 8 corners of the bounding box.


    opv2v's left hand coord

    ^ z
    |
    |
    | . x
    |/
    +-------> y

    """

    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
    boxes3d_ = boxes3d

    if order == "hwl":
        boxes3d_ = boxes3d[:, [0, 1, 2, 5, 4, 3, 6]]

    template = (
        boxes3d_.new_tensor(
            (
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, -1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
                [-1, -1, 1],
            )
        )
        / 2
    )

    corners3d = boxes3d_[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = common_utils.rotate_points_along_z(
        corners3d.view(-1, 8, 3), boxes3d_[:, 6]
    ).view(-1, 8, 3)
    corners3d += boxes3d_[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def box3d_to_2d(box3d):
    """
    Convert 3D bounding box to 2D.

    Parameters
    ----------
    box3d : np.ndarray
        (n, 8, 3)

    Returns
    -------
    box2d : np.ndarray
        (n, 4, 2), project 3d to 2d.
    """
    box2d = box3d[:, :4, :2]
    return box2d


def corner2d_to_standup_box(box2d):
    """
    Find the minmaxx, minmaxy for each 2d box. (N, 4, 2) -> (N, 4)
    x1, y1, x2, y2

    Parameters
    ----------
    box2d : np.ndarray
        (n, 4, 2), four corners of the 2d bounding box.

    Returns
    -------
    standup_box2d : np.ndarray
        (n, 4)
    """
    N = box2d.shape[0]
    standup_boxes2d = np.zeros((N, 4))

    standup_boxes2d[:, 0] = np.min(box2d[:, :, 0], axis=1)
    standup_boxes2d[:, 1] = np.min(box2d[:, :, 1], axis=1)
    standup_boxes2d[:, 2] = np.max(box2d[:, :, 0], axis=1)
    standup_boxes2d[:, 3] = np.max(box2d[:, :, 1], axis=1)

    return standup_boxes2d


def corner_to_standup_box_torch(box_corner):
    """
    Find the minmax x and y for each bounding box.

    Parameters
    ----------
    box_corner : torch.Tensor
        Shape: (N, 8, 3) or (N, 4)

    Returns
    -------
    standup_box2d : torch.Tensor
        (n, 4)
    """
    N = box_corner.shape[0]
    standup_boxes2d = torch.zeros((N, 4))

    standup_boxes2d = standup_boxes2d.to(box_corner.device)

    standup_boxes2d[:, 0] = torch.min(box_corner[:, :, 0], dim=1).values
    standup_boxes2d[:, 1] = torch.min(box_corner[:, :, 1], dim=1).values
    standup_boxes2d[:, 2] = torch.max(box_corner[:, :, 0], dim=1).values
    standup_boxes2d[:, 3] = torch.max(box_corner[:, :, 1], dim=1).values

    return standup_boxes2d


def project_box3d(box3d, transformation_matrix):
    """
    Project the 3d bounding box to another coordinate system based on the
    transfomration matrix.

    Parameters
    ----------
    box3d : torch.Tensor or np.ndarray
        3D bounding box, (N, 8, 3)

    transformation_matrix : torch.Tensor or np.ndarray
        Transformation matrix, (4, 4)

    Returns
    -------
    projected_box3d : torch.Tensor
        The projected bounding box, (N, 8, 3)
    """
    assert transformation_matrix.shape == (4, 4)
    box3d, is_numpy = common_utils.check_numpy_to_torch(box3d)
    transformation_matrix, _ = common_utils.check_numpy_to_torch(transformation_matrix)

    # (N, 3, 8)
    box3d_corner = box3d.transpose(1, 2)
    # (N, 1, 8)
    torch_ones = torch.ones((box3d_corner.shape[0], 1, 8))
    torch_ones = torch_ones.to(box3d_corner.device)
    # (N, 4, 8)
    box3d_corner = torch.cat((box3d_corner, torch_ones), dim=1)
    # (N, 4, 8)
    projected_box3d = torch.matmul(transformation_matrix, box3d_corner)
    # (N, 8, 3)
    projected_box3d = projected_box3d[:, :3, :].transpose(1, 2)

    return projected_box3d if not is_numpy else projected_box3d.numpy()


def project_points_by_matrix_torch(points, transformation_matrix):
    """
    Project the points to another coordinate system based on the
    transfomration matrix.

    IT NOT USED. LATTER ONE WITH THE SAME NAME WILL BE USED.

    Parameters
    ----------
    points : torch.Tensor
        3D points, (N, 3)

    transformation_matrix : torch.Tensor
        Transformation matrix, (4, 4)

    Returns
    -------
    projected_points : torch.Tensor
        The projected points, (N, 3)
    """
    # convert to homogeneous  coordinates via padding 1 at the last dimension.
    # (N, 4)
    points_homogeneous = F.pad(points, (0, 1), mode="constant", value=1)
    # (N, 4)
    projected_points = torch.einsum(
        "ik, jk->ij", points_homogeneous, transformation_matrix
    )
    return projected_points[:, :3]


def get_mask_for_boxes_within_range_torch(boxes, gt_range):
    """
    Generate mask to remove the bounding boxes
    outside the range.

    Parameters
    ----------
    boxes : torch.Tensor
        Groundtruth bbx, shape: N,8,3 or N,4,2

    gt_range: list
        [xmin, ymin, zmin, xmax, ymax, zmax]
    Returns
    -------
    mask: torch.Tensor
        The mask for bounding box -- True means the
        bbx is within the range and False means the
        bbx is outside the range.
    """

    # mask out the gt bounding box out fixed range (-140, -40, -3, 140, 40 1)
    device = boxes.device
    boundary_lower_range = torch.Tensor(gt_range[:2]).reshape(1, 1, -1).to(device)
    boundary_higher_range = torch.Tensor(gt_range[3:5]).reshape(1, 1, -1).to(device)

    mask = torch.all(
        torch.all(boxes[:, :, :2] >= boundary_lower_range, dim=-1)
        & torch.all(boxes[:, :, :2] <= boundary_higher_range, dim=-1),
        dim=-1,
    )

    return mask


def mask_boxes_outside_range_numpy(
    boxes, limit_range, order, min_num_corners=8, return_mask=False
):
    """
    Parameters
    ----------
    boxes: np.ndarray
        (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    limit_range: list
        [minx, miny, minz, maxx, maxy, maxz]

    min_num_corners: int
        The required minimum number of corners to be considered as in range.

    order : str
        'lwh' or 'hwl'

    return_mask : bool
        Whether return the mask.

    Returns
    -------
    boxes: np.ndarray
        The filtered boxes.
    """
    assert boxes.shape[1] == 8 or boxes.shape[1] == 7

    new_boxes = boxes.copy()
    if boxes.shape[1] == 7:
        new_boxes = boxes_to_corners_3d(new_boxes, order)

    mask = ((new_boxes >= limit_range[0:3]) & (new_boxes <= limit_range[3:6])).all(
        axis=2
    )
    mask = mask.sum(axis=1) >= min_num_corners  # (N)

    if return_mask:
        return boxes[mask], mask
    return boxes[mask]


def create_bbx(extent):
    """
    Create bounding box with 8 corners under obstacle vehicle reference.

    Parameters
    ----------
    extent : list
        half length, width and height

    Returns
    -------
    bbx : np.array
        The bounding box with 8 corners, shape: (8, 3)
    """

    bbx = np.array(
        [
            [extent[0], -extent[1], -extent[2]],
            [extent[0], extent[1], -extent[2]],
            [-extent[0], extent[1], -extent[2]],
            [-extent[0], -extent[1], -extent[2]],
            [extent[0], -extent[1], extent[2]],
            [extent[0], extent[1], extent[2]],
            [-extent[0], extent[1], extent[2]],
            [-extent[0], -extent[1], extent[2]],
        ]
    )

    return bbx


def project_world_objects(
    object_dict, output_dict, lidar_pose, lidar_range, order, dataset, enlarge_z=False
):
    """
    Project the objects under world coordinates into another coordinate
    based on the provided extrinsic.

    Parameters
    ----------
    object_dict : dict
        The dictionary contains all objects surrounding a certain cav.

    output_dict : dict
        key: object id, value: object bbx (xyzlwhyaw).

    lidar_pose : list
        (6, ), lidar pose under world coordinate, [x, y, z, roll, yaw, pitch].

    lidar_range : list
         [minx, miny, minz, maxx, maxy, maxz]

    order : str
        'lwh' or 'hwl'
    """
    for object_id, object_content in object_dict.items():
        location = object_content["location"]
        rotation = object_content["angle"]
        if dataset == "dair":
            center = (
                [0, 0, 0]
                if "center" not in object_content
                else object_content["center"]
            )
        else:
            center = object_content["center"]
        extent = object_content["extent"]

        object_pose = [
            location[0] + center[0],
            location[1] + center[1],
            location[2] + center[2],
            rotation[0],
            rotation[1],
            rotation[2],
        ]

        object2lidar = x1_to_x2(object_pose, lidar_pose)

        # shape (3, 8)
        bbx = create_bbx(extent).T
        # bounding box under ego coordinate shape (4, 8)
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]

        # project the 8 corners to world coordinate
        bbx_lidar = np.dot(object2lidar, bbx).T
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)
        bbx_lidar = corner_to_center(bbx_lidar, order=order)

        if enlarge_z:
            lidar_range_z_larger = copy.deepcopy(lidar_range)
            lidar_range_z_larger[2] -= 10
            lidar_range_z_larger[5] += 10
            lidar_range = lidar_range_z_larger

        bbx_lidar = mask_boxes_outside_range_numpy(bbx_lidar, lidar_range, order)

        if bbx_lidar.shape[0] > 0:
            output_dict.update({object_id: bbx_lidar})


def project_world_objects_airv2x(
    object_dict, output_dict, lidar_pose, lidar_range, order, dataset, enlarge_z=False
):
    """
    Project the objects under world coordinates into another coordinate
    based on the provided extrinsic.

    Parameters
    ----------
    object_dict : dict
        The dictionary contains all objects surrounding a certain cav.

    output_dict : dict
        key: object id, value: object bbx (xyzlwhyaw).

    lidar_pose : list
        (6, ), lidar pose under world coordinate, [x, y, z, roll, yaw, pitch].

    lidar_range : list
         [minx, miny, minz, maxx, maxy, maxz]

    order : str
        'lwh' or 'hwl'
    """
    for object_id, object_content in object_dict.items():
        location = object_content["location"][:3]

        angle = [
            object_content["location"][3],
            object_content["location"][4],
            object_content["location"][5],
        ]  # [roll, yaw, pitch]
        # rotation = object_content['angle']
        rotation = angle

        class_id = object_content["class"]  # airv2x use multi-class

        center = object_content["center"]
        extent = object_content["extent"]

        object_pose = [
            location[0] + center[0],
            location[1] + center[1],
            location[2] + center[2],
            rotation[0],
            rotation[1],
            rotation[2],
        ]

        object2lidar = x1_to_x2(object_pose, lidar_pose)

        # shape (3, 8)
        bbx = create_bbx(extent).T
        # bounding box under ego coordinate shape (4, 8)
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]

        # project the 8 corners to world coordinate
        bbx_lidar = np.dot(object2lidar, bbx).T
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)
        bbx_lidar = corner_to_center(bbx_lidar, order=order)

        if enlarge_z:
            lidar_range_z_larger = copy.deepcopy(lidar_range)
            lidar_range_z_larger[2] -= 10
            lidar_range_z_larger[5] += 10
            lidar_range = lidar_range_z_larger
        bbx_lidar, mask = mask_boxes_outside_range_numpy(
            bbx_lidar, lidar_range, order, return_mask=True
        )

        if bbx_lidar.shape[0] > 0:
            output_dict.update({object_id: (bbx_lidar, class_id)})


def project_world_objects_v2x(
    object_dict, output_dict, reference_lidar_pose, lidar_range, order, lidar_np
):
    """
    Project the objects under world coordinates into another coordinate
    based on the provided extrinsic.

    Parameters
    ----------
    object_dict :
        gt boxes: numpy.ndarray (N,10)
            [x,y,z,dx,dy,dz,w,a,b,c], dxdydz=lwh
        object_ids: numpy.ndarray (N,)

    output_dict : dict
        key: object id, value: object bbx (xyzlwhyaw).

    reference_lidar_pose : list
        (6, ), lidar pose under world coordinate, [x, y, z, roll, yaw, pitch].

    lidar_range : list
         [minx, miny, minz, maxx, maxy, maxz]

    order : str
        'lwh' or 'hwl'

    lidar_np: np.ndarray
        point cloud in ego coord. Used to determine if any lidar point hits the box


    output_dict: [x,y,z, lwh or hwl, yaw]
    """
    from icecream import ic

    gt_boxes = object_dict["gt_boxes"]
    object_ids = object_dict["object_ids"]
    for i, object_content in enumerate(gt_boxes):
        x, y, z, dx, dy, dz, w, a, b, c = object_content

        q = Quaternion([w, a, b, c])
        T_world_object = q.transformation_matrix
        T_world_object[:3, 3] = object_content[:3]

        T_world_lidar = x_to_world(reference_lidar_pose)

        object2lidar = np.linalg.solve(T_world_lidar, T_world_object)  # T_lidar_object

        # shape (3, 8).
        # or we can use the create_bbx funcion.
        x_corners = dx / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])  # (8,)
        y_corners = dy / 2 * np.array([-1, 1, 1, -1, -1, 1, 1, -1])
        z_corners = dz / 2 * np.array([-1, -1, -1, -1, 1, 1, 1, 1])

        bbx = np.vstack((x_corners, y_corners, z_corners))  # (3, 8)

        # bounding box under ego coordinate shape (4, 8)
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]

        # project the 8 corners to world coordinate
        bbx_lidar = np.dot(object2lidar, bbx).T  # (8, 4)
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)  # (1, 8, 3)
        bbx_lidar = corner_to_center(bbx_lidar, order=order)

        lidar_range_z_larger = copy.deepcopy(lidar_range)
        lidar_range_z_larger[2] -= 1
        lidar_range_z_larger[5] += 1

        bbx_lidar = mask_boxes_outside_range_numpy(
            bbx_lidar, lidar_range_z_larger, order
        )

        if bbx_lidar.shape[0] > 0:
            output_dict.update({object_ids[i]: bbx_lidar})


def get_points_in_rotated_box(p, box_corner):
    """
    Get points within a rotated bounding box (2D version).

    Parameters
    ----------
    p : numpy.array
        Points to be tested with shape (N, 2).
    box_corner : numpy.array
        Corners of bounding box with shape (4, 2).

    Returns
    -------
    p_in_box : numpy.array
        Points within the box.

    """
    edge1 = box_corner[1, :] - box_corner[0, :]
    edge2 = box_corner[3, :] - box_corner[0, :]
    p_rel = p - box_corner[0, :].reshape(1, -1)

    l1 = get_projection_length_for_vector_projection(p_rel, edge1)
    l2 = get_projection_length_for_vector_projection(p_rel, edge2)
    # A point is within the box, if and only after projecting the
    # point onto the two edges s.t. p_rel = [edge1, edge2] @ [l1, l2]^T,
    # we have 0<=l1<=1 and 0<=l2<=1.
    mask = np.logical_and(l1 >= 0, l1 <= 1)
    mask = np.logical_and(mask, l2 >= 0)
    mask = np.logical_and(mask, l2 <= 1)
    p_in_box = p[mask, :]
    return p_in_box


def get_points_in_rotated_box_3d(p, box_corner):
    """
    Get points within a rotated bounding box (3D version).

    Parameters
    ----------
    p : numpy.array
        Points to be tested with shape (N, 3).
    box_corner : numpy.array
        Corners of bounding box with shape (8, 3).

    Returns
    -------
    p_in_box : numpy.array
        Points within the box.

    """
    edge1 = box_corner[1, :] - box_corner[0, :]
    edge2 = box_corner[3, :] - box_corner[0, :]
    edge3 = box_corner[4, :] - box_corner[0, :]

    p_rel = p - box_corner[0, :].reshape(1, -1)

    l1 = get_projection_length_for_vector_projection(p_rel, edge1)
    l2 = get_projection_length_for_vector_projection(p_rel, edge2)
    l3 = get_projection_length_for_vector_projection(p_rel, edge3)
    # A point is within the box, if and only after projecting the
    # point onto the two edges s.t. p_rel = [edge1, edge2] @ [l1, l2]^T,
    # we have 0<=l1<=1 and 0<=l2<=1.
    mask1 = np.logical_and(l1 >= 0, l1 <= 1)
    mask2 = np.logical_and(l2 >= 0, l2 <= 1)
    mask3 = np.logical_and(l3 >= 0, l3 <= 1)

    mask = np.logical_and(mask1, mask2)
    mask = np.logical_and(mask, mask3)
    p_in_box = p[mask, :]

    return p_in_box


def get_projection_length_for_vector_projection(a, b):
    """
    Get projection length for the Vector projection of a onto b s.t.
    a_projected = length * b. (2D version) See
    https://en.wikipedia.org/wiki/Vector_projection#Vector_projection_2
    for more details.

    Parameters
    ----------
    a : numpy.array
        The vectors to be projected with shape (N, 2).

    b : numpy.array
        The vector that is projected onto with shape (2).

    Returns
    -------
    length : numpy.array
        The length of projected a with respect to b.
    """
    assert np.sum(b**2, axis=-1) > 1e-6
    length = a.dot(b) / np.sum(b**2, axis=-1)
    return length


def nms_rotated(boxes, scores, threshold):
    """Performs rorated non-maximum suppression and returns indices of kept
    boxes.

    Parameters
    ----------
    boxes : torch.tensor
        The location preds with shape (N, 4, 2).

    scores : torch.tensor
        The predicted confidence score with shape (N,)

    threshold: float
        IoU threshold to use for filtering.

    Returns
    -------
        An array of index
    """
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int32)
    boxes = boxes.cpu().detach().numpy()
    scores = scores.cpu().detach().numpy()

    polygons = common_utils.convert_format(boxes)

    top = 1000
    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1][:top]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = common_utils.compute_iou(polygons[i], polygons[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)

    return np.array(pick, dtype=np.int32)


def nms_pytorch(boxes: torch.tensor, thresh_iou: float):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.

    Parameters
    ----------
    boxes : torch.tensor
        The location preds along with the class predscores,
         Shape: [num_boxes,5].
    thresh_iou : float
        (float) The overlap thresh for suppressing unnecessary boxes.
    Returns
    -------
        A list of index
    """

    # we extract coordinates for every
    # prediction box present in P
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # we extract the confidence scores as well
    scores = boxes[:, 4]

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)

    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()

    # initialise an empty list for
    # filtered prediction boxes
    keep = []

    while len(order) > 0:
        # extract the index of the
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]

        # push S in filtered predictions list
        keep.append(
            idx.numpy().item() if not idx.is_cuda else idx.cpu().detach().numpy().item()
        )

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break

        # select coordinates of BBoxes according to
        # the indices in order
        xx1 = torch.index_select(x1, dim=0, index=order)
        xx2 = torch.index_select(x2, dim=0, index=order)
        yy1 = torch.index_select(y1, dim=0, index=order)
        yy2 = torch.index_select(y2, dim=0, index=order)

        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1

        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection area
        inter = w * h

        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim=0, index=order)

        # find the union of every prediction T in P
        # with the prediction S
        # Note that areas[idx] represents area of S
        union = (rem_areas - inter) + areas[idx]

        # find the IoU of every prediction in P with S
        IoU = inter / union

        # keep the boxes with IoU less than thresh_iou
        mask = IoU < thresh_iou
        order = order[mask]

    return keep

# TODO(YH): 
def multiclass_nms(boxes, scores, score_thr, nms_thr, max_num):
    N = boxes.shape[0]
    if N == 0:
        return np.array([], dtype=np.int32)
    boxes = boxes.cpu().detach().numpy()
    scores = scores.cpu().detach().numpy()

    num_classes = boxes.shape[1] - 1 # exclude the bg (0)
    bboxes = boxes[:, None].expand(N, num_classes, 4)


def remove_large_pred_bbx(bbx_3d, dataset):
    """
    Remove large bounding box.

    Parameters
    ----------
    bbx_3d : torch.Tensor
        Predcited 3d bounding box, shape:(N,8,3)

    Returns
    -------
    index : torch.Tensor
        The keep index.
    """
    bbx_x_max = torch.max(bbx_3d[:, :, 0], dim=1)[0]
    bbx_x_min = torch.min(bbx_3d[:, :, 0], dim=1)[0]
    x_len = bbx_x_max - bbx_x_min

    bbx_y_max = torch.max(bbx_3d[:, :, 1], dim=1)[0]
    bbx_y_min = torch.min(bbx_3d[:, :, 1], dim=1)[0]
    y_len = bbx_y_max - bbx_y_min

    if dataset == "dair":
        bbx_z_max = torch.max(bbx_3d[:, :, 1], dim=1)[0]
        bbx_z_min = torch.min(bbx_3d[:, :, 1], dim=1)[0]
    else:
        bbx_z_max = torch.max(bbx_3d[:, :, 2], dim=1)[0]
        bbx_z_min = torch.min(bbx_3d[:, :, 2], dim=1)[0]
    z_len = bbx_z_max - bbx_z_min

    index = torch.logical_and(x_len <= 6, y_len <= 6)
    index = torch.logical_and(index, z_len)

    return index


def remove_bbx_abnormal_z(bbx_3d, z_min=-3, z_max=1):
    """
    Remove bounding box that has negative z axis.

    Parameters
    ----------
    bbx_3d : torch.Tensor
        Predcited 3d bounding box, shape:(N,8,3)

    Returns
    -------
    index : torch.Tensor
        The keep index.
    """
    bbx_z_min = torch.min(bbx_3d[:, :, 2], dim=1)[0]
    bbx_z_max = torch.max(bbx_3d[:, :, 2], dim=1)[0]
    index = torch.logical_and(bbx_z_min >= z_min, bbx_z_max <= z_max)

    return index


def project_points_by_matrix_torch(points, transformation_matrix):
    """
    Project the points to another coordinate system based on the
    transformation matrix.

    Parameters
    ----------
    points : torch.Tensor
        3D points, (N, 3)
    transformation_matrix : torch.Tensor
        Transformation matrix, (4, 4)
    Returns
    -------
    projected_points : torch.Tensor
        The projected points, (N, 3)
    """
    points, is_numpy = common_utils.check_numpy_to_torch(points)
    transformation_matrix, _ = common_utils.check_numpy_to_torch(transformation_matrix)

    # convert to homogeneous coordinates via padding 1 at the last dimension.
    # (N, 4)
    points_homogeneous = F.pad(points, (0, 1), mode="constant", value=1)
    # (N, 4)
    projected_points = torch.einsum(
        "ik, jk->ij", points_homogeneous, transformation_matrix
    )

    if not is_numpy:
        return projected_points[:, :3]
    return projected_points[:, :3].numpy()


def box_encode(
    boxes,
    anchors,
    encode_angle_to_vector=False,
    encode_angle_with_residual=False,
    smooth_dim=False,
    norm_velo=False,
):
    """box encode for VoxelNet
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r.
        anchors ([N, 7] Tensor): anchors.
    """

    box_ndim = anchors.shape[-1]

    if box_ndim == 7:
        xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, wg, lg, hg, rg = torch.split(boxes, 1, dim=-1)
    else:
        xa, ya, za, wa, la, ha, vxa, vya, ra = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, wg, lg, hg, vxg, vyg, rg = torch.split(boxes, 1, dim=-1)

    diagonal = torch.sqrt(la**2 + wa**2)
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    zt = (zg - za) / ha

    if smooth_dim:
        lt = lg / la - 1
        wt = wg / wa - 1
        ht = hg / ha - 1
    else:
        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)

    ret = [xt, yt, zt, wt, lt, ht]

    if box_ndim > 7:
        if norm_velo:
            vxt = (vxg - vxa) / diagonal
            vyt = (vyg - vya) / diagonal
        else:
            vxt = vxg - vxa
            vyt = vyg - vya
        ret.extend([vxt, vyt])

    if encode_angle_to_vector:
        rgx = torch.cos(rg)
        rgy = torch.sin(rg)
        if encode_angle_with_residual:
            rax = torch.cos(ra)
            ray = torch.sin(ra)
            rtx = rgx - rax
            rty = rgy - ray
            ret.extend([rtx, rty])
        else:
            ret.extend([rgx, rgy])
    else:
        rt = rg - ra
        ret.append(rt)

    return torch.cat(ret, dim=-1)


def box_decode(
    box_encodings,
    anchors,
    encode_angle_to_vector=False,
    encode_angle_with_residual=False,
    bin_loss=False,
    smooth_dim=False,
    norm_velo=False,
):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
    """
    box_ndim = anchors.shape[-1]

    if box_ndim == 9:  # False
        xa, ya, za, wa, la, ha, vxa, vya, ra = torch.split(anchors, 1, dim=-1)
        if encode_angle_to_vector:
            xt, yt, zt, wt, lt, ht, vxt, vyt, rtx, rty = torch.split(
                box_encodings, 1, dim=-1
            )
        else:
            xt, yt, zt, wt, lt, ht, vxt, vyt, rt = torch.split(box_encodings, 1, dim=-1)

    elif box_ndim == 7:
        xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
        if encode_angle_to_vector:  # False
            xt, yt, zt, wt, lt, ht, rtx, rty = torch.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)

    diagonal = torch.sqrt(la**2 + wa**2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za

    ret = [xg, yg, zg]

    if smooth_dim:  # False
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
        hg = (ht + 1) * ha
    else:
        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
    ret.extend([wg, lg, hg])

    if encode_angle_to_vector:  # False
        if encode_angle_with_residual:
            rax = torch.cos(ra)
            ray = torch.sin(ra)
            rgx = rtx + rax
            rgy = rty + ray
            rg = torch.atan2(rgy, rgx)
        else:
            rg = torch.atan2(rty, rtx)
    else:
        rg = rt + ra

    if box_ndim > 7:  # False
        if norm_velo:
            vxg = vxt * diagonal + vxa
            vyg = vyt * diagonal + vya
        else:
            vxg = vxt + vxa
            vyg = vyt + vya
        ret.extend([vxg, vyg])

    ret.append(rg)

    return torch.cat(ret, dim=-1)


def project_world_objects_dairv2x(
    object_list, output_dict, lidar_pose, lidar_range, order
):
    """
    Project the objects under world coordinates into another coordinate
    based on the provided extrinsic.

    Parameters
    ----------
    object_list : list
        The list contains all objects surrounding a certain cav.

    output_dict : dict
        key: object id, value: object bbx (xyzlwhyaw).

    lidar_pose : list
        (6, ), lidar pose under world coordinate, [x, y, z, roll, yaw, pitch].

    lidar_range : list
         [minx, miny, minz, maxx, maxy, maxz]

    order : str
        'lwh' or 'hwl'
    """
    i = 0

    for object_content in object_list:
        object_id = i
        i = i + 1
        lidar_to_world = x_to_world(lidar_pose)  # T_world_lidar
        world_to_lidar = np.linalg.inv(lidar_to_world)

        corners_world = np.array(object_content["world_8_points"])  # [8,3]
        corners_world_homo = np.pad(
            corners_world, ((0, 0), (0, 1)), constant_values=1
        )  # [8, 4]
        corners_lidar = (world_to_lidar @ corners_world_homo.T).T

        lidar_range_z_larger = copy.deepcopy(lidar_range)
        lidar_range_z_larger[2] -= 1
        lidar_range_z_larger[5] += 1

        bbx_lidar = corners_lidar
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)  # [1, 8, 3]
        bbx_lidar = corner_to_center(bbx_lidar, order=order)
        bbx_lidar = mask_boxes_outside_range_numpy(
            bbx_lidar, lidar_range_z_larger, order
        )
        if bbx_lidar.shape[0] > 0:
            output_dict.update({object_id: bbx_lidar})


def load_single_objects_dairv2x(object_list, output_dict, lidar_range, order):
    """

    Parameters
    ----------
    object_list : list
        The list contains all objects surrounding a certain cav.

    output_dict : dict
        key: object id, value: object bbx (xyzlwhyaw).

    lidar_range : list
         [minx, miny, minz, maxx, maxy, maxz]

    order : str
        'lwh' or 'hwl'
    """

    i = 0
    for object_content in object_list:
        object_id = i
        if "rotation" not in object_content:
            print(object_content)
        x = object_content["3d_location"]["x"]
        y = object_content["3d_location"]["y"]
        z = object_content["3d_location"]["z"]
        l = object_content["3d_dimensions"]["l"]
        h = object_content["3d_dimensions"]["h"]
        w = object_content["3d_dimensions"]["w"]
        rotation = object_content["rotation"]

        if l == 0 or h == 0 or w == 0:
            continue
        i = i + 1

        lidar_range_z_larger = copy.deepcopy(lidar_range)
        lidar_range_z_larger[2] -= 1
        lidar_range_z_larger[5] += 1

        bbx_lidar = (
            [x, y, z, h, w, l, rotation]
            if order == "hwl"
            else [x, y, z, l, w, h, rotation]
        )  # suppose order is in ['hwl', 'lwh']
        bbx_lidar = np.array(bbx_lidar).reshape(1, -1)  # [1,7]

        bbx_lidar = mask_boxes_outside_range_numpy(
            bbx_lidar, lidar_range_z_larger, order
        )
        if bbx_lidar.shape[0] > 0:
            if (
                object_content["type"] == "Car"
                or object_content["type"] == "Van"
                or object_content["type"] == "Truck"
                or object_content["type"] == "Bus"
            ):
                output_dict.update({object_id: bbx_lidar})


def box_is_visible(bbx_lidar, visibility_map):
    """
    fitler bbx_lidar by visibility map.

    Parameters:

    (0,0)------------px
    |        ^ x      |
    |        |        |
    |        o---> y  |
    |                 |
    |                 |
    py-----------------(256,256)

    bbx_lidar : np.ndarray
        (1, 7), x, y, z, dx, dy, dz, yaw. dx,dy,dz follows order.

    visibility_map : np.ndarray
        (256, 256). Non zero is visible.
    """

    x, y = bbx_lidar[0, :2]

    # rasterize x and y
    py = 127 - int(x / 0.39)
    px = 127 + int(y / 0.39)

    if py < 0 or py >= 256 or px < 0 or px >= 256:
        return False

    return visibility_map[py, px] > 0


def project_world_visible_objects(
    object_dict,
    output_dict,
    lidar_pose,
    lidar_range,
    order,
    visibility_map,
    enlarge_z=False,
):
    """
    It's used by CameraDataset. Filtered by visibility map.

    Project the objects under world coordinates into another coordinate
    based on the provided extrinsic.

    Parameters
    ----------
    object_dict : dict
        The dictionary contains all objects surrounding a certain cav.

    output_dict : dict
        key: object id, value: object bbx (xyzlwhyaw).

    lidar_pose : list
        (6, ), lidar pose under world coordinate, [x, y, z, roll, yaw, pitch].

    lidar_range : list
         [minx, miny, minz, maxx, maxy, maxz]

    order : str
        'lwh' or 'hwl'

    visibility_map : np.ndarray
        for OPV2V, its 256*256 resolution. 0.39m per pixel. heading up.
    """
    for object_id, object_content in object_dict.items():
        location = object_content["location"]
        rotation = object_content["angle"]
        center = (
            [0, 0, 0] if "center" not in object_content else object_content["center"]
        )
        extent = object_content["extent"]

        object_pose = [
            location[0] + center[0],
            location[1] + center[1],
            location[2] + center[2],
            rotation[0],
            rotation[1],
            rotation[2],
        ]

        object2lidar = x1_to_x2(object_pose, lidar_pose)

        # shape (3, 8)
        bbx = create_bbx(extent).T
        # bounding box under ego coordinate shape (4, 8)
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]

        # project the 8 corners to world coordinate
        bbx_lidar = np.dot(object2lidar, bbx).T
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)
        bbx_lidar = corner_to_center(bbx_lidar, order=order)
        if enlarge_z:
            lidar_range_z_larger = copy.deepcopy(lidar_range)
            lidar_range_z_larger[2] -= 10
            lidar_range_z_larger[5] += 10
            lidar_range = lidar_range_z_larger

        bbx_lidar = mask_boxes_outside_range_numpy(bbx_lidar, lidar_range, order)

        if bbx_lidar.shape[0] > 0 and box_is_visible(bbx_lidar, visibility_map):
            output_dict.update({object_id: bbx_lidar})



def convert_boxes_to_format(boxes3d, output_format='extend', order='lwh'):
    """
    Convert 3D bounding boxes from 7-DoF format [x, y, z, l, w, h, heading] 
    to different output formats.
    
    Parameters
    ----------
    boxes3d : np.ndarray or torch.Tensor
        (N, 7) [x, y, z, l, w, h, heading], or [x, y, z, h, w, l, heading].
        (x, y, z) is the box center.
    output_format : str
        '9dof' for [x, y, z, l, w, h, roll, pitch, yaw]
        'extend' for [x, y, z, roll, pitch, yaw, extend_x, extend_y, extend_z]
    order : str
        'lwh' or 'hwl'.
        
    Returns
    -------
    transformed_boxes : np.ndarray or torch.Tensor
        Boxes in the requested output format
    """
    # Convert numpy array to torch tensor if needed
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
    
    # Rearrange parameters if the order is 'hwl'
    if order == 'hwl':
        # Reorder from [x, y, z, h, w, l, heading] to [x, y, z, l, w, h, heading]
        boxes3d = boxes3d[:, [0, 1, 2, 5, 4, 3, 6]]
    
    # Set roll and pitch to zero
    zeros = torch.zeros_like(boxes3d[:, 0:1])
    
    if output_format == '9dof':
        # Create format: [x, y, z, l, w, h, roll, pitch, yaw]
        transformed_boxes = torch.cat([boxes3d[:, 0:6], zeros, zeros, boxes3d[:, 6:7]], dim=1)
    
    elif output_format == 'extend':
        # Create format: [x, y, z, roll, pitch, yaw, extend_x, extend_y, extend_z]
        # where extend_* is half the dimension (l/2, w/2, h/2)
        transformed_boxes = torch.cat([
            boxes3d[:, 0:3],  # x, y, z
            zeros, zeros, boxes3d[:, 6:7],  # roll, pitch, yaw
            boxes3d[:, 3:4] / 2,  # extend_x = l/2
            boxes3d[:, 4:5] / 2,  # extend_y = w/2
            boxes3d[:, 5:6] / 2,  # extend_z = h/2
        ], dim=1)
    
    else:
        raise ValueError(f"Unsupported output_format: {output_format}. Use '9dof' or 'extend'.")
    
    # Convert back to numpy if the input was a numpy array
    if is_numpy:
        transformed_boxes = transformed_boxes.numpy()
    
    return transformed_boxes


def corner_3d_to_params(corners3d, order='lwh'):
    """
    Convert 3D corners to box parameters.
    
    Parameters
    __________
    corners3d: np.ndarray or torch.Tensor
        (N, 8, 3), the 8 corners of the bounding box.
        Corner points should follow the order defined in boxes_to_corners_3d:
           4 -------- 5
          /|         /|
         7 -------- 6 .
         | |        | |
         . 0 -------- 1
         |/         |/
         3 -------- 2
    
    order : str
        'lwh' or 'hwl'
    
    Returns:
        boxes3d: np.ndarray or torch.Tensor
        (N, 9) [x, y, z, row, pitch, yaw, length, width, height]
        
        (x, y, z) is the box center.
        (row, pitch, yaw) are rotation angles in radians.
    """
    corners3d, is_numpy = common_utils.check_numpy_to_torch(corners3d)
    
    # Compute center as mean of all corners
    center = corners3d.mean(dim=1)  # (N, 3)
    
    # Based on the corner ordering in the template:
    #    4 -------- 5
    #   /|         /|
    #  7 -------- 6 .
    #  | |        | |
    #  . 0 -------- 1
    #  |/         |/
    #  3 -------- 2
    
    # Calculate dimensions directly from opposite corners
    # These represent the main diagonals of the box
    diag_1 = corners3d[:, 6] - corners3d[:, 0]  # from 0 to 6
    diag_2 = corners3d[:, 5] - corners3d[:, 3]  # from 3 to 5
    diag_3 = corners3d[:, 7] - corners3d[:, 1]  # from 1 to 7
    diag_4 = corners3d[:, 4] - corners3d[:, 2]  # from 2 to 4
    
    # Define edges along principal axes of the box
    # These are the edges of the box along its local coordinate axes
    x_edges = torch.stack([
        corners3d[:, 1] - corners3d[:, 0],  # Edge from 0 to 1 (along local x-axis)
        corners3d[:, 2] - corners3d[:, 3],  # Edge from 3 to 2
        corners3d[:, 5] - corners3d[:, 4],  # Edge from 4 to 5
        corners3d[:, 6] - corners3d[:, 7],  # Edge from 7 to 6
    ], dim=1)
    
    y_edges = torch.stack([
        corners3d[:, 3] - corners3d[:, 0],  # Edge from 0 to 3 (along local y-axis)
        corners3d[:, 2] - corners3d[:, 1],  # Edge from 1 to 2
        corners3d[:, 7] - corners3d[:, 4],  # Edge from 4 to 7
        corners3d[:, 6] - corners3d[:, 5],  # Edge from 5 to 6
    ], dim=1)
    
    z_edges = torch.stack([
        corners3d[:, 4] - corners3d[:, 0],  # Edge from 0 to 4 (along local z-axis)
        corners3d[:, 5] - corners3d[:, 1],  # Edge from 1 to 5
        corners3d[:, 6] - corners3d[:, 2],  # Edge from 2 to 6
        corners3d[:, 7] - corners3d[:, 3],  # Edge from 3 to 7
    ], dim=1)
    
    # Compute dimensions (average edge lengths along each axis)
    length = torch.norm(x_edges, dim=2).mean(dim=1)  # (N,)
    width = torch.norm(y_edges, dim=2).mean(dim=1)   # (N,)
    height = torch.norm(z_edges, dim=2).mean(dim=1)  # (N,)
    
    # Compute unit vectors along each axis of the box
    # Average all edges along the same direction for robustness
    x_vec = x_edges.mean(dim=1)  # (N, 3)
    y_vec = y_edges.mean(dim=1)  # (N, 3)
    z_vec = z_edges.mean(dim=1)  # (N, 3)
    
    # Normalize vectors
    x_norm = torch.norm(x_vec, dim=1, keepdim=True)
    y_norm = torch.norm(y_vec, dim=1, keepdim=True)
    z_norm = torch.norm(z_vec, dim=1, keepdim=True)
    
    x_unit = x_vec / (x_norm + 1e-8)
    y_unit = y_vec / (y_norm + 1e-8)
    z_unit = z_vec / (z_norm + 1e-8)
    
    # Compute yaw angle (rotation around z-axis)
    # yaw is the angle between the x-axis of the box and the global x-axis in the xy-plane
    yaw = torch.atan2(x_unit[:, 1], x_unit[:, 0])
    
    # Compute pitch angle (rotation around y-axis)
    # pitch is the angle between the x-axis of the box projected onto the xz-plane and the global x-axis
    pitch = torch.atan2(-x_unit[:, 2], torch.sqrt(x_unit[:, 0]**2 + x_unit[:, 1]**2))
    
    # Compute roll angle (rotation around x-axis)
    # Use the y-axis and z-axis unit vectors
    # First, create a reference vector that lies in the yz-plane
    ref_y = torch.zeros_like(y_unit)
    ref_y[:, 1] = 1.0  # Unit vector along global y-axis
    
    # Compute the reference z-vector (should be perpendicular to x_unit and ref_y)
    ref_z = torch.cross(x_unit, ref_y, dim=1)
    ref_z = ref_z / (torch.norm(ref_z, dim=1, keepdim=True) + 1e-8)
    
    # Project y_unit onto the plane defined by ref_y and ref_z
    # This gives us the component of y_unit in the yz-plane after accounting for yaw and pitch
    y_proj = y_unit - torch.sum(y_unit * x_unit, dim=1, keepdim=True) * x_unit
    y_proj = y_proj / (torch.norm(y_proj, dim=1, keepdim=True) + 1e-8)
    
    # Roll is the angle between y_proj and ref_y in the plane perpendicular to x_unit
    cos_roll = torch.sum(y_proj * ref_y, dim=1)
    sin_roll = torch.sum(torch.cross(ref_y, y_proj, dim=1) * x_unit, dim=1)
    roll = torch.atan2(sin_roll, cos_roll)
    
    # Stack all parameters
    params = torch.stack([
        center[:, 0], center[:, 1], center[:, 2],  # x, y, z
        roll, pitch, yaw,                          # roll, pitch, yaw
        length, width, height                      # length, width, height
    ], dim=1)
    
    # Adjust order if necessary
    if order == 'hwl':
        # Swap length and height
        params_hwl = params.clone()
        params_hwl[:, 6] = params[:, 8]  # height becomes length
        params_hwl[:, 7] = params[:, 7]  # width stays the same
        params_hwl[:, 8] = params[:, 6]  # length becomes height
        params = params_hwl
    
    return params.numpy() if is_numpy else params
def multiclass_nms_rotated(boxes, scores, labels, threshold):
    """
    Multiclass rotated NMS.

    Parameters
    ----------
    boxes : torch.Tensor
        Rotated boxes of shape (N, 4, 2) — 4 corners per box.
    scores : torch.Tensor
        Confidence scores of shape (N,)
    labels : torch.Tensor
        Class labels of shape (N,)
    threshold : float
        IoU threshold for NMS.

    Returns
    -------
    keep_indices : np.ndarray
        Array of indices of kept boxes after multiclass NMS.
    """
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int32)

    boxes = boxes.cpu().detach()
    scores = scores.cpu().detach()
    labels = labels.cpu().detach()

    keep = []

    unique_classes = labels.unique()
    for cls in unique_classes:
        cls_mask = labels == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = torch.where(cls_mask)[0]  # original indices

        # Convert to numpy
        cls_boxes_np = cls_boxes.numpy()
        cls_scores_np = cls_scores.numpy()
        cls_polygons = common_utils.convert_format(cls_boxes_np)

        top = 1000
        ixs = cls_scores_np.argsort()[::-1][:top]
        pick = []
        while len(ixs) > 0:
            i = ixs[0]
            pick.append(i)
            iou = common_utils.compute_iou(cls_polygons[i], [cls_polygons[j] for j in ixs[1:]])
            remove_ixs = np.where(iou > threshold)[0] + 1
            ixs = np.delete(ixs, remove_ixs)
            ixs = np.delete(ixs, 0)

        # Map back to original indices
        keep.extend(cls_indices[pick].tolist())

    return np.array(keep, dtype=np.int32)