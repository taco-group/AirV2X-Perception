# -*- coding: utf-8 -*-
"""
This script contains the transformations between world and different sensors.
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
import bisect

import numpy as np
import torch
from matplotlib import cm

from opencood.utils.opencda_carla import Transform

VIRIDIS = np.array(cm.get_cmap("viridis").colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


def project_lidar_to_camera(
    index, rgb_image, point_cloud, camera_intrinsic, image_size
):
    """
    Project lidar to camera space.

    Parameters
    ----------
    rgb_image : np.ndarray
        RGB image from camera.

    point_cloud : np.ndarray
        Cloud points, shape: (n, 4).

    camera_intrinsic:
        cam_K, shape (3, 3).

    image_size:
        (image_h, image_w).

    camera : carla.sensor
        RGB camera.

    Returns
    -------
    rgb_image : np.ndarray
        New rgb image with lidar points projected.

    points_2d : np.ndarrya
        Point cloud projected to camera space.

    """
    # x=11, y=11, z=11, pitch=0, yaw=, roll=0
    """
    transform_params = [
        [11, 11, 11, 0, 0, 0],
        [11, 11, 11, 0, 260, 0],
        [11, 11, 11, 0, 100, 0],
        [11, 11, 11, 0, 180, 0]
    ]
    """
    transform_params = [
        [7, 10, 11, 0, 0, 0],
        [7, 10, 11, 0, 260, 0],
        [7, 10, 11, 0, 100, 0],
        [8, 10, 11, 0, 180, 0],
    ]

    # Lidar intensity array of shape (p_cloud_size,) but, for now, let's
    # focus on the 3D points.
    intensity = np.array(point_cloud[:, 3])

    # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
    local_lidar_points = np.array(point_cloud[:, :3]).T

    # Add an extra 1.0 at the end of each 3d point so it becomes of
    # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
    local_lidar_points = np.r_[
        local_lidar_points, [np.ones(local_lidar_points.shape[1])]
    ]

    # This (4, 4) matrix transforms the points from lidar space to world space.
    # lidar_2_world = x_to_world_transformation(lidar.get_transform())
    lidar_2_world = x_to_world_transformation(Transform(*transform_params[index]))

    # transform lidar points from lidar space to world space
    world_points = np.dot(lidar_2_world, local_lidar_points)

    # project lidar world points to camera space
    # sensor_points = world_to_sensor(world_points, camera.get_transform())
    sensor_points = world_to_sensor(world_points, Transform(x=10, y=10, z=10))

    # New we must change from UE4's coordinate system to an "standard"
    # camera coordinate system (the same used by OpenCV):

    # ^ z                       . z
    # |                        /
    # |              to:      +-------> x
    # | . x                   |
    # |/                      |
    # +-------> y             v y

    # (x, y ,z) -> (y, -z, x)
    point_in_camera_coords = np.array(
        [sensor_points[1], sensor_points[2] * -1, sensor_points[0]]
    )

    # retrieve camera intrinsic
    K = camera_intrinsic
    image_h, image_w = int(image_size[0]), int(image_size[1])

    # project the 3d points in camera space to image space
    points_2d = np.dot(K, point_in_camera_coords)

    # normalize x,y,z
    points_2d = np.array(
        [
            points_2d[0, :] / points_2d[2, :],
            points_2d[1, :] / points_2d[2, :],
            points_2d[2, :],
        ]
    )

    # remove points out the camera scope
    points_2d = points_2d.T
    intensity = intensity.T
    points_in_canvas_mask = (
        (points_2d[:, 0] > 0.0)
        & (points_2d[:, 0] < image_w)
        & (points_2d[:, 1] > 0.0)
        & (points_2d[:, 1] < image_h)
        & (points_2d[:, 2] > 0.0)
    )
    # print(points_in_canvas_mask.shape, intensity.shape)
    new_points_2d = points_2d[points_in_canvas_mask]
    new_intensity = intensity[points_in_canvas_mask]

    # Extract the screen coords (uv) as integers.
    u_coord = new_points_2d[:, 0].astype(np.int)
    v_coord = new_points_2d[:, 1].astype(np.int)

    # Since at the time of the creation of this script, the intensity function
    # is returning high values, these are adjusted to be nicely visualized.
    new_intensity = 4 * new_intensity - 3
    color_map = (
        np.array(
            [
                np.interp(new_intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0,
                np.interp(new_intensity, VID_RANGE, VIRIDIS[:, 1]) * 255.0,
                np.interp(new_intensity, VID_RANGE, VIRIDIS[:, 2]) * 255.0,
            ]
        )
        .astype(np.int)
        .T
    )

    for i in range(len(new_points_2d)):
        rgb_image[v_coord[i] - 1 : v_coord[i] + 1, u_coord[i] - 1 : u_coord[i] + 1] = (
            color_map[i]
        )

    return rgb_image, points_2d


class Location(object):
    """A mock class for Location."""

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Rotation(object):
    """A mock class for Rotation."""

    def __init__(self, pitch, yaw, roll):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll


class Transform(object):
    """A mock class for transform"""

    def __init__(self, x, y, z, pitch=0, yaw=0, roll=0):
        self.location = Location(x, y, z)
        self.rotation = Rotation(pitch, yaw, roll)


class Camera(object):
    """A mock class for camera."""

    def __init__(self, attributes: dict):
        self.attributes = attributes
        self.transform = Transform(x=10, y=10, z=10)

    def get_transform(self):
        return self.transform


class Lidar(object):
    """A mock class for lidar."""

    def __init__(self, attributes: dict):
        self.attributes = attributes
        self.transform = Transform(x=11, y=11, z=11)

    def get_transform(self):
        return self.transform


def x_to_world_transformation(transform):
    """
    Get the transformation matrix from x(it can be vehicle or sensor)
    coordinates to world coordinate.

    Parameters
    ----------
    transform : carla.Transform
        The transform that contains location and rotation

    Returns
    -------
    matrix : np.ndarray
        The transformation matrx.

    """
    rotation = transform.rotation
    location = transform.location

    # used for rotation matrix
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def world_to_sensor(cords, sensor_transform):
    """
    Transform coordinates from world reference to sensor reference.

    Parameters
    ----------
    cords : np.ndarray
        Coordinates under world reference, shape: (4, n).

    sensor_transform : carla.Transform
        Sensor position in the world.

    Returns
    -------
    sensor_cords : np.ndarray
        Coordinates in the sensor reference.

    """
    sensor_world_matrix = x_to_world_transformation(sensor_transform)
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    sensor_cords = np.dot(world_sensor_matrix, cords)

    return sensor_cords
