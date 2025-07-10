# Author: Yuheng Wu <yuhengwu@kaist.ac.kr>

"""
utilities for airv2x
"""

from typing import Dict, Any, List, Union, Tuple
import yaml
import pickle
import heapq
import os
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union
import matplotlib.pyplot as plt

import numpy as np
import torch
import yaml
from matplotlib import cm
from tqdm import tqdm
import matplotlib.colors as mcolors


from opencood.hypes_yaml.yaml_utils import load_pickle, load_yaml
from opencood.utils.transformation_utils import (
    get_abs_world_pose,
    tfm_to_pose,
    x1_to_x2,
    x_to_world,
)



RSU_FILES = [
    "back_camera.png",
    "front_camera.png",
    "left_camera.png",
    "right_camera.png",
    
    "back_depth.png",
    "front_depth.png",
    "left_depth.png",
    "right_depth.png",
    
    "lidar.pcd",
    "semantic_lidar.pcd",
    "semantic_lidar_semantic.npz",
    "vector_map.json",
    # "map_dynamic_bev.jpeg",
    # "map_static_bev.png",
    # "map_vis_bev.jpeg",
    "metadata.pkl",
    # "static_background.png",
    # "static_lane.png",
    # "static_road.png",
    "map_static_background.png",
    "map_static_lane.png",
    "map_static_road.png",
] + [f"map_dynamic_bev_layer_{i}.png" for i in range(7)]  # 0 to 10 inclusive



VEHICLE_FILES = [
    "front_camera.png",
    "front_left_camera.png",
    "front_right_camera.png",
    "rear_camera.png",
    "rear_left_camera.png",
    "rear_right_camera.png",
    
    "front_depth.png",
    "front_left_depth.png",
    "front_right_depth.png",
    "rear_depth.png",
    "rear_left_depth.png",
    "rear_right_depth.png",
    
    "lidar.pcd",
    "semantic_lidar.pcd",
    "semantic_lidar_semantic.npz",
    "vector_map.json",
    # "map_dynamic_bev.jpeg",
    # "map_static_bev.jpeg",
    # "map_vis_bev.jpeg",
    "metadata.pkl",
    # "static_background.png",
    # "static_lane.png",
    # "static_road.png",
    "map_static_background.png",
    "map_static_lane.png",
    "map_static_road.png",
] + [f"map_dynamic_bev_layer_{i}.png" for i in range(7)]

DRONE_FILES = [
    "bev_camera.png",
    
    "bev_depth.png",
    
    "lidar.pcd",
    "vector_map.json",
    # "map_dynamic_bev.jpeg",
    # "map_static_bev.jpeg",
    # "map_vis_bev.jpeg",
    "metadata.pkl",
    # "static_background.png",
    # "static_lane.png",
    # "static_road.png",
    "map_static_background.png",
    "map_static_lane.png",
    "map_static_road.png",
] + [f"map_dynamic_bev_layer_{i}.png" for i in range(7)]  # 0–10 layers


VIRIDIS = np.array(cm.get_cmap("viridis").colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


def parse_agent_idx(agent_str: str) -> int:
    agent_str = os.path.basename(os.path.normpath(agent_str))
    return int(agent_str.split("_")[1])


def parse_timestamp_idx(timestamp_str: str) -> int:
    timestamp_str = os.path.basename(os.path.normpath(timestamp_str))
    return int(timestamp_str.split("_")[1])


def parse_timestamp_agent(agent_path: str) -> Dict[str, Any]:
    """parse the agent info at certain timestamp

    eg. dataset/airv2x/2025_04_09_22_34_07/timestamp_000000/agent_001514

    Args:
        agent_path (str): path

    Returns:
        Dict[str, Any]: [category, list of files]
    """
    info = OrderedDict()
    metadata_path = os.path.join(agent_path, "metadata.pkl")
    metadata = load_pickle(metadata_path)
    # info["metadata"] = metadata
    agent_type = metadata["agent_type"]
    if agent_type == "rsu":
        target_files = RSU_FILES
    elif agent_type == "vehicle":
        target_files = VEHICLE_FILES
    elif agent_type == "drone":
        target_files = DRONE_FILES
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # collate file path in order
    for filename in target_files:
        file_path = os.path.join(agent_path, filename)
        if os.path.isfile(file_path):
            if "camera" in filename:
                # rsu: [back, front, left, right]
                # vehicle: [front, fl, fr, rear, rl, rr]
                # drone: [bev_cam]
                info.setdefault("cameras", []).append(file_path)
            elif "depth" in filename:
                info.setdefault("depth", []).append(file_path)
            elif "lidar" in filename:
                # rsu/vehicle: [lidar, semantic_liadr, semantic_lidar_semantic.npz]
                # drone:  []
                info.setdefault("lidars", []).append(file_path)
            elif "map" in filename:
                # rsu/vehicle/drone: [vector_map, map_dynamic_bev, map_static_bev, map_vis_bev, map_dynamic_bev_layer{i}]
                info.setdefault("map", []).append(file_path)
            else:
                info["metadata_path"] = file_path
                info["agent_type"] = agent_type
    return info


def parse_timestamp(path: str) -> Dict[Union[str, int], Any]:
    """parse all agents at this timestamp

    eg. dataset/airv2x/2025_04_09_22_34_07/timestamp_000000/<agent_idx>/

    Args:
        path (str): path to the timestamp

    Returns:
        Dict[Union[str, int], Any]: [agent_idx, agent_info]
    """

    data = OrderedDict()
    objs_path = os.path.join(path, "objects.pkl")
    agent_paths = sorted([os.path.join(path, subpath) for subpath in os.listdir(path)])
    for agent_path in agent_paths:
        # skip objects.yaml
        if os.path.isfile(agent_path):
            continue
        else:
            agent_idx = parse_agent_idx(agent_path)
            agent_info = parse_timestamp_agent(agent_path)
            agent_info["objects"] = objs_path  # attach objs path to all agents
            data[agent_idx] = agent_info
    return data


def parse_seq(path: str, opv2vformat: bool = True) -> Dict[int, Any]:
    """parse the given sequence

    eg. /code/dataset/airv2x/2025_04_09_22_34_07/<timestamp>/<agent>

    Args:
        path (str): path ot sequence
        opv2vformat (bool): convert to opv2v format seq/agent/timestamp, airv2x is seq/timestamp/agent

    Returns:
        Dict[int, Any]: agents under each timestamp of this sequence
    """
    seq_data = OrderedDict()
    sub_timestamps = sorted(
        [os.path.join(path, timestamp) for timestamp in os.listdir(path)]
    )
    t = 0  # for debugging
    for sub_timestamp in sub_timestamps:
        if not os.path.isdir(sub_timestamp):
            continue
        t += 1
        timestamp_idx = parse_timestamp_idx(sub_timestamp)
        timestamp_data = parse_timestamp(sub_timestamp)
        seq_data[timestamp_idx] = timestamp_data
        # if t > 5:
        #     break
    if opv2vformat:
        seq_data = convert2opv2v(seq_data)
    return seq_data


def filter_objects(objects) -> Dict[int, Any]:
    """
    Given a loaded objects.pkl file, parse the objects.
    Current implementation is used to remove some object types.
    """

    filtered_objects = dict()
    for key, obj in objects.items():
        if obj["class"] in [1, 2, 3, 4, 5, 6]:
            filtered_objects[key] = obj

    return filtered_objects
    
    # return objects


def convert2opv2v(seq_dict) -> Dict[int, Any]:
    # airv2x: seq/timestamp/agent
    # opv2v: seq/agent/timestamp
    opv2v_format_dict = OrderedDict()
    for timestamp_idx, agent_timestamp_dict in seq_dict.items():
        for agent_idx, agent_info in agent_timestamp_dict.items():
            if agent_idx not in opv2v_format_dict:
                opv2v_format_dict[agent_idx] = OrderedDict()
            opv2v_format_dict[agent_idx][timestamp_idx] = agent_info
    return opv2v_format_dict


def get_ex_intrinsic(metainfo) -> Tuple[np.ndarray, np.ndarray]:
    # metainfo = load_yaml(metafile)
    # drone doesn't have extrinsic and need special processing
    # assert metainfo["agent_type"] != "drone"
    intrinsics = []
    extrinsics = []
    camera_list = []
    
    
    # TODO: make the code more elegant
    vehicle_cam = [
        "front_camera",
        "front_left_camera",
        "front_right_camera",
        "rear_camera",
        "rear_left_camera",
        "rear_right_camera"
        ]
    
    rsu_cam = [
        "back_camera",
        "front_camera",
        "left_camera",
        "right_camera",
    ]
    
    drone_cam = ["bev_camera"]
    
    
    if metainfo["agent_type"] == "vehicle":
        cam = vehicle_cam
    elif metainfo["agent_type"] == "rsu":
        cam = rsu_cam
    elif metainfo["agent_type"] == "drone":
        cam = drone_cam
    else:
        raise ValueError(f"Unknown agent type: {metainfo['agent_type']}")
    
    for k in cam:
        v = metainfo[k]
        intrinsic = np.array(v["intrinsic"], dtype=np.float32)
        extrinsic = np.array(v["extrinsic"], dtype=np.float32)
        intrinsics.append(intrinsic)
        extrinsics.append(extrinsic)
        camera_list.append(k)

    # import pdb; pdb.set_trace()
    # # ['front_camera', 'front_left_camera', 'front_right_camera', 'rear_camera', 'rear_left_camera', 'rear_right_camera']
    return np.array(intrinsics), np.array(extrinsics)


def project_lidar_to_cam_single(
    lidar_np, cam_intrinsic, imgH, imgW, cam_pos, lidar_pos, vis_image
):
    # only used for vehicle and rsu, drone doesn't have lidar, instead we use prediction depth map for drone as in LSS
    intensity = lidar_np[:, 3]

    # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
    local_lidar_points = lidar_np[:, :3].T  # (3, pcd size)

    # Add an extra 1.0 at the end of each 3d point so it becomes of
    # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
    local_lidar_points = np.r_[
        local_lidar_points, [np.ones(local_lidar_points.shape[1])]
    ]  # (4, pcd size)

    # project lidar world points to camera space using inv(intrinsic)
    lidar2cam = x1_to_x2(lidar_pos, cam_pos)

    # we only need the [x, y, z]
    cam_points = np.dot(lidar2cam, local_lidar_points)[
        :3, :
    ]  # (4, pcd size) -> (3, pcd size)

    # New we must change from UE4's coordinate system to an "standard"
    # camera coordinate system (the same used by OpenCV):

    # ^ z                       . z
    # |                        /
    # |              to:      +-------> x
    # | . x                   |
    # |/                      |
    # +-------> y             v y

    cam_points = np.array([cam_points[1], cam_points[2] * -1, cam_points[0]])

    points_2d = np.dot(
        cam_intrinsic, cam_points
    )  # (3, 3) (3, pcd size) -> (3, pcd size)
    points_2d = np.array(
        [
            points_2d[0, :] / points_2d[2, :],
            points_2d[1, :] / points_2d[2, :],
            points_2d[2, :],
        ]
    )

    # remove points out the camera scope
    points_2d = points_2d.T  # (pcd size, 3)
    intensity = intensity.T
    points_in_canvas_mask = (
        (points_2d[:, 0] > 0.0)
        & (points_2d[:, 0] < imgW)
        & (points_2d[:, 1] > 0.0)
        & (points_2d[:, 1] < imgH)
        & (points_2d[:, 2] > 0.0)  # can only see points in front of cam
    )

    # visualize depth
    new_points_2d = points_2d[points_in_canvas_mask]  # (pcd size, 3)
    new_intensity = intensity[points_in_canvas_mask]
    # Extract the screen coords (uv) as integers.
    u_coord = new_points_2d[:, 0].astype(np.int32)
    v_coord = new_points_2d[:, 1].astype(np.int32)

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
        .astype(np.int32)
        .T
    )

    for i in range(len(new_points_2d)):
        vis_image[v_coord[i] - 1 : v_coord[i] + 1, u_coord[i] - 1 : u_coord[i] + 1] = (
            color_map[i]
        )

    return vis_image, points_2d


def get_bevcam_visible_region_from_intrinsic(
    bevcam_world_pos, ego_lidar_pos, intrinsic, image_shape=(720, 1280)
):
    """
    Project the image corners through the camera model and pose, and return
    the bounding box [x1, y1, x2, y2] on the ground plane (z=0).

    Parameters
    ----------
    bevcam_world_pos : list
        [x, y, z, roll, yaw, pitch] in degrees (world coordinates of the camera).
    intrinsic : np.ndarray
        3x3 intrinsic matrix of the camera.
    image_shape : tuple
        (height, width) of the image in pixels.

    Returns
    -------
    list of float : [x_min, y_min, x_max, y_max] in world coordinates
    """
    H, W = image_shape
    # we only have one camera for each drone
    fx, fy = intrinsic[0, 0, 0], intrinsic[0, 1, 1]
    cx, cy = intrinsic[0, 0, 2], intrinsic[0, 1, 2]

    # 4 corners in pixel coordinates: (u, v)
    corners_uv = np.array(
        [
            [0, 0],  # top-left
            [W - 1, 0],  # top-right
            [0, H - 1],  # bottom-left
            [W - 1, H - 1],  # bottom-right
        ]
    )

    # Step 1: normalize image plane coordinates
    rays_cam = []
    for u, v in corners_uv:
        x = (u - cx) / fx
        y = (v - cy) / fy
        ray = np.array([x, y, 1.0])  # direction in OpenCV cam frame
        ray /= np.linalg.norm(ray)
        rays_cam.append(ray)

    rays_cam = np.stack(rays_cam)  # (4, 3)

    # Step 2: Convert OpenCV camera frame to CARLA frame
    R_opencv_to_carla = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
    rays_carla = (R_opencv_to_carla @ rays_cam.T).T  # (4, 3)

    # Step 3: Get camera-to-world transform
    T_world_cam = x_to_world(bevcam_world_pos)
    cam_pos = T_world_cam[:3, 3]  # camera position in world
    rays_world = (T_world_cam[:3, :3] @ rays_carla.T).T  # (4, 3)

    # Step 4: Intersect rays with ground plane (z = 0)
    ground_points = []
    for ray in rays_world:
        if abs(ray[2]) < 1e-6:
            continue  # skip nearly parallel rays
        t = -cam_pos[2] / ray[2]
        point = cam_pos + t * ray
        ground_points.append(point[:2])  # only x, y

    ground_points = np.stack(ground_points, axis=0)
    x_min, y_min = np.min(ground_points, axis=0)
    x_max, y_max = np.max(ground_points, axis=0)

    # transform to ego lidar
    world2lidar = np.linalg.inv(x_to_world(ego_lidar_pos))

    corners_world = np.array(
        [
            [x_min, y_min, 0, 1],
            [x_min, y_max, 0, 1],
            [x_max, y_min, 0, 1],
            [x_max, y_max, 0, 1],
        ]
    )

    corners_lidar = (world2lidar @ corners_world.T).T  # (4, 4)
    x_l = corners_lidar[:, 0]
    y_l = corners_lidar[:, 1]
    x_min = np.min(x_l)
    y_min = np.min(y_l)
    x_max = np.max(x_l)
    y_max = np.max(y_l)

    return [x_min, y_min, -3, x_max, y_max, 1]


def get_smallest_k_indices(distance_to_ego, k):
    """
    Get the indices of the k smallest distances to ego.

    Args:
        distance_to_ego (list): List of distances to ego. [(cav_id, distance), ...]
        k (int): Number of smallest distances to find.

    Returns:
        tuple: Indices of the k smallest distances and their corresponding cav_ids.
    """
    smallest_k = heapq.nsmallest(k, enumerate(distance_to_ego), key=lambda x: x[1][1])
    smallest_k_cav_ids = set(cav_id for _, (cav_id, _) in smallest_k)

    indices = []
    cav_ids = []
    # NOTE(YH): set may shuffle the order, so we code as following
    for idx, (cav_id, _) in enumerate(distance_to_ego):
        if cav_id in smallest_k_cav_ids:
            indices.append(idx)
            cav_ids.append(cav_id)
            if len(indices) == k:
                break

    # indices = [idx for idx, _ in smallest_k]
    # cav_ids = [distance_to_ego[idx][0] for idx in indices]
    return indices, cav_ids


def merge_lidar_cam_feats(processed_feature_list):
    merged_feature_dict = {}
    for i in range(len(processed_feature_list)):
        for feature_name, feature in processed_feature_list[i].items():
            if feature_name not in merged_feature_dict:
                merged_feature_dict[feature_name] = []
            merged_feature_dict[feature_name].append(feature)
    for feature_name, feature in merged_feature_dict.items():
        if isinstance(feature, list):
            merged_feature_dict[feature_name] = torch.cat(feature, dim=0)
    return merged_feature_dict


def mock_lidar_for_drone(drone_record_len, device):
    # dict_keys(['voxel_features', 'voxel_coords', 'voxel_num_points', 'record_len', 'pillar_features', 'spatial_features_3d', 'spatial_features'])
    # voxel_features: torch.Size([5999, 32, 4])
    # voxel_coords: torch.Size([5999, 4])
    # voxel_num_points: torch.Size([5999])
    # record_len: [batch_size]
    # pillar_features: torch.Size([5999, 64])
    # spatial_features_3d: torch.Size([4, 64, 1, 200, 704])
    # spatial_features: torch.Size([4, 64, 200, 704])
    voxel_features = torch.zeros((1, 32, 4), dtype=torch.float32).to(
        device
    )  # 1 dummy voxel
    voxel_coords = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32).to(device)
    voxel_num_points = torch.tensor([0], dtype=torch.int32).to(device)
    record_len = torch.tensor([drone_record_len], dtype=torch.int32).to(device)
    pillar_features = torch.zeros((1, 64), dtype=torch.float32).to(device)
    spatial_features_3d = torch.zeros(
        (drone_record_len, 64, 1, 200, 704), dtype=torch.float32
    ).to(device)
    spatial_features = torch.zeros(
        (drone_record_len, 64, 200, 704), dtype=torch.float32
    ).to(device)
    pc_dict = {
        "voxel_features": voxel_features,
        "voxel_coords": voxel_coords,
        "voxel_num_points": voxel_num_points,
        "record_len": record_len,
        "pillar_features": pillar_features,
        "spatial_features_3d": spatial_features_3d,
        "spatial_features": spatial_features,
    }
    return pc_dict


def visualize_cls_labels_bev(cls_labels, method="max"):
    """
    Visualize the cls_labels as a BEV heatmap.

    Parameters
    ----------
    cls_labels : np.ndarray
        Shape (H, W, A), integer class labels per anchor.
    method : str
        Method to reduce anchor dimension. Options: 'max', 'first_nonzero'.
    """
    H, W, A = cls_labels.shape

    cls_labels = cls_labels.astype(int)
    cls_map = np.max(cls_labels, axis=2)

    num_classes = np.max(cls_map) + 1
    bounds = np.arange(num_classes + 1)
    norm = mcolors.BoundaryNorm(bounds, ncolors=num_classes, clip=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(
        cls_map.T, cmap="tab20", origin="lower", interpolation="nearest", norm=norm
    )
    plt.colorbar(ticks=np.arange(num_classes), label="Class ID")

    # # Display as BEV
    # plt.figure(figsize=(8, 6))
    # plt.imshow(cls_map.T, cmap='tab20', origin='lower')  # Transpose for (x, y)
    # plt.colorbar(label='Class ID')
    plt.title("BEV Visualization of Class Labels")
    plt.xlabel("X (BEV Width)")
    plt.ylabel("Y (BEV Height)")
    plt.tight_layout()
    plt.savefig("dummy_images/cls_vis.png")


if __name__ == "__main__":
    # # timestamp = "/code/dataset/airv2x/2025_04_09_22_34_07/timestamp_000000"
    # # data = parse_timestamp(timestamp)
    # seq_path = "/code/dataset/airv2x/2025_04_09_22_34_07"
    # seq_dict = parse_seq(seq_path)
    # print(f"k {seq_dict[1514].keys()}")
    # print(f"seq {seq_dict[1514][0].keys()}")

    # test get visible region
    intrinsic = np.array(
        [[448.13282445, 0.0, 640.0], [0.0, 448.13282445, 360.0], [0.0, 0.0, 1.0]]
    )
    bevcam_rel = [2.5, 0.0, -2.0, 0.0, 0.0, -90.0]
    drone_ego = [-50.5, 33.2, 71.5, 0.0, -179.9, 0.0]
    mock_ego = [-50.5, 33.2, 0, 0.0, 90, 0.0]
    bevcam_world_pos = get_abs_world_pose(bevcam_rel, drone_ego)
    visible = get_bevcam_visible_region_from_intrinsic(
        bevcam_world_pos, mock_ego, intrinsic
    )
    print(visible)
