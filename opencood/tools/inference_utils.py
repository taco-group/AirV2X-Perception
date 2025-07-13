# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# Modifier: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
from collections import OrderedDict

import numpy as np
import torch

from opencood.tools import train_utils
from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.utils.scenario_utils import scenarios_params
from opencood.utils import box_utils
from collections import defaultdict
import pickle


def inference_late_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(
        batch_data, output_dict
    )

    return pred_box_tensor, pred_score, gt_box_tensor, output_dict


def inference_no_fusion(batch_data, model, dataset):
    """
    Model inference for no fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict_ego = OrderedDict()

    output_dict_ego["ego"] = model(batch_data["ego"])
    # output_dict only contains ego
    # but batch_data havs all cavs, because we need the gt box inside.

    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process_no_fusion(
        batch_data,  # only for late fusion dataset
        output_dict_ego,
    )

    return pred_box_tensor, pred_score, gt_box_tensor


def inference_early_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data["ego"]
    # print('type(cav_content) is ',type(cav_content))
    output_dict["ego"] = model(cav_content)

    try:
        pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(
            batch_data, output_dict
        )
    except:
        pred_box_tensor, pred_score, pred_labels, pred_boxes3d, gt_box_tensor, gt_class_label_list, gt_track_list = dataset.post_process(
            batch_data, output_dict
        )
        
    return pred_box_tensor, pred_score, gt_box_tensor, pred_boxes3d


def inference_intermediate_fusion_withcomm(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data["ego"]
    output_dict["ego"] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(
        batch_data, output_dict
    )
    comm_rates = output_dict["ego"]["comm_rate"]
    mask = output_dict["ego"]["mask"]
    each_mask = output_dict["ego"]["each_mask"]
    return pred_box_tensor, pred_score, gt_box_tensor, comm_rates, mask, each_mask
    # return pred_box_tensor, pred_score, gt_box_tensor, comm_rates, mask


def inference_intermediate_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """

    return inference_early_fusion(batch_data, model, dataset)


def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to txt file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, "%04d_pcd.npy" % timestamp), pcd_np)
    np.save(os.path.join(save_path, "%04d_pred.npy" % timestamp), pred_np)
    np.save(os.path.join(save_path, "%04d_gt.npy" % timestamp), gt_np)



# ====================================================================================================
# airv2x: (1) multiclass (2) segmentation (3) tracking
# ====================================================================================================

def inference_intermediate_fusion_airv2x(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """

    return inference_early_fusion_airv2x(batch_data, model, dataset)


def inference_early_fusion_airv2x(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data["ego"]
    # print('type(cav_content) is ',type(cav_content))
    output_dict["ego"] = model(cav_content)

    pred_box_tensor, pred_score, pred_labels, gt_box_tensor, gt_class_label_list, gt_track_list = dataset.post_process(
        batch_data, output_dict
    )
    return pred_box_tensor, pred_score, pred_labels, gt_box_tensor, gt_class_label_list, gt_track_list



# segmentation
def inference_intermediate_fusion_airv2x_segmentation(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """

    return inference_early_fusion_airv2x_segmentation(batch_data, model, dataset)


def inference_early_fusion_airv2x_segmentation(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data["ego"]
    # print('type(cav_content) is ',type(cav_content))
    output_dict["ego"] = model(cav_content)

    pred_dynamic_seg_map, pred_static_seg_map, gt_dynamic_seg_map, gt_static_seg_map = dataset.post_process_seg(
        batch_data, output_dict
    )
    return pred_dynamic_seg_map, pred_static_seg_map, gt_dynamic_seg_map, gt_static_seg_map


def save_preds_airv2x(pred_box_tensor, pred_score, pred_boxes3d, batch_data, save_path, pred_labels=0):
    # if type(pred_box_tensor) == torch.Tensor:
    #     pred_box_tensor = torch_tensor_to_numpy(pred_box_tensor)
    if type(pred_score) == torch.Tensor:
        pred_score = torch_tensor_to_numpy(pred_score)
    if type(pred_labels) == torch.Tensor:
        pred_labels = torch_tensor_to_numpy(pred_labels)
    if type(pred_boxes3d) == torch.Tensor:
        pred_boxes3d = torch_tensor_to_numpy(pred_boxes3d)
    metadata_path = batch_data['ego']["metadata_path_list"][0]
    ego_lidar_pose = batch_data['ego']["ego_lidar_pose_list"][0]
    
    meta_data_root = os.path.join(*metadata_path.split('/')[-5:-1])
    save_dir = os.path.join(save_path, "preds", meta_data_root)
    os.makedirs(save_dir, exist_ok=True)
    
    pred = box_utils.convert_boxes_to_format(pred_boxes3d)
    location = pred[:, :6]
    extent = pred[:, 6:]
    metadata_dict = dict()
    for idx in range(len(pred)):
        metadata = {
            'location': location[idx].tolist(), 
            'extent': extent[idx].tolist(),
            'class': pred_labels[idx] if type(pred_labels) == np.ndarray else pred_labels,
            'confidence': pred_score[idx]
        }
        metadata_dict[idx] = metadata
    metadata_dict['ego_lidar_pose'] = ego_lidar_pose
    with open(os.path.join(save_dir, 'predictions.pkl'), 'wb') as f:
        pickle.dump(metadata_dict, f)
    
    
    
def combine_stat(combined_stat, stat):
    """
    Combine statistics from different scenarios.

    Parameters
    ----------
    combined_stat : dict
        Combined statistics.
    stat : dict
        Statistics to be combined.

    Returns
    -------
    dict
        Combined statistics.
    """
    for key, value in stat.items():
        if key in combined_stat:
            combined_stat[key]["tp"].extend(value["tp"])
            combined_stat[key]["fp"].extend(value["fp"])
            combined_stat[key]["gt"] += value["gt"]
            combined_stat[key]["score"].extend(value["score"])
        else:
            combined_stat[key] = value
    return combined_stat


def combine_stat_by_scenarios(result_stat_dict):
    result_stat_init = lambda: {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }
    combined_stat = defaultdict(result_stat_init)

    for scenario, stat in result_stat_dict.items():
        if scenario not in scenarios_params:
            print(f"Warning: scenario {scenario} not in scenarios_params, skipping...")
            continue
        for key, value in scenarios_params[scenario].items():
            if value:
                combined_stat[key] = combine_stat(combined_stat[key], stat)   
        combined_stat['all'] = combine_stat(combined_stat['all'], stat)           
    return combined_stat


def combine_stat_by_scenarios_segmentation(result_stat_dict):
    
    result_stat_init = lambda: {
        "gt_dynamic_seg_map_list": [],
        "pred_dynamic_seg_map_list": [],
        "gt_static_seg_map_list": [],
        "pred_static_seg_map_list": [],
    }
    combined_stat = defaultdict(result_stat_init)
    
    for scenario, stat in result_stat_dict.items():
        if scenario not in scenarios_params:
            print(f"Warning: scenario {scenario} not in scenarios_params, skipping...")
            continue    
        for key, value in scenarios_params[scenario].items():
            if value:
                combined_stat[key]["gt_dynamic_seg_map_list"].extend(stat["gt_dynamic_seg_map_list"]) 
                combined_stat[key]["pred_dynamic_seg_map_list"].extend(stat["pred_dynamic_seg_map_list"])
                combined_stat[key]["gt_static_seg_map_list"].extend(stat["gt_static_seg_map_list"])
                combined_stat[key]["pred_static_seg_map_list"].extend(stat["pred_static_seg_map_list"]) 
                 
        combined_stat['all']["gt_dynamic_seg_map_list"].extend(stat["gt_dynamic_seg_map_list"])
        combined_stat['all']["pred_dynamic_seg_map_list"].extend(stat["pred_dynamic_seg_map_list"])
        combined_stat['all']["gt_static_seg_map_list"].extend(stat["gt_static_seg_map_list"])
        combined_stat['all']["pred_static_seg_map_list"].extend(stat["pred_static_seg_map_list"])
        
    return combined_stat