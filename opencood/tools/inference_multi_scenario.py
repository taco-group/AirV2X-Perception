"""
Multi-scenario inference script for OpenCOOD.

This script performs inference on multiple scenarios using different fusion methods
and evaluates the model performance.

Author: Runsheng Xu <rxx3386@ucla.edu>
Modifier: Xiangbo Gao <xiangbogaobarry@gmail.com>
License: TDG-Attribution-NonCommercial-NoDistrib
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch, resource
torch.multiprocessing.set_sharing_strategy('file_system')
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to python path
root_path = Path(__file__).resolve().parents[2]
sys.path.append(str(root_path))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import inference_utils, train_utils
from opencood.visualization import simple_vis

# Constants
SUPPORTED_FUSION_METHODS = [
    "late",
    "early",
    "intermediate",
    "intermediate_with_comm",
    "no",
]

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multi-scenario inference")
    parser.add_argument(
        "--model_dir", 
        type=str, 
        required=True, 
        help="Path to model directory"
    )
    parser.add_argument(
        "--fusion_method",
        type=str,
        default="intermediate",
        choices=SUPPORTED_FUSION_METHODS,
        help="Fusion method to use"
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Whether to save visualization results"
    )
    parser.add_argument(
        "--save_vis_n",
        type=int,
        default=10,
        help="Number of visualization results to save"
    )
    parser.add_argument(
        "--save_npy",
        action="store_true",
        help="Whether to save predictions and ground truth in npy format"
    )
    parser.add_argument(
        "--save_pred",
        action="store_true",
        help="Whether to save predictions in pkl format"
    )
    parser.add_argument(
        "--eval_epoch",
        type=int,
        default=20,
        help="Epoch to evaluate"
    )
    parser.add_argument(
        "--eval_best_epoch",
        action="store_true",
        help="Whether to evaluate best epoch"
    )
    parser.add_argument(
        "--comm_thre",
        type=float,
        default=None,
        help="Communication confidence threshold"
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default="results.txt",
        help="Path to save results"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    return parser.parse_args()

def setup_model(hypes: dict, device: torch.device) -> torch.nn.Module:
    """Create and setup the model.
    
    Args:
        hypes: Configuration dictionary
        device: Device to run model on

    Returns:
        Initialized model
    """
    print("Creating Model")
    model = train_utils.create_model(hypes)
    model.to(device)
    return model

def load_model_checkpoint(
    model: torch.nn.Module,
    model_dir: str,
    eval_epoch: int,
    eval_best: bool,
    device: torch.device
) -> Tuple[int, torch.nn.Module]:
    """Load model checkpoint.
    
    Args:
        model: Model to load weights into
        model_dir: Directory containing checkpoints
        eval_epoch: Epoch to evaluate
        eval_best: Whether to evaluate best epoch
        device: Device to load model on

    Returns:
        Tuple of (epoch_id, loaded_model)
    """
    print("Loading Model from checkpoint")
    epoch_id, model = train_utils.load_model(
        model_dir, 
        model,
        eval_epoch,
        start_from_best=eval_best
    )
    model.eval()
    return epoch_id, model

def initialize_result_stats() -> Dict:
    """Initialize dictionary for tracking evaluation metrics."""
    return {
        "tp": [],
        "fp": [],
        "gt": 0,
        "score": []
    }

def process_batch(
    batch_data: Dict,
    model: torch.nn.Module,
    dataset: object,
    fusion_method: str,
    device: torch.device
) -> Tuple:
    """Process a single batch of data.
    
    Args:
        batch_data: Batch of data to process
        model: Model to run inference with
        dataset: Dataset object
        fusion_method: Type of fusion to perform
        device: Device to run on

    Returns:
        Tuple of prediction tensors and metrics
    """
    batch_data = train_utils.to_device(batch_data, device)
    
    if fusion_method == "late":
        pred_box_tensor, pred_score, gt_box_tensor, output_dict = \
            inference_utils.inference_late_fusion(batch_data, model, dataset)
        comm_rate = sum(output_dict[k]["comm_rates"] for k in output_dict)
        return pred_box_tensor, pred_score, gt_box_tensor, comm_rate
        
    elif fusion_method == "early":
        pred_box_tensor, pred_score, gt_box_tensor = \
            inference_utils.inference_early_fusion(batch_data, model, dataset)
        return pred_box_tensor, pred_score, gt_box_tensor, 0
        
    elif fusion_method == "intermediate":
        pred_box_tensor, pred_score, gt_box_tensor, pred_boxes3d = \
            inference_utils.inference_intermediate_fusion(batch_data, model, dataset)
        return pred_box_tensor, pred_score, gt_box_tensor, 0, pred_boxes3d
        
    elif fusion_method == "intermediate_with_comm":
        results = inference_utils.inference_intermediate_fusion_withcomm(
            batch_data, model, dataset)
        return results
        
    elif fusion_method == "no":
        pred_box_tensor, pred_score, gt_box_tensor = \
            inference_utils.inference_no_fusion(batch_data, model, dataset)
        return pred_box_tensor, pred_score, gt_box_tensor, 0
        
    else:
        raise NotImplementedError(
            f"Fusion method {fusion_method} not supported. "
            f"Supported methods: {SUPPORTED_FUSION_METHODS}"
        )

def save_visualization(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    batch_data: Dict,
    hypes: Dict,
    save_path: str,
    idx: int,
    left_hand: bool
) -> None:
    """Save visualization results.
    
    Args:
        pred_boxes: Predicted bounding boxes
        gt_boxes: Ground truth bounding boxes
        batch_data: Batch data dictionary
        hypes: Configuration dictionary
        save_path: Base path to save visualizations
        idx: Sample index
        left_hand: Whether to use left-hand coordinate system
    """
    # 3D visualization
    vis_save_path_3d = os.path.join(save_path, "vis_3d")
    os.makedirs(vis_save_path_3d, exist_ok=True)
    vis_save_file_3d = os.path.join(vis_save_path_3d, f"3d_{idx:05d}.png")
    
    simple_vis.visualize(
        pred_boxes,
        gt_boxes,
        batch_data["ego"]["origin_lidar"][0],
        hypes["preprocess"]["cav_lidar_range"],
        vis_save_file_3d,
        method="3d",
        left_hand=left_hand,
        vis_pred_box=True,
        pcd_rsu=batch_data["ego"]["origin_lidar_rsu"][0],
        pcd_drone=batch_data["ego"]["origin_lidar_drone"][0],
        batch_data=batch_data
    )

    # BEV visualization  
    vis_save_path_bev = os.path.join(save_path, "vis_bev")
    os.makedirs(vis_save_path_bev, exist_ok=True)
    vis_save_file_bev = os.path.join(vis_save_path_bev, f"bev_{idx:05d}.png")
    
    simple_vis.visualize(
        pred_boxes,
        gt_boxes,
        batch_data["ego"]["origin_lidar"][0],
        hypes["preprocess"]["cav_lidar_range"],
        vis_save_file_bev,
        method="bev",
        left_hand=left_hand,
        vis_pred_box=True,
        pcd_rsu=batch_data["ego"]["origin_lidar_rsu"][0],
        pcd_drone=batch_data["ego"]["origin_lidar_drone"][0],
        batch_data=batch_data
    )

def main():
    """Main inference function."""
    opt = parse_arguments()
    
    # Load config and setup
    hypes = yaml_utils.load_yaml(None, opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Update communication threshold if specified
    if opt.comm_thre is not None:
        hypes["model"]["args"]["fusion_args"]["communication"]["thre"] = opt.comm_thre

    # Determine evaluation utilities based on dataset
    if any(x in opt.model_dir for x in ["opv2v", "V2XR", "airv2x"]):
        from opencood.utils import eval_utils_opv2v as eval_utils
        left_hand = True
    elif "dair" in opt.model_dir:
        from opencood.utils import eval_utils_where2comm as eval_utils
        hypes["validate_dir"] = hypes["test_dir"]
        left_hand = False
    else:
        print("Model directory must contain one of: [opv2v|dair]")
        return

    # Update test directory if specified
    if "test_dir" in hypes:
        hypes["validate_dir"] = hypes["test_dir"]
    else:
        raise ValueError("test_dir not found in config file")

    print(f"Left hand visualizing: {left_hand}")

    # Build dataset and dataloader
    print("Building Dataset")
    dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(dataset)} samples found")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False
    )

    # Setup model
    model = setup_model(hypes, device)
    epoch_id, model = load_model_checkpoint(
        model, opt.model_dir, opt.eval_epoch, opt.eval_best_epoch, device
    )

    # Initialize statistics tracking
    result_stat_init = lambda: {
        th: initialize_result_stats() for th in [0.3, 0.5, 0.7]
    }
    result_stat_dict = defaultdict(result_stat_init)
    total_comm_rates = []

    # Main inference loop
    for i, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract timestamp from metadata
        timestamp_match = re.search(
            r'(\d{4}(?:_\d{2}){5})', 
            batch_data['ego']['metadata_path_list'][0]
        )
        if not timestamp_match:
            raise ValueError("Timestamp not found in metadata path")
        timestamp = timestamp_match.group(1)
        
        result_stat = result_stat_dict[timestamp]

        with torch.no_grad():
            # Process batch based on fusion method
            outputs = process_batch(batch_data, model, dataset, opt.fusion_method, device)
            
            if opt.fusion_method == "intermediate_with_comm":
                (pred_box_tensor, pred_score, gt_box_tensor, 
                 comm_rates, mask, each_mask) = outputs
                total_comm_rates.append(comm_rates)
            elif opt.fusion_method == "late":
                pred_box_tensor, pred_score, gt_box_tensor, comm_rate = outputs
                total_comm_rates.append(comm_rate)
            elif opt.fusion_method == "intermediate":
                pred_box_tensor, pred_score, gt_box_tensor, _, pred_boxes3d = outputs
            else:
                pred_box_tensor, pred_score, gt_box_tensor, _ = outputs

            if pred_box_tensor is None:
                continue

            # Calculate metrics
            for threshold in [0.3, 0.5, 0.7]:
                eval_utils.caluclate_tp_fp(
                    pred_box_tensor, pred_score, gt_box_tensor, result_stat, threshold
                )

            # Save predictions if requested
            if opt.save_pred:
                inference_utils.save_preds_airv2x(
                    pred_box_tensor, pred_score, pred_boxes3d, batch_data, opt.model_dir
                )

            # Save predictions and ground truth if requested
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, "npy")
                os.makedirs(npy_save_path, exist_ok=True)
                inference_utils.save_prediction_gt(
                    pred_box_tensor,
                    gt_box_tensor,
                    batch_data["ego"]["origin_lidar"][0],
                    i,
                    npy_save_path
                )

            # Save visualizations if requested
            if opt.save_vis:
                save_visualization(
                    pred_box_tensor,
                    gt_box_tensor,
                    batch_data,
                    hypes,
                    opt.model_dir,
                    i,
                    left_hand
                )

    # Calculate final metrics
    combined_stat = inference_utils.combine_stat_by_scenarios(result_stat_dict)
    
    # Calculate average communication rate
    comm_rate = (sum(total_comm_rates) / len(total_comm_rates)) if total_comm_rates else 0
    if isinstance(comm_rate, torch.Tensor):
        comm_rate = comm_rate.item()

    # Save results
    result_path = os.path.join(opt.model_dir, opt.result_file)
    for scenario, result_stat in combined_stat.items():
        ap_30, ap_50, ap_70 = eval_utils.eval_final_results(
            result_stat, opt.model_dir
        )
        
        msg = (
            f"Epoch: {epoch_id} | scenario: {scenario} | "
            f"AP @0.3: {ap_30:.4f} | AP @0.5: {ap_50:.4f} | "
            f"AP @0.7: {ap_70:.4f} | comm_rate: {comm_rate:.6f}"
        )
        
        if opt.comm_thre is not None:
            msg += f" | comm_thre: {opt.comm_thre:.4f}"
        msg += "\n"
        
        with open(result_path, "a+") as f:
            f.write(msg)
            print(msg)

if __name__ == "__main__":
    main()
