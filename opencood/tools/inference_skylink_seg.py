# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import sys
import time
import re

root_path = os.path.abspath(__file__)
root_path = "/".join(root_path.split("/")[:-3])
sys.path.append(root_path)

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset

# from opencood.tools import train_utils as train_utils
from opencood.tools import inference_utils as inference_utils
from opencood.tools import train_utils
from opencood.visualization import simple_vis
from collections import defaultdict


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Continued training path"
    )
    parser.add_argument(
        "--fusion_method",
        type=str,
        default="intermediate",
        help="no, no_w_uncertainty, late, early or intermediate",
    )
    parser.add_argument(
        "--save_vis",
        type=bool,
        default=False,
        help="save how many numbers of visualization result?",
    )
    parser.add_argument(
        "--save_vis_n",
        type=int,
        default=10,
        help="save how many numbers of visualization result?",
    )
    parser.add_argument(
        "--save_npy",
        action="store_true",
        help="whether to save prediction and gt resultin npy file",
    )
    parser.add_argument("--eval_epoch", type=int, default=20, help="Set the checkpoint")
    parser.add_argument(
        "--eval_best_epoch", type=bool, default=False, help="Set the checkpoint"
    )
    parser.add_argument(
        "--comm_thre",
        type=float,
        default=None,
        help="Communication confidence threshold",
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default="results.txt",
        help="The path to save the result",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        help="The path to the config file",
    )
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in [
        "late",
        "early",
        "intermediate",
        "intermediate_with_comm",
        "no",
    ]

    hypes = yaml_utils.load_yaml(None, opt)

    if opt.comm_thre is not None:
        hypes["model"]["args"]["fusion_args"]["communication"]["thre"] = opt.comm_thre

    if ("opv2v" in opt.model_dir) or ("V2X" in opt.model_dir):
        from opencood.utils import eval_utils_opv2v as eval_utils

        left_hand = True

    elif "dair" in opt.model_dir:
        from opencood.utils import eval_utils_where2comm as eval_utils

        hypes["validate_dir"] = hypes["test_dir"]
        left_hand = False
    
    elif "skylink" in opt.model_dir:
        from opencood.utils import eval_utils_skylink as eval_utils
        left_hand = True

    else:
        print(f"The path should contain one of the following strings [opv2v|dair] .")
        return

    print(f"Left hand visualizing: {left_hand}")

    print("Dataset Building")
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")

    data_loader = DataLoader(
        opencood_dataset,
        batch_size=1,
        num_workers=16,
        collate_fn=opencood_dataset.collate_batch_test,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
    )

    print("Creating Model")
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Model from checkpoint")
    saved_path = opt.model_dir
    epoch_id, model = train_utils.load_model(
        saved_path, model, opt.eval_epoch, start_from_best=opt.eval_best_epoch
    )

    model.zero_grad()
    model.eval()

    total_comm_rates = []
    
    result_stat_init = lambda: {
        "gt_dynamic_seg_map_list": [],
        "pred_dynamic_seg_map_list": [],
        "gt_static_seg_map_list": [],
        "pred_static_seg_map_list": [],
    }
    
    result_stat_dict = defaultdict(result_stat_init)
    
    # total_box = []
    for i, batch_data in tqdm(enumerate(data_loader)):
        m = re.search(r'(\d{4}(?:_\d{2}){5})', batch_data['ego']['metadata_path_list'][0])
        if m:
            timestamp = m.group(1)  # '2025_05_06_08_16_50'
        else:
            raise ValueError("Timestamp not found in the path")
        result_stat = result_stat_dict[timestamp]
        if i == 1000:
            break
        with torch.no_grad():
            # _batch_data = batch_data[0]
            batch_data = train_utils.to_device(batch_data, device)
            # print(_batch_data.keys())
            # _batch_data = train_utils.to_device(_batch_data, device)
            # if 'scope' in hypes['name'] or 'how2comm' in hypes['name']:
            #     batch_data= _batch_data
            if opt.fusion_method == "late":
                pred_box_tensor, pred_score, gt_box_tensor, output_dict = (
                    inference_utils.inference_late_fusion(
                        batch_data, model, opencood_dataset
                    )
                )
                comm = 0
                for key in output_dict:
                    comm += output_dict[key]["comm_rates"]
                total_comm_rates.append(comm)
            elif opt.fusion_method == "early":
                pred_dynamic_seg_map, pred_static_seg_map, gt_dynamic_seg_map, gt_static_seg_map = (
                    inference_utils.inference_early_fusion_skylink_segmentation(
                        batch_data, model, opencood_dataset
                    )
                )
            elif opt.fusion_method == "intermediate":
                    pred_dynamic_seg_map, pred_static_seg_map, gt_dynamic_seg_map, gt_static_seg_map = (
                        inference_utils.inference_intermediate_fusion_skylink_segmentation(
                        batch_data, model, opencood_dataset
                    )
                )
            # TODO(YH): not adapt to the new version yet
            # elif opt.fusion_method == "no":
            #     pred_box_tensor, pred_score, gt_box_tensor = (
            #         inference_utils.inference_no_fusion(
            #             batch_data, model, opencood_dataset
            #         )
            #     )

            # elif opt.fusion_method == "intermediate_with_comm":
            #     (
            #         pred_box_tensor,
            #         pred_score,
            #         gt_box_tensor,
            #         comm_rates,
            #         mask,
            #         each_mask,
            #     ) = inference_utils.inference_intermediate_fusion_withcomm(
            #         batch_data, model, opencood_dataset
            #     )
            #     total_comm_rates.append(comm_rates)
            else:
                raise NotImplementedError(
                    "Only early, late and intermediate, no, intermediate_with_comm fusion modes are supported."
                )
            
            result_stat["gt_dynamic_seg_map_list"].append(gt_dynamic_seg_map.cpu())
            result_stat["pred_dynamic_seg_map_list"].append(pred_dynamic_seg_map.cpu())
            result_stat["gt_static_seg_map_list"].append(gt_static_seg_map.cpu())
            result_stat["pred_static_seg_map_list"].append(pred_static_seg_map.cpu())
                
            
            

            # eval_utils.eval_multiclass_results(pred_box_tensor, pred_score, pred_labels, gt_box_tensor, gt_class_label_list, opt.model_dir)
            # eval_utils.eval_multiclass_results(pred_box_tensor, pred_score, pred_labels, gt_box_tensor, gt_class_label_list, opt.model_dir)
            # eval_utils.eval_multiclass_results(pred_box_tensor, pred_score, pred_labels, gt_box_tensor, gt_class_label_list, opt.model_dir)
            # result =  eval_utils.evaluate_segmentation(gt_dynamic_seg_map, pred_dynamic_seg_map, gt_static_seg_map, pred_static_seg_map)
            # import pdb; pdb.set_trace()
            # print(result)


            # if opt.save_npy:
            #     npy_save_path = os.path.join(opt.model_dir, "npy")
            #     if not os.path.exists(npy_save_path):
            #         os.makedirs(npy_save_path)
            #     inference_utils.save_prediction_gt(
            #         pred_box_tensor,
            #         gt_box_tensor,
            #         batch_data["ego"]["origin_lidar"][0],
            #         i,
            #         npy_save_path,
            #     )

            # if opt.save_vis_n and opt.save_vis_n >i:
            if opt.save_vis:
                
                vis_save_path = os.path.join(opt.model_dir, "vis_segmentation")
                if not os.path.exists(vis_save_path):
                    os.makedirs(vis_save_path)
                # vis_save_path = os.path.join(opt.model_dir, "vis_segmentation/seg_%05d.png" % i)
                simple_vis.visualize_segmentation(pred_dynamic_seg_map, 7, vis_save_path, "pred_dynamic", i)
                simple_vis.visualize_segmentation(pred_static_seg_map, 3, vis_save_path, "pred_static", i)
                simple_vis.visualize_segmentation(gt_dynamic_seg_map, 7, vis_save_path, "gt_dynamic", i)
                simple_vis.visualize_segmentation(gt_static_seg_map, 3, vis_save_path, "gt_static", i)
        
    combined_stat = inference_utils.combine_stat_by_scenarios_segmentation(result_stat_dict)
    results_dict = dict()
    for scenario in combined_stat:
        result_stat = combined_stat[scenario]
        result =  eval_utils.evaluate_segmentation(torch.stack(result_stat["gt_dynamic_seg_map_list"]), torch.stack(result_stat["pred_dynamic_seg_map_list"]), torch.stack(result_stat["gt_static_seg_map_list"]), torch.stack(result_stat["pred_static_seg_map_list"]))
        results_dict[scenario] = result
    with open(os.path.join(saved_path, opt.result_file.replace('txt', 'json')), 'w') as f:
        json.dump(results_dict, f, indent=4)
        
        
        
        

    # print('total_box: ', sum(total_box)/len(total_box))

    # if len(total_comm_rates) > 0:
    #     comm_rates = sum(total_comm_rates) / len(total_comm_rates)
    #     if not isinstance(comm_rates, float):
    #         comm_rates = comm_rates.item()
    # else:
    #     comm_rates = 0

    # import math

    # # comm_rates_base2=math.log(comm_rates,2)
    # with open(os.path.join(saved_path, "result.txt"), "a+") as f:
    #     msg = "Epoch: {} | AP @0.3: {:.04f} | AP @0.5: {:.04f} | AP @0.7: {:.04f} | comm_rate: {:.06f} \n".format(
    #         epoch_id, ap_30, ap_50, ap_70, comm_rates
    #     )
    #     if opt.comm_thre is not None:
    #         msg = "Epoch: {} | AP @0.3: {:.04f} | AP @0.5: {:.04f} | AP @0.7: {:.04f} | comm_rate: {:.06f} | comm_thre: {:.04f}\n".format(
    #             epoch_id, ap_30, ap_50, ap_70, comm_rates, opt.comm_thre
    #         )
    #     f.write(msg)
    #     print(msg)


if __name__ == "__main__":
    main()
