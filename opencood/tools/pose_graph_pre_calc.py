# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import copy
import importlib
import json
import os
import statistics
from collections import OrderedDict

import numpy as np
import torch
from icecream import ic
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Subset

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument(
        "--hypes_yaml",
        "-y",
        type=str,
        default="/GPFS/rhome/yifanlu/OpenCOOD/opencood/hypes_yaml/dair-v2x/uncertainty/pose_graph_pre_calc_dair.yaml",
        help="data generation yaml file needed ",
    )
    parser.add_argument("--model_dir", type=str, default="")
    opt = parser.parse_args()
    return opt


SAVE_BOXES = True


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    pos_std_list = [0]  # , 0.2, 0.4, 0.6, 0.8, 1.0]
    rot_std_list = [0]  # , 0.2, 0.4, 0.6, 0.8, 1.0]
    pos_mean_list = [0]  # , 0, 0, 0, 0]
    rot_mean_list = [0]  # , 0, 0, 0, 0]

    for pos_mean, pos_std, rot_mean, rot_std in zip(
        pos_mean_list, pos_std_list, rot_mean_list, rot_std_list
    ):
        # setting noise
        noise_setting = OrderedDict()
        noise_args = {
            "pos_std": pos_std,
            "rot_std": rot_std,
            "pos_mean": pos_mean,
            "rot_mean": rot_mean,
        }

        noise_setting["add_noise"] = True
        noise_setting["args"] = noise_args

        # build dataset for each noise setting
        print(f"Noise Added: {pos_std}/{rot_std}/{pos_mean}/{rot_mean}.")
        hypes.update({"noise_setting": noise_setting})

        print("Dataset Building")
        opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
        # opencood_train_subset = Subset(opencood_train_dataset, [5578, 5572, 5103, 3048, 2445])
        opencood_train_subset = opencood_train_dataset
        # opencood_train_subset = Subset(opencood_train_dataset, [5876, 3328, 3338, 4956, 4957, 5079])
        # ind = np.random.permutation(6000)[:100].tolist()
        # opencood_train_subset = Subset(opencood_train_dataset, ind)
        opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

        hypes_ = copy.deepcopy(hypes)
        hypes_["validate_dir"] = hypes_["test_dir"]
        opencood_test_dataset = build_dataset(hypes_, visualize=False, train=False)

        train_loader = DataLoader(
            opencood_train_subset,
            batch_size=1,
            num_workers=4,
            collate_fn=opencood_train_dataset.collate_batch_test,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )
        val_loader = DataLoader(
            opencood_validate_dataset,
            batch_size=1,
            num_workers=4,
            collate_fn=opencood_train_dataset.collate_batch_test,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )

        test_loader = DataLoader(
            opencood_test_dataset,
            batch_size=1,
            num_workers=4,
            collate_fn=opencood_train_dataset.collate_batch_test,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        proj_first = hypes["fusion"]["args"]["proj_first"]
        assert proj_first is False
        # used to help schedule learning rate

        ########################################################################
        ########################################################################

        # from opencood.models.sub_modules.box_align import vis_pose_graph, box_alignment_relative_sample
        # from opencood.utils.transformation_utils import get_pairwise_transformation_torch
        hypes = yaml_utils.load_point_pillar_params_stage1(hypes)

        stage1_model_name = hypes["box_align_pre_calc"][
            "stage1_model"
        ]  # point_pillar_disconet_teacher
        stage1_model_config = hypes["box_align_pre_calc"]["stage1_model_config"]
        stage1_checkpoint_path = hypes["box_align_pre_calc"]["stage1_model_path"]

        # import the model
        model_filename = "opencood.models." + stage1_model_name
        model_lib = importlib.import_module(model_filename)
        stage1_model_class = None
        target_model_name = stage1_model_name.replace("_", "")

        for name, cls in model_lib.__dict__.items():
            if name.lower() == target_model_name.lower():
                stage1_model_class = cls

        stage1_model = stage1_model_class(stage1_model_config)
        stage1_model.load_state_dict(torch.load(stage1_checkpoint_path), strict=False)

        # import the postprocessor
        stage1_postprocessor_name = hypes["box_align_pre_calc"][
            "stage1_postprocessor_name"
        ]
        stage1_postprocessor_config = hypes["box_align_pre_calc"][
            "stage1_postprocessor_config"
        ]
        postprocessor_lib = importlib.import_module(
            "opencood.data_utils.post_processor"
        )
        stage1_postprocessor_class = None
        target_postprocessor_name = stage1_postprocessor_name.replace("_", "")

        for name, cls in postprocessor_lib.__dict__.items():
            if name.lower() == target_postprocessor_name:
                stage1_postprocessor_class = cls

        stage1_postprocessor = stage1_postprocessor_class(
            stage1_postprocessor_config, train=False
        )

        for p in stage1_model.parameters():
            p.requires_grad_(False)

        if torch.cuda.is_available():
            stage1_model.to(device)

        stage1_model.eval()
        stage1_anchor_box = torch.from_numpy(stage1_postprocessor.generate_anchor_box())

        for split in ["train", "val", "test"]:
            stage1_boxes_dict = dict()

            stage1_boxes_save_path = f"/GPFS/rhome/yifanlu/OpenCOOD/stage1_boxes/dair_new/{split}/stage1_boxes.json"
            for i, batch_data in enumerate(eval(f"{split}_loader")):
                if batch_data is None:
                    continue

                batch_data = train_utils.to_device(batch_data, device)
                print(
                    i, batch_data["ego"]["sample_idx"], batch_data["ego"]["cav_id_list"]
                )
                output_stage1 = stage1_model(batch_data["ego"])
                pred_corner3d_list, pred_box3d_list, uncertainty_list = (
                    stage1_postprocessor.post_process_stage1(
                        output_stage1, stage1_anchor_box
                    )
                )
                record_len = batch_data["ego"]["record_len"]
                lidar_pose = batch_data["ego"]["lidar_pose"]
                lidar_pose_clean = batch_data["ego"]["lidar_pose_clean"]

                if pred_corner3d_list is None:
                    continue

                """
                    Save the corners, uncertainty, lidar_pose_clean
                """

                if SAVE_BOXES:
                    if pred_corner3d_list is None:
                        stage1_boxes_dict[batch_data["ego"]["sample_idx"]] = None
                        continue
                    sample_idx = batch_data["ego"]["sample_idx"]
                    pred_corner3d_np_list = [
                        x.cpu().numpy().tolist() for x in pred_corner3d_list
                    ]
                    uncertainty_np_list = [
                        x.cpu().numpy().tolist() for x in uncertainty_list
                    ]
                    lidar_pose_clean_np = lidar_pose_clean.cpu().numpy().tolist()
                    stage1_boxes_dict[sample_idx] = OrderedDict()

                    stage1_boxes_dict[sample_idx]["pred_corner3d_np_list"] = (
                        pred_corner3d_np_list
                    )
                    stage1_boxes_dict[sample_idx]["uncertainty_np_list"] = (
                        uncertainty_np_list
                    )
                    stage1_boxes_dict[sample_idx]["lidar_pose_clean_np"] = (
                        lidar_pose_clean_np
                    )
                    stage1_boxes_dict[sample_idx]["cav_id_list"] = batch_data["ego"][
                        "cav_id_list"
                    ]

            if SAVE_BOXES:
                with open(stage1_boxes_save_path, "w") as f:
                    json.dump(stage1_boxes_dict, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
