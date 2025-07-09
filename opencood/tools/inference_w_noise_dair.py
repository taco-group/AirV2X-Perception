"""
Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
"""

import argparse
import os
import time
from typing import OrderedDict

import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import inference_utils, train_utils
from opencood.utils import eval_utils_where2comm as eval_utils
from opencood.visualization import simple_vis

torch.multiprocessing.set_sharing_strategy("file_system")


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Continued training path"
    )
    parser.add_argument(
        "--also_laplace",
        action="store_true",
        help="whether to use laplace to simulate noise. Otherwise Gaussian",
    )
    parser.add_argument(
        "--fusion_method",
        type=str,
        default="intermediate",
        help="no, no_w_uncertainty, late, early or intermediate",
    )
    parser.add_argument(
        "--save_vis", type=bool, default=False, help="save visualization result"
    )
    parser.add_argument(
        "--save_npy",
        type=bool,
        default=False,
        help="whether to save prediction and gt result in npy file",
    )
    parser.add_argument(
        "--eval_epoch", type=str, default=None, help="Set the checkpoint"
    )
    parser.add_argument(
        "--comm_thre",
        type=float,
        default=None,
        help="Communication confidence threshold",
    )
    parser.add_argument("--note", default="", type=str, help="any other thing?")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in [
        "late",
        "early",
        "intermediate",
        "no",
        "no_w_uncertainty",
        "single",
    ]

    hypes = yaml_utils.load_yaml(None, opt)

    if opt.comm_thre is not None:
        hypes["model"]["args"]["fusion_args"]["communication"]["thre"] = opt.comm_thre
    hypes["validate_dir"] = hypes["test_dir"]
    if "OPV2V" in hypes["test_dir"] or "v2xsim" in hypes["test_dir"]:
        assert "test" in hypes["validate_dir"]
    left_hand = (
        True
        if ("OPV2V" in hypes["test_dir"] or "V2XSET" in hypes["test_dir"])
        else False
    )

    if "dair_v2x" in hypes["test_dir"]:
        opt.also_laplace = True

    print(f"Left hand visualizing: {left_hand}")

    if "box_align" in hypes.keys():
        hypes["box_align"]["val_result"] = hypes["box_align"]["test_result"]

    print("Creating Model")
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Model from checkpoint")
    saved_path = opt.model_dir
    if opt.eval_epoch is not None:
        epoch_id = opt.eval_epoch
        epoch_id, model = train_utils.load_saved_model(saved_path, model, epoch_id)
    else:
        epoch_id, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # add noise to pose.
    pos_std_list = [0, 0.2, 0.4, 0.6]
    rot_std_list = [0, 0.2, 0.4, 0.6]
    pos_mean_list = [0, 0, 0, 0]
    rot_mean_list = [0, 0, 0, 0]

    pos_std_list = [0.6]
    rot_std_list = [0.6]
    pos_mean_list = [0]
    rot_mean_list = [0]

    if opt.also_laplace:
        use_laplace_options = [False, True]
    else:
        use_laplace_options = [False]

    for use_laplace in use_laplace_options:
        AP30 = []
        AP50 = []
        AP70 = []
        for pos_mean, pos_std, rot_mean, rot_std in zip(
            pos_mean_list, pos_std_list, rot_mean_list, rot_std_list
        ):
            # setting noise
            np.random.seed(303)
            noise_setting = OrderedDict()
            noise_args = {
                "pos_std": pos_std,
                "rot_std": rot_std,
                "pos_mean": pos_mean,
                "rot_mean": rot_mean,
            }

            noise_setting["add_noise"] = True
            noise_setting["args"] = noise_args

            suffix = ""
            if use_laplace:
                noise_setting["args"]["laplace"] = True
                suffix = "_laplace"

            # build dataset for each noise setting
            print("Dataset Building")
            print(f"Noise Added: {pos_std}/{rot_std}/{pos_mean}/{rot_mean}.")
            hypes.update({"noise_setting": noise_setting})
            opencood_dataset = build_dataset(hypes, visualize=True, train=False)
            data_loader = DataLoader(
                opencood_dataset,
                batch_size=1,
                num_workers=8,
                collate_fn=opencood_dataset.collate_batch_test,
                shuffle=False,
                pin_memory=False,
                drop_last=False,
            )

            # Create the dictionary for evaluation
            result_stat = {
                0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
                0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
                0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
            }

            noise_level = (
                f"{pos_std}_{rot_std}_{pos_mean}_{rot_mean}_"
                + opt.fusion_method
                + suffix
                + opt.note
            )

            for i, batch_data in enumerate(data_loader):
                print(f"{noise_level}_{i}")
                if batch_data is None:
                    continue
                with torch.no_grad():
                    batch_data = train_utils.to_device(batch_data, device)

                    if opt.fusion_method == "late":
                        pred_box_tensor, pred_score, gt_box_tensor = (
                            inference_utils.inference_late_fusion(
                                batch_data, model, opencood_dataset
                            )
                        )
                    elif opt.fusion_method == "early":
                        pred_box_tensor, pred_score, gt_box_tensor = (
                            inference_utils.inference_early_fusion(
                                batch_data, model, opencood_dataset
                            )
                        )
                    elif opt.fusion_method == "intermediate":
                        pred_box_tensor, pred_score, gt_box_tensor = (
                            inference_utils.inference_intermediate_fusion(
                                batch_data, model, opencood_dataset
                            )
                        )
                    elif opt.fusion_method == "no":
                        pred_box_tensor, pred_score, gt_box_tensor = (
                            inference_utils.inference_no_fusion(
                                batch_data, model, opencood_dataset
                            )
                        )

                    elif opt.fusion_method == "intermediate_with_comm":
                        pred_box_tensor, pred_score, gt_box_tensor, comm_rates = (
                            inference_utils.inference_intermediate_fusion_withcomm(
                                batch_data, model, opencood_dataset
                            )
                        )
                        total_comm_rates.append(comm_rates)
                    else:
                        raise NotImplementedError(
                            "Only single, no, no_w_uncertainty, early, late and intermediate"
                            "fusion is supported."
                        )

                    # pred_box_tensor = infer_result['pred_box_tensor']
                    # gt_box_tensor = infer_result['gt_box_tensor']
                    # pred_score = infer_result['pred_score']

                    eval_utils.caluclate_tp_fp(
                        pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.3
                    )
                    eval_utils.caluclate_tp_fp(
                        pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.5
                    )
                    eval_utils.caluclate_tp_fp(
                        pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.7
                    )

                    if opt.save_npy:
                        npy_save_path = os.path.join(opt.model_dir, "npy")
                        if not os.path.exists(npy_save_path):
                            os.makedirs(npy_save_path)
                        inference_utils.save_prediction_gt(
                            pred_box_tensor,
                            gt_box_tensor,
                            batch_data["ego"]["origin_lidar"][0],
                            i,
                            npy_save_path,
                        )

                    if opt.save_vis:
                        vis_save_path = os.path.join(opt.model_dir, "vis_3d")
                        if not os.path.exists(vis_save_path):
                            os.makedirs(vis_save_path)
                        vis_save_path = os.path.join(
                            opt.model_dir, "vis_3d/3d_%05d.png" % i
                        )
                        simple_vis.visualize(
                            pred_box_tensor,
                            gt_box_tensor,
                            batch_data["ego"]["origin_lidar"][0],
                            hypes["postprocess"]["gt_range"],
                            vis_save_path,
                            method="3d",
                            left_hand=left_hand,
                            vis_pred_box=True,
                        )

                        vis_save_path = os.path.join(opt.model_dir, "vis_bev")
                        if not os.path.exists(vis_save_path):
                            os.makedirs(vis_save_path)
                        vis_save_path = os.path.join(
                            opt.model_dir, "vis_bev/bev_%05d.png" % i
                        )
                        simple_vis.visualize(
                            pred_box_tensor,
                            gt_box_tensor,
                            batch_data["ego"]["origin_lidar"][0],
                            hypes["postprocess"]["gt_range"],
                            vis_save_path,
                            method="bev",
                            left_hand=left_hand,
                            vis_pred_box=True,
                        )

                torch.cuda.empty_cache()

            ap30, ap50, ap70 = eval_utils.eval_final_results(
                result_stat, opt.model_dir, noise_level
            )
            AP30.append(ap30)
            AP50.append(ap50)
            AP70.append(ap70)

            dump_dict = {"ap30": AP30, "ap50": AP50, "ap70": AP70}
            yaml_utils.save_yaml(
                dump_dict,
                os.path.join(opt.model_dir, f"AP030507_{opt.note}{suffix}.yaml"),
            )


if __name__ == "__main__":
    main()
