# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
import sys

root_path = os.path.abspath(__file__)
root_path = "/".join(root_path.split("/")[:-3])
sys.path.append(root_path)

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from opencood.data_utils import datasets
from opencood.data_utils.datasets.dair.lidar_camera_intermediate_fusion_dataset_v2 import (
    LiDARCameraIntermediateFusionDatasetDAIR,
)
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import inference_utils, train_utils
from opencood.visualization import simple_vis, vis_utils

if __name__ == "__main__":
    current_path = os.path.dirname(os.path.realpath(__file__))
    params = load_yaml(
        os.path.join(current_path, "../hypes_yaml/where2comm/dair-v2x/dair_m2fuse.yaml")
    )
    output_path = "/home/test_vis_result/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    opencda_dataset = LiDARCameraIntermediateFusionDatasetDAIR(
        params, visualize=True, train=False
    )
    dataset_len = len(opencda_dataset)
    subset_idx = range(10, dataset_len)
    subset_dataset = Subset(opencda_dataset, subset_idx)

    data_loader = DataLoader(
        opencda_dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=opencda_dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
    )
    vis_gt_box = False  # True
    vis_pred_box = False
    hypes = params

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cpu")

    for i, batch_data in enumerate(data_loader):
        print(i)
        batch_data = train_utils.to_device(batch_data, device)
        gt_box_tensor = opencda_dataset.post_processor.generate_gt_bbx(batch_data)

        """
        vis_save_path = os.path.join(output_path, '3d_%05d.png' % i)
        simple_vis.visualize(None,
                            gt_box_tensor,
                            batch_data['ego']['origin_lidar'][0],
                            hypes['postprocess']['gt_range'],
                            vis_save_path,
                            method='3d',
                            vis_gt_box = vis_gt_box,
                            vis_pred_box = vis_pred_box,
                            left_hand=False)
        
        """
        vis_save_path = os.path.join(output_path, "bev_%05d.png" % i)
        simple_vis.visualize(
            None,
            gt_box_tensor,
            batch_data["ego"]["origin_lidar"][0],
            hypes["postprocess"]["gt_range"],
            vis_save_path,
            method="bev",
            vis_gt_box=True,  # vis_gt_box,
            vis_pred_box=vis_pred_box,
            left_hand=False,
        )
        """

        image = batch_data['ego']['image_inputs']['ori_imgs'][0][0].cpu().numpy()
        depth_map = batch_data['ego']['image_inputs']['depth_map'][0][0].cpu().numpy()
        # 3, 360, 480 / 1, 360, 480
        #vis_save_path = os.path.join(output_path, 'camera_%05d.png' % i)
        #pil_image = Image.fromarray(image.astype(np.uint8))
        #pil_image.save(vis_save_path)

        vis_save_path = os.path.join(output_path, 'depth_%05d.png' % i)
        _max, _min = np.max(depth_map), np.min(depth_map)
        depth_map = 1 - (depth_map - _min) / (_max - _min)
        depth_mask = (depth_map >= 1).astype(np.float32)
        depth_map = depth_map[0]
        H, W = depth_map.shape
        draw_depth = np.ones((3,H,W))
        draw_depth[0] = draw_depth[0] * (depth_map*(1-depth_mask)*255+depth_mask*255)
        draw_depth[1] = draw_depth[1] * (depth_map*(1-depth_mask)*140+depth_mask*255)
        draw_depth[2] = draw_depth[2] * (depth_map*(1-depth_mask)*0+depth_mask*255)
        draw_depth = draw_depth.transpose(1,2,0)
        pil_depth = Image.fromarray(draw_depth.astype(np.uint8))
        pil_depth.save(vis_save_path)
        """
