"""
Seg head for bev understanding
"""

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


class BevSegHead(nn.Module):
    def __init__(self, target, H, W, input_dim, dynamic_class, static_class, seg_res, cav_range):
        super(BevSegHead, self).__init__()
        self.target = target
        self.H = H
        self.W = W
        self.seg_res = seg_res
        self.cav_range = cav_range

        if self.target == 'dynamic':
            self.dynamic_head = nn.Conv2d(input_dim,
                                          dynamic_class,
                                          kernel_size=1)
        if self.target == 'static':
            # segmentation head
            self.static_head = nn.Conv2d(input_dim,
                                         static_class,
                                         kernel_size=1,)
        else:
            self.dynamic_head = nn.Conv2d(input_dim,
                                          dynamic_class,
                                          kernel_size=1)
            self.static_head = nn.Conv2d(input_dim,
                                         static_class,
                                         kernel_size=1)
        self.cal_crop_stat()
            
    def cal_crop_stat(self):
        seg_range_H = self.H * self.seg_res
        seg_range_W = self.W * self.seg_res
        range_H = self.cav_range[4] - self.cav_range[1]
        range_W = self.cav_range[3] - self.cav_range[0]
        self.crop_factor_H = seg_range_H / range_H
        self.crop_factor_W = seg_range_W / range_W
        
        
    def crop_or_pad_feature(self, feature_map):
        """
        Crop or pad a feature map based on crop factors.
        
        Args:
            feature_map: Input tensor of shape [B, C, H, W]
            
        Returns:
            Processed tensor that's either cropped or padded
        """
        B, C, H, W = feature_map.shape
        
        # Calculate new dimensions
        if self.crop_factor_H <= 1:  # Need to crop height
            new_H = int(H * self.crop_factor_H)
            start_H = (H - new_H) // 2
            end_H = start_H + new_H
        else:  # Need to pad height
            new_H = H
            pad_H = int(H * (self.crop_factor_H - 1))
            pad_top = pad_H // 2
            pad_bottom = pad_H - pad_top
        
        if self.crop_factor_W <= 1:  # Need to crop width
            new_W = int(W * self.crop_factor_W)
            start_W = (W - new_W) // 2
            end_W = start_W + new_W
        else:  # Need to pad width
            new_W = W
            pad_W = int(W * (self.crop_factor_W - 1))
            pad_left = pad_W // 2
            pad_right = pad_W - pad_left
        
        # Apply crop or pad
        if self.crop_factor_H <= 1 and self.crop_factor_W <= 1:
            # Both dimensions need cropping
            return feature_map[:, :, start_H:end_H, start_W:end_W]
        
        elif self.crop_factor_H > 1 and self.crop_factor_W > 1:
            # Both dimensions need padding
            pad = (pad_left, pad_right, pad_top, pad_bottom)
            return F.pad(feature_map, pad, mode='constant', value=0)
        
        elif self.crop_factor_H <= 1 and self.crop_factor_W > 1:
            # Crop height, pad width
            cropped = feature_map[:, :, start_H:end_H, :]
            pad = (pad_left, pad_right, 0, 0)
            return F.pad(cropped, pad, mode='constant', value=0)
        
        else:  # self.crop_factor_H > 1 and self.crop_factor_W <= 1
            # Pad height, crop width
            cropped = feature_map[:, :, :, start_W:end_W]
            pad = (0, 0, pad_top, pad_bottom)
            return F.pad(cropped, pad, mode='constant', value=0)
    
    

    def forward(self,  x, return_cls=False):
        # NOTE(YH): this return_cls is only used for where2com-style methods
        # which they need the foreground mask for sharing
        ori_x = None
        if self.target == 'dynamic':
            # x = [B, C, H', W']
            dynamic_map = self.dynamic_head(x)
            if return_cls:
                ori_x = dynamic_map
            dynamic_map_cropped = self.crop_or_pad_feature(dynamic_map)
            dynamic_map = F.interpolate(dynamic_map_cropped, size=(self.H, self.W),
                                   mode='bilinear', align_corners=False)
            static_map = torch.zeros_like(dynamic_map,
                                          device=dynamic_map.device)

        elif self.target == 'static':
            # NOTE(YH): static should have no return_cls
            static_map = self.static_head(x)
            if return_cls:
                ori_x = dynamic_map
            static_map_cropped = self.crop_or_pad_feature(static_map)
            static_map = F.interpolate(static_map_cropped, size=(self.H, self.W),
                                   mode='bilinear', align_corners=False)
            dynamic_map = torch.zeros_like(static_map,
                                           device=static_map.device)

        else:
            dynamic_map = self.dynamic_head(x)
            if return_cls:
                ori_x = dynamic_map
            dynamic_map_cropped = self.crop_or_pad_feature(dynamic_map)
            dynamic_map = F.interpolate(dynamic_map_cropped, size=(self.H, self.W),
                                   mode='bilinear', align_corners=False) 
            
            static_map = self.static_head(x)
            static_map_cropped = self.crop_or_pad_feature(static_map)
            static_map = F.interpolate(static_map_cropped, size=(self.H, self.W),
                                   mode='bilinear', align_corners=False)
        output_dict = {'static_seg': static_map,
                       'dynamic_seg': dynamic_map}

        if return_cls:
            return output_dict, ori_x

        return output_dict