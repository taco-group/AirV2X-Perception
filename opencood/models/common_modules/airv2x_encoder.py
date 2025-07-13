# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# Modifier: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torchvision.models as models
from einops import rearrange

from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.lss_submodule import (
    BevEncode,
    CamEncode,
    CamEncode_Resnet101,
)
from opencood.utils.camera_utils import (
    QuickCumsum,
    bin_depths,
    cumsum_trick,
    depth_discretization,
    gen_dx_bx,
)

from opencood.models.common_modules.debug_helper import *

VISUALIZE = False

class LiftSplatShootEncoder(nn.Module):
    def __init__(self, args, agent_type):
        super(LiftSplatShootEncoder, self).__init__()
        self.grid_conf = args["grid_conf"]  # Grid configuration parameters
        self.data_aug_conf = args["data_aug_conf"]  # Data augmentation parameters
        self.bevout_feature = args["bevout_feature"]
        self.agent_type = agent_type
        
        # Generate grid parameters
        dx, bx, nx = gen_dx_bx(
            self.grid_conf["xbound"],
            self.grid_conf["ybound"],
            self.grid_conf["zbound"],
        )

        # Store grid parameters on GPU
        self.dx = dx.clone().detach().requires_grad_(False).to(torch.device("cuda"))  # [0.4, 0.4, 20]
        self.bx = bx.clone().detach().requires_grad_(False).to(torch.device("cuda"))  # [-49.8, -49.8, 0]
        self.nx = nx.clone().detach().requires_grad_(False).to(torch.device("cuda"))  # [250, 250, 1]

        self.downsample = args["img_downsample"]
        self.camC = args["img_features"]
        
        # Create and store frustum on GPU
        self.frustum = (
            self.create_frustum()
            .clone()
            .detach()
            .requires_grad_(False)
            .to(torch.device("cuda"))
        )  # frustum: DxfHxfWx3(41x8x16x3)

        self.D, _, _, _ = self.frustum.shape  # D: 41 (depth bins)
        
        # Initialize camera encoder based on type
        self.camera_encoder_type = args["camera_encoder"]
        if self.camera_encoder_type == "EfficientNet":
            self.camencode = CamEncode(
                self.D,
                self.camC,
                self.downsample,
                self.grid_conf["ddiscr"],
                self.grid_conf["mode"],
                args["use_depth_gt"],
                args["depth_supervision"],
            )
        elif self.camera_encoder_type == "Resnet101":
            self.camencode = CamEncode_Resnet101(
                self.D,
                self.camC,
                self.downsample,
                self.grid_conf["ddiscr"],
                self.grid_conf["mode"],
                args["use_depth_gt"],
                args["depth_supervision"],
            )

        # Initialize BEV encoder
        self.bevencode = BevEncode(inC=self.camC, outC=self.bevout_feature)

        # Toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

    def create_frustum(self):
        """
        Create a frustum grid in the image plane.
        
        Returns:
            torch.Tensor: Frustum coordinates of shape DxfHxfWx3
        """
        # Get original and downsampled image dimensions
        ogfH, ogfW = self.data_aug_conf["final_dim"]  # Original image size: ogfH=128, ogfW=288
        fH, fW = ogfH // self.downsample, ogfW // self.downsample  # Downsampled size: fH=12, fW=22
        
        # Create depth discretization
        ds = (
            torch.tensor(
                depth_discretization(*self.grid_conf["ddiscr"], self.grid_conf["mode"]),
                dtype=torch.float,
            )
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )

        D, _, _ = ds.shape  # D: 41 (number of depth bins)
        
        # Create coordinate grids
        xs = (
            torch.linspace(0, ogfW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )  # Width coordinates
        ys = (
            torch.linspace(0, ogfH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )  # Height coordinates

        # Stack coordinates to form frustum grid
        frustum = torch.stack((xs, ys, ds), -1)  # DxfHxfWx3
        return frustum

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """
        Determine the (x,y,z) locations in the ego frame for points in the point cloud.
        
        Args:
            rots, trans, intrins, post_rots, post_trans: Camera parameters
            
        Returns:
            torch.Tensor: Points in ego frame, shape B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape  # B: batch size, N: number of cameras

        # Undo post-transformation (data augmentation)
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )

        # Convert to camera coordinates
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        
        # Transform to ego frame
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x):
        """
        Extract camera features and depth information.
        
        Args:
            x: Input tensor of shape B x N x C x imH x imW
            
        Returns:
            tuple: Camera features and depth items
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B * N, C, imH, imW)
        depth_items, x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH // self.downsample, imW // self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x, depth_items
    
    def get_depth_feats(self, x):
        """
        Extract depth features from input tensor.
        
        Args:
            x: Input tensor of shape B x N x C x imH x imW
            
        Returns:
            torch.Tensor: Depth features
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B * N, C, imH, imW)
        x = x[:, 3].unsqueeze(1)  # Extract depth channel
        x = F.interpolate(x, size=(imH // self.downsample, imW // self.downsample), mode="nearest")
        x = x.view(B, N, 1, 1, imH // self.downsample, imW // self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def voxel_pooling(self, geom_feats, x):
        """
        Pool features into voxel grid.
        
        Args:
            geom_feats: Geometry features of shape B x N x D x H x W x 3
            x: Input features of shape B x N x D x fH x fW x C
            
        Returns:
            torch.Tensor: Pooled features in BEV
        """
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # Flatten features
        x = x.reshape(Nprime, C)

        # Convert to voxel coordinates
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        
        # Create batch indices
        batch_ix = torch.cat(
            [torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
             for ix in range(B)]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # Filter points outside the grid
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        # Sort by voxel and batch
        ranks = (
            geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)
            + geom_feats[:, 1] * (self.nx[2] * B)
            + geom_feats[:, 2] * B
            + geom_feats[:, 3]
        )
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # Cumulative sum pooling
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # Reshape to BEV grid (B x C x Z x Y x X)
        final = torch.zeros(
            (B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device
        )
        final[
            geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]
        ] = x

        # Collapse Z dimension
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans, i=None):
        """
        Convert image features to voxel representation.
        
        Args:
            x, rots, trans, intrins, post_rots, post_trans: Input tensors
            i: Optional index for visualization
            
        Returns:
            tuple: Voxel features, depth items, and depth
        """
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        
        if i == 0 and VISUALIZE:
            geom_cpu = geom.detach().cpu()
            visualize_bev_points(geom_cpu)
            visualize_cameras_frustum(geom_cpu)
            visualize_density(geom_cpu)

        x_img, depth_items = self.get_cam_feats(x)
        
        if VISUALIZE:
            depth = self.get_depth_feats(x)
            depth = bin_depths(depth, self.grid_conf["mode"], self.grid_conf["ddiscr"][0], 
                             self.grid_conf["ddiscr"][1], self.grid_conf["ddiscr"][2], target=True)[0]
            cv2.imwrite(f"debug/depth_encoded_drone_{i}.png", depth[0,0].squeeze().cpu().numpy().astype(np.uint8))
            depth = depth_to_one_hot(depth)
            visualize_3d_points(geom, depth, f"debug/point_cloud_v1_drone_{i}.png")

        x = self.voxel_pooling(geom, x_img)
        return x, depth_items, None
    
    def forward(self, data_dict):
        """
        Forward pass of the network.
        
        Args:
            data_dict: Dictionary containing input data
            
        Returns:
            dict: Output dictionary with spatial features
        """
        image_inputs_dict = data_dict[self.agent_type]["batch_merged_cam_inputs"]
        x, rots, trans, intrins, post_rots, post_trans = (
            image_inputs_dict["imgs"],
            image_inputs_dict["rots"],
            image_inputs_dict["trans"],
            image_inputs_dict["intrinsics"],
            image_inputs_dict["post_rots"],
            image_inputs_dict["post_trans"],
        )
        x, depth_items, depth = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)

        output_dict = {
            "spatial_features": x,
            "spatial_features_3d": x.unsqueeze(2),
        }
        return output_dict


def visualize_bev_feature(
        bev_feat: torch.Tensor,
        batch_idx: int = 0,
        channel_idx: int = None,
        save_path: str = None
    ) -> np.ndarray:
    """
    Visualize a BEV feature map as a heatmap.

    Args:
        bev_feat (torch.Tensor): Input tensor of shape [B, C, H, W].
        batch_idx (int): Index of the sample in the batch to visualize. Default is 0.
        channel_idx (int): If specified, visualize only this channel; otherwise, average across all channels.
        save_path (str): If given, save the heatmap image to this path; otherwise, display it in a window.

    Returns:
        np.ndarray: The resulting BGR heatmap image (dtype uint8).
    """
    # Extract the chosen sample and move to CPU numpy array
    if isinstance(bev_feat, torch.Tensor):
        array = bev_feat[batch_idx].detach().cpu().numpy()  # shape: (C, H, W)
    else:
        array = bev_feat  # assume already a numpy array

    # Select a single channel or average across channels
    if channel_idx is None:
        feature = np.mean(array, axis=0)  # shape: (H, W)
    else:
        feature = array[channel_idx]      # shape: (H, W)

    # Normalize values to [0, 255]
    # min_val, max_val = feature.min(), feature.max()
    # normalized = (feature - min_val) / (max_val - min_val + 1e-6)
    gray_img = (feature * 255).astype(np.uint8)

    # Save or display the result
    if save_path:
        cv2.imwrite(save_path, gray_img)
    else:
        cv2.imshow("BEV Feature Heatmap", heatmap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def visualize_bev_feature_in_once(sp):

    imgs = [Image.open(sp + f"bev_feature_{i}.png").convert('L') for i in range(6)]

    w, h = imgs[0].size  

    # 2. 定义 6 种 RGB 颜色
    colors = [
        (255,   0,   0),
        (  0, 255,   0),
        (  0,   0, 255),
        (255, 255,   0),
        (255,   0, 255),
        (  0, 255, 255),
    ]

    # 3. 累加上色
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    for gray, col in zip(imgs, colors):
        arr = np.array(gray, dtype=np.float32) / 255.0  # 归一化
        for c in range(3):
            canvas[..., c] += arr * col[c]

    # 4. 裁剪并转回 uint8
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    out = Image.fromarray(canvas)

    # 5. 保存或展示
    out.save(sp+"bev_overlay.png")
    # 或者直接显示


def depth_to_one_hot(x, num_bins=41, min_val=0.0, max_val=256.0):
    """
    Args:
        x: Tensor of shape [B, N, 1, fH, fW, C], depth values in [min_val, max_val]
        num_bins: number of one-hot bins (here 41)
        min_val: minimum depth value
        max_val: maximum depth value
    Returns:
        Tensor of shape [B, N, num_bins, fH, fW, C], one-hot along the depth dimension
    """
    # remove the singleton depth dim
    depth = x.squeeze(2)           # -> [B, N, fH, fW, C]
    # compute bin edges
    num_bins = int(depth.max()) + 1
    

    # find which bin each depth value falls into: 0 .. num_bins-1
    # torch.bucketize returns indices in [1..num_bins], so subtract 1
    # idx = torch.bucketize(depth, edges) - 1
    # idx = idx.clamp(0, num_bins-1)  # just in case any value == max_val

    # one-hot encode
    # idx: [B, N, fH, fW, C] -> one_hot: [..., num_bins]
    one_hot = F.one_hot(depth, num_classes=num_bins)  # -> [B,N,fH,fW,C,num_bins]
    
    # move the bin axis into position 2
    one_hot = one_hot.permute(0, 1, 5, 2, 3, 4).float()  # -> [B,N,num_bins,fH,fW,C]
    return one_hot
