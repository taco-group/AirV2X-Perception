# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# Modifier: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from matplotlib import pyplot as plt
from torch import nn
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

from debug_helper import *


class LiftSplatShootEncoder(nn.Module):
    def __init__(self, args, agent_type):
        super(LiftSplatShootEncoder, self).__init__()
        self.grid_conf = args[agent_type]["grid_conf"]  # 网格配置参数
        self.data_aug_conf = args[agent_type]["data_aug_conf"]  # 数据增强配置参数
        self.bevout_feature = args[agent_type]["bevout_feature"]
        self.agent_type = agent_type
        dx, bx, nx = gen_dx_bx(
            self.grid_conf["xbound"],
            self.grid_conf["ybound"],
            self.grid_conf["zbound"],
        )  # 划分网格

        self.dx = (
            dx.clone().detach().requires_grad_(False).to(torch.device("cuda"))
        )  # [0.4,0.4,20]
        self.bx = (
            bx.clone().detach().requires_grad_(False).to(torch.device("cuda"))
        )  # [-49.8,-49.8,0]
        self.nx = (
            nx.clone().detach().requires_grad_(False).to(torch.device("cuda"))
        )  # [250,250,1]

        self.downsample = args[agent_type]["img_downsample"]
        self.camC = args[agent_type]["img_features"]
        self.frustum = (
            self.create_frustum()
            .clone()
            .detach()
            .requires_grad_(False)
            .to(torch.device("cuda"))
        )  # frustum: DxfHxfWx3(41x8x16x3)

        self.D, _, _, _ = self.frustum.shape  # D: 41
        self.camera_encoder_type = args[agent_type]["camera_encoder"]
        if self.camera_encoder_type == "EfficientNet":
            self.camencode = CamEncode(
                self.D,
                self.camC,
                self.downsample,
                self.grid_conf["ddiscr"],
                self.grid_conf["mode"],
                args[agent_type]["use_depth_gt"],
                args[agent_type]["depth_supervision"],
            )
        elif self.camera_encoder_type == "Resnet101":
            self.camencode = CamEncode_Resnet101(
                self.D,
                self.camC,
                self.downsample,
                self.grid_conf["ddiscr"],
                self.grid_conf["mode"],
                args[agent_type]["use_depth_gt"],
                args[agent_type]["depth_supervision"],
            )

        self.bevencode = BevEncode(inC=self.camC, outC=self.bevout_feature)
        # self.shrink_flag = False
        # if "shrink_header" in args:
        #     self.shrink_flag = True
        #     self.shrink_conv = DownsampleConv(args["shrink_header"])

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf["final_dim"]  # 原始图片大小  ogfH:128  ogfW:288
        fH, fW = (
            ogfH // self.downsample,
            ogfW // self.downsample,
        )  # 下采样16倍后图像大小  fH: 12  fW: 22
        # ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)  # 在深度方向上划分网格 ds: DxfHxfW(41x12x22)
        ds = (
            torch.tensor(
                depth_discretization(*self.grid_conf["ddiscr"], self.grid_conf["mode"]),
                dtype=torch.float,
            )
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )

        D, _, _ = ds.shape  # D: 41 表示深度方向上网格的数量
        xs = (
            torch.linspace(0, ogfW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )  # 在0到288上划分18个格子 xs: DxfHxfW(41x12x22)
        ys = (
            torch.linspace(0, ogfH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )  # 在0到127上划分8个格子 ys: DxfHxfW(41x12x22)

        # D x H x W x 3
        frustum = torch.stack(
            (xs, ys, ds), -1
        )  # 堆积起来形成网格坐标, frustum[i,j,k,0]就是(i,j)位置，深度为k的像素的宽度方向上的栅格坐标   frustum: DxfHxfWx3
        return frustum

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape  # B:4(batchsize)    N: 4(相机数目)

        # undo post-transformation
        # B x N x D x H x W x 3
        # 抵消数据增强及预处理对像素的变化
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )

        # cam_to_ego
        points = torch.cat(
            (
                points[:, :, :, :, :, :2]
                * points[
                    :, :, :, :, :, 2:3
                ],  # points[:, :, :, :, :, 2:3] ranges from [4, 45) meters
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )  # 将像素坐标(u,v,d)变成齐次坐标(du,dv,d)
        # d[u,v,1]^T=intrins*rots^(-1)*([x,y,z]^T-trans)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(
            B, N, 1, 1, 1, 3
        )  # 将像素坐标d[u,v,1]^T转换到车体坐标系下的[x,y,z]

        return points  # B x N x D x H x W x 3 (4 x 4 x 41 x 16 x 22 x 3)

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C"""
        B, N, C, imH, imW = x.shape  # B: 4  N: 4  C: 3  imH: 600  imW: 800

        x = x.view(B * N, C, imH, imW)  # B和N两个维度合起来  x: 16 x 4 x 256 x 352
        # depth_items, x = self.camencode(
        #     x
        # )  # 进行图像编码  x: B*N x C x D x fH x fW(24 x 64 x 41 x 16 x 22)
        # x = x.view(
        #     B, N, self.camC, self.D, imH // self.downsample, imW // self.downsample
        # )  # 将前两维拆开 x: B x N x C x D x fH x fW(4 x 6 x 64 x 41 x 16 x 22)
        x = x[:, 3].unsqueeze(1)  # 只取深度通道  x: B*N x 1 x H x W(16 x 1 x 256 x 352)
        x = F.interpolate(x, size=(imH // self.downsample, imW // self.downsample), mode="nearest")
        x = x.view(B, N, 1, 1, imH // self.downsample, imW // self.downsample)
        
        x = x.permute(
            0, 1, 3, 4, 5, 2
        )  # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64)

        return x, None

    def voxel_pooling(self, geom_feats, x):
        # geom_feats: B x N x D x H x W x 3 (4 x 6 x 41 x 16 x 22 x 3), D is discretization in "UD" or "LID"
        # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64), D is num_bins

        B, N, D, H, W, C = x.shape  # B: 4  N: 6  D: 41  H: 16  W: 22  C: 64
        Nprime = B * N * D * H * W  # Nprime

        # flatten x
        x = x.reshape(Nprime, C)  # 将图像展平，一共有 B*N*D*H*W 个点

        # flatten indices

        geom_feats = (
            (geom_feats - (self.bx - self.dx / 2.0)) / self.dx
        ).long()  # 将[-48,48] [-10 10]的范围平移到 [0, 240), [0, 1) 计算栅格坐标并取整
        geom_feats = geom_feats.view(
            Nprime, 3
        )  # 将像素映射关系同样展平  geom_feats: B*N*D*H*W x 3
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )  # 每个点对应于哪个batch
        geom_feats = torch.cat(
            (geom_feats, batch_ix), 1
        )  # geom_feats: B*N*D*H*W x 4, geom_feats[:,3]表示batch_id

        # filter out points that are outside box
        # 过滤掉在边界线之外的点 x:0~240  y: 0~240  z: 0
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

        # get tensors from the same voxel next to each other
        ranks = (
            geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)
            + geom_feats[:, 1] * (self.nx[2] * B)
            + geom_feats[:, 2] * B
            + geom_feats[:, 3]
        )  # 给每一个点一个rank值，rank相等的点在同一个batch，并且在在同一个格子里面
        sorts = ranks.argsort()
        x, geom_feats, ranks = (
            x[sorts],
            geom_feats[sorts],
            ranks[sorts],
        )  # 按照rank排序，这样rank相近的点就在一起了
        # x: 168648 x 64  geom_feats: 168648 x 4  ranks: 168648

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(
                x, geom_feats, ranks
            )  # 一个batch的一个格子里只留一个点 x: 29072 x 64  geom_feats: 29072 x 4

        # griddify (B x C x Z x X x Y)
        # final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  # final: 4 x 64 x Z x X x Y
        # final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x  # 将x按照栅格坐标放到final中

        # modify griddify (B x C x Z x Y x X) by Yifan Lu 2022.10.7
        # ------> x
        # |
        # |
        # y
        final = torch.zeros(
            (B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device
        )  # final: 4 x 64 x Z x Y x X
        final[
            geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]
        ] = x  # 将x按照栅格坐标放到final中

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)  # 消除掉z维

        return final  # final: 4 x 64 x 240 x 240  # B, C, H, W

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(
            rots, trans, intrins, post_rots, post_trans
        )  # 像素坐标到自车中坐标的映射关系 geom: B x N x D x H x W x 3 ()
        x_img, depth_items = self.get_cam_feats(x)  # 提取图像特征并预测深度编码 x: B x N x 1 x fH x fW x 1()
        # x_img = depth_to_one_hot(x_img, num_bins=48) # B x N x D x fH x fW x 1()
        x_img = bin_depths(x_img, self.grid_conf["mode"], self.grid_conf["ddiscr"][0], self.grid_conf["ddiscr"][1], self.grid_conf["ddiscr"][2], target=True)[0]
        x_img = depth_to_one_hot(x_img)
        visualize_point_cloud_to_file(geom, x_img, "/home/xiangbog/Folder/Research/SkyLink/airv2x/debug/point_cloud.png")
        visualize_3d_points(geom, x_img, "/home/xiangbog/Folder/Research/SkyLink/airv2x/debug/point_cloud_v1.png")
        x = self.voxel_pooling(geom, x_img)  # x: 4 x 64 x 240 x 240

        return x, depth_items

    def forward(self, data_dict):
        image_inputs_dict = data_dict[self.agent_type]["batch_merged_cam_inputs"]
        x, rots, trans, intrins, post_rots, post_trans = (
            image_inputs_dict["imgs"],
            image_inputs_dict["rots"],
            image_inputs_dict["trans"],
            image_inputs_dict["intrinsics"],
            image_inputs_dict["post_rots"],
            image_inputs_dict["post_trans"],
        )
        x, depth_items = self.get_voxels(
            x, rots, trans, intrins, post_rots, post_trans
        )  # 将图像转换到BEV下，x: B x C x 240 x 240 (4 x 64 x 240 x 240)
        sp = "/home/xiangbog/Folder/Research/SkyLink/airv2x/debug/"
        cv2.imwrite(sp+'depth.png', image_inputs_dict["imgs"][0, 0, 3].cpu().numpy().astype(np.uint8))
        

        
        visualize_bev_feature(x!=0, batch_idx=0, channel_idx=None, save_path=sp+'bev_feature.png')
        import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        
        # x = self.bevencode(x)  # 用resnet18提取特征  x: 4 x C x 240 x 240

        output_dict = {
            "spatial_features": x,
            "spatial_features_3d": x.unsqueeze(2),
        }

        return output_dict
    
    
import numpy as np
import cv2
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


import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_point_cloud_to_file(geom: torch.Tensor,
                                  x_img: torch.Tensor,
                                  save_path: str = 'point_cloud.png',
                                  max_points: int = 100000):
    """
    Map depth-one-hot + geometry to 3D points, save scatter plot to file.

    Args:
        geom (torch.Tensor): (B, N, D, H, W, 3)
        x_img (torch.Tensor): (B, N, D, H, W, 1)
        save_path (str): output image file path
        max_points (int): max number of points to plot
    """
    depth_one_hot = x_img.squeeze(-1)                     # (B, N, D, H, W)
    pts = (depth_one_hot.unsqueeze(-1) * geom).sum(dim=2)  # (B, N, H, W, 3)
    pts = pts[0].reshape(-1, 3).cpu().numpy()           # 取 batch0/cam0

    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'Saved 3D point cloud to {save_path}')
    
    
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
import torch

def visualize_3d_points(geom, x_img, save_path='3d_points.png', threshold=0):
    """
    将深度编码的图像特征映射到3D空间并可视化
    
    参数:
        geom: B x N x D x H x W x 3 tensor, 像素坐标到自车坐标系的映射关系
        x_img: B x N x D x fH x fW x 1 tensor, 深度的one-hot编码
        save_path: 保存可视化结果的路径
        threshold: 深度bin被认为是有效的阈值
    """
    # 确保输入是numpy数组
    if isinstance(geom, torch.Tensor):
        geom = geom.detach().cpu().numpy()
    if isinstance(x_img, torch.Tensor):
        x_img = x_img.detach().cpu().numpy()
    
    # 只处理第一个batch
    batch_idx = 0
    
    # 创建一个带有3D轴的figure（无GUI模式）
    fig = plt.figure(figsize=(10, 8))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection='3d')
    
    # 获取维度信息
    B, N, D, H, W, _ = geom.shape
    B, N, D, fH, fW, _ = x_img.shape
    
    # 计算缩放因子（如果feature map的尺寸与原图像不同）
    scale_h = H / fH
    scale_w = W / fW
    
    # 遍历每个相机
    colors = plt.cm.rainbow(np.linspace(0, 1, N))  # 为每个相机分配不同的颜色
    
    for cam_idx in range(N):
        # 提取当前相机的几何映射和深度编码
        cam_geom = geom[batch_idx, cam_idx]  # D x H x W x 3
        cam_depth = x_img[batch_idx, cam_idx]  # D x fH x fW x 1
        
        # 移除最后一个维度，使cam_depth的形状为D x fH x fW
        cam_depth = cam_depth.squeeze(-1)  # D x fH x fW
        
        # 对于每个像素位置，找到值最大的深度bin（即one-hot中的1）
        depth_values = np.max(cam_depth, axis=0)  # fH x fW
        depth_indices = np.argmax(cam_depth, axis=0)  # fH x fW
        
        # 收集3D点
        points_3d = []
        
        # 遍历每个像素
        for h in range(fH):
            for w in range(fW):
                # 检查深度值是否大于阈值
                if depth_values[h, w] > threshold:
                    # 获取深度bin索引
                    d = depth_indices[h, w]
                    
                    # 计算对应原图像中的像素位置
                    orig_h = min(int(h * scale_h), H - 1)
                    orig_w = min(int(w * scale_w), W - 1)
                    
                    # 获取3D坐标
                    point_3d = cam_geom[d, orig_h, orig_w]
                    points_3d.append(point_3d)
        
        print(f"相机 {cam_idx+1}: 找到 {len(points_3d)} 个有效3D点")
        
        # 将点列表转换为numpy数组
        if points_3d:
            points_3d = np.array(points_3d)
            
            # 在3D图中绘制点
            ax.scatter(
                points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                c=[colors[cam_idx]], s=1, alpha=0.5, label=f'相机 {cam_idx+1}'
            )
    
    # 设置图的标题和标签
    ax.set_title('3D点云可视化')
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')
    ax.legend()
    
    # 保存图像而不显示（无GUI模式）
    fig.savefig(save_path)
    plt.close(fig)
    
    print(f"3D可视化结果已保存至 {save_path}")



class ResnetEncoder(nn.Module):
    """
    Resnet family to encode image.

    Parameters
    ----------
    params: dict
        The parameters of resnet encoder.
    """

    def __init__(self, params):
        super(ResnetEncoder, self).__init__()

        self.num_layers = params["num_layers"]
        self.pretrained = params["pretrained"]

        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }

        if self.num_layers not in resnets:
            raise ValueError(
                "{} is not a valid number of resnet layers".format(self.num_layers)
            )

        self.encoder = resnets[self.num_layers](self.pretrained)

    def forward(self, input_images):
        """
        Compute deep features from input images.
        todo: multi-scale feature support

        Parameters
        ----------
        input_images : torch.Tensor
            The input images have shape of (B,L,M,H,W,3), where L, M are
            the num of agents and num of cameras per agents.

        Returns
        -------
        features: torch.Tensor
            The deep features for each image with a shape of (B,L,M,C,H,W)
        """
        b, l, m, h, w, c = input_images.shape
        input_images = input_images.view(b * l * m, h, w, c)
        # b, h, w, c -> b, c, h, w
        input_images = input_images.permute(0, 3, 1, 2).contiguous()

        x = self.encoder.conv1(input_images)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)

        x = self.encoder.layer1(self.encoder.maxpool(x))
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        x = rearrange(x, "(b l m) c h w -> b l m c h w", b=b, l=l, m=m)

        return x
