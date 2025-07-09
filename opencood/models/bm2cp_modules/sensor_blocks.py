# Author: Binyu Zhao <byzhao@stu.hit.edu.cn>
# Author: Yue Hu <18671129361@sjtu.edu.cn>

from re import A

import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from einops import rearrange
from matplotlib import pyplot as plt
from torch import nn
from torchvision.models.resnet import resnet18

from opencood.models.bm2cp_modules.attentioncomm import ScaledDotProductAttention
from opencood.models.common_modules.pillar_vfe import PillarVFE
from opencood.models.common_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.lss_submodule import (
    BevEncode,
    CamEncode,
    CamEncode_Resnet101,
)
from opencood.utils import skylink_utils
from opencood.utils.camera_utils import (
    QuickCumsum,
    bin_depths,
    cumsum_trick,
    depth_discretization,
    gen_dx_bx,
)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=True
        )  # 上采样 BxCxHxW->BxCx2Hx2W

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

        # x1 = self.up(x1)
        # x1 = torch.cat([x2, x1], dim=1)
        # return self.conv(x1)


class ImgCamEncode(nn.Module):
    def __init__(
        self, D, C, downsample, ddiscr, mode, use_gt_depth=False, depth_supervision=True
    ):
        super(ImgCamEncode, self).__init__()
        self.D = D  # 42
        self.C = C  # 64
        self.downsample = downsample
        self.d_min = ddiscr[0]
        self.d_max = ddiscr[1]
        self.num_bins = ddiscr[2]
        self.mode = mode
        self.use_gt_depth = use_gt_depth
        self.depth_supervision = depth_supervision  # in the case of not use gt depth
        self.chain_channels = 256  # 512

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        print(
            "Number of parameter EfficientNet: %d"
            % (sum([param.nelement() for param in self.trunk.parameters()]))
        )

        self.up1 = Up(320 + 112, self.chain_channels)
        if downsample == 8:
            self.up2 = Up(self.chain_channels + 40, self.chain_channels)
        if not use_gt_depth:
            self.depth_head = nn.Conv2d(
                self.chain_channels, self.D, kernel_size=1, padding=0
            )
        self.image_head = nn.Conv2d(
            self.chain_channels, self.C, kernel_size=1, padding=0
        )

    def get_eff_features(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(
            self.trunk._bn0(self.trunk._conv_stem(x))
        )  #  x: 24 x 32 x 64 x 176
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(
                    self.trunk._blocks
                )  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints["reduction_{}".format(len(endpoints) + 1)] = prev_x
            prev_x = x

        # Head
        endpoints["reduction_{}".format(len(endpoints) + 1)] = x
        x = self.up1(endpoints["reduction_5"], endpoints["reduction_4"])
        if self.downsample == 8:
            x = self.up2(x, endpoints["reduction_3"])
        return x

    def forward(self, x, depth_maps, record_len):
        _, _, oriH, oriW = x.shape
        cum_sum_len = torch.cumsum(record_len, dim=0)

        if depth_maps.dim() == 5:
            # B, T, N, H, W -> B, N, H, W
            depth_map = depth_maps[:, 0, :, :, :]

            B, T, N, _, _ = depth_maps.shape
            assert T == 2, (
                f"T={T} should be 2, which first for self-image and second for ego-image"
            )

            ego_index = 0
            # get fused depth map for ego agent
            for next_ego_index in cum_sum_len:
                maps_for_ego = depth_maps[
                    ego_index:next_ego_index, 1, :, :, :
                ]  # size= [sum(cav), num(camera), H, W]
                max_value = torch.max(maps_for_ego)
                maps_for_ego[maps_for_ego < 0] = max_value + 1
                maps_for_ego, _ = torch.min(maps_for_ego, dim=0)
                maps_for_ego[maps_for_ego > max_value] = -1

                ego_depth_mask = (
                    (maps_for_ego[0]) > 0
                ).long()  # size= [num(camera), H, W]
                # torch.count_nonzero(), tensor.numel()
                depth_map[ego_index] = depth_map[
                    ego_index
                ] * ego_depth_mask + maps_for_ego * (1 - ego_depth_mask)

                # update index
                ego_index += next_ego_index
        else:
            # B, N, H, W
            depth_map = depth_maps

        x_img = x[:, :3:, :, :]  # origin x: (B*num(cav), C, H, W)
        features = self.get_eff_features(
            x_img
        )  # 8x downscale feature: (B*num(cav), set_channels(e.g.256), H/4, W/4)
        x_img = self.image_head(
            features
        )  #  8x downscale feature: B*N x C x fH x fW(24 x 64 x 8 x 22). C is the channel for next stage (i.e. bev)

        # resize depth
        batch, _, h, w = features.shape
        max_value = torch.max(depth_map)
        depth_map[depth_map < 0] = max_value + 1
        if oriH % h == 0 and oriW % w == 0:
            scaleh, scalew = oriH // h, oriW // w
            pool_layer = nn.MaxPool2d(
                kernel_size=(scaleh, scalew), stride=(scaleh, scalew)
            )
        else:
            pool_layer = nn.AdaptiveMaxPool2d((h, w), return_indices=False)
        depth_map = -1 * pool_layer(-1 * depth_map)
        depth_map[depth_map > max_value] = 0

        # generate one-hot refered ground truth
        # TODO: check shape here, bug here for now, figure out what is H, W should be
        depth_mask = ((depth_map) > 0).long().reshape(-1, 1, h, w)
        depth_map = depth_map.to(torch.int64).flatten(2).squeeze(1)
        one_hot_depth_map = []
        for batch_map in depth_map:
            one_hot_depth_map.append(F.one_hot(batch_map, num_classes=self.D))
        one_hot_depth_map = (
            torch.stack(one_hot_depth_map)
            .reshape(batch, h, w, self.D)
            .permute(0, 3, 1, 2)
        )  # [B*N, num_bins, fH, fW]

        depth_score = self.depth_head(features)
        depth_pred = F.softmax(depth_score, dim=1)

        final_depth = depth_mask * one_hot_depth_map + (1 - depth_mask) * depth_pred

        # size: final_depth=[B*num(cav), D, H, W]; x_img=[B*num(cav), C, H, W]
        new_x = final_depth.unsqueeze(1) * x_img.unsqueeze(2)
        # nodepth_x = depth_pred.unsqueeze(1) * x_img.unsqueeze(2)
        return x_img, new_x  # , nodepth_x


class ImgModalFusion(nn.Module):
    def __init__(self, dim, threshold=0.5):
        super().__init__()
        self.att = ScaledDotProductAttention(dim)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.Sigmoid()
        self.thres = threshold

    def forward(self, img_voxel, pc_voxel):
        B, C, imZ, imH, imW = pc_voxel.shape
        pc_voxel = pc_voxel.view(B, C, -1)
        img_voxel = img_voxel.view(B, C, -1)
        voxel_mask = self.att(pc_voxel, img_voxel, img_voxel)
        voxel_mask = self.act(self.proj(voxel_mask.permute(0, 2, 1)))
        voxel_mask = voxel_mask.permute(0, 2, 1)
        voxel_mask = voxel_mask.view(B, C, imZ, imH, imW)

        ones_mask = torch.ones_like(voxel_mask).to(voxel_mask.device)
        zeros_mask = torch.zeros_like(voxel_mask).to(voxel_mask.device)
        mask = torch.where(voxel_mask > self.thres, ones_mask, zeros_mask)

        mask[0] = ones_mask[0]

        img_voxel = img_voxel.view(B, C, imZ, imH, imW)
        return mask


class MultiModalFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.img_fusion = ImgModalFusion(dim)

        self.multigate = nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU(inplace=True)
        self.multifuse = nn.Conv3d(dim * 2, dim, 1, 1, 0)

    def forward(self, img_voxel, pc_dict):
        pc_voxel = pc_dict["spatial_features_3d"]
        B, C, Z, Y, X = pc_voxel.shape

        # pc->pc; img->img*mask; pc+img->
        ones_mask = torch.ones_like(pc_voxel).to(pc_voxel.device)
        zeros_mask = torch.zeros_like(pc_voxel).to(pc_voxel.device)
        mask = torch.ones_like(pc_voxel).to(pc_voxel.device)

        pc_mask = torch.where(pc_voxel != 0, ones_mask, zeros_mask)
        pc_mask, _ = torch.max(pc_mask, dim=1)
        pc_mask = pc_mask.unsqueeze(1)
        # FIXME(YH): error here, why pc_voxel's B sometimes < img_voxel's B?
        # the reason is when we set a large COMM_RANGE, the cav project pcd may be 0 (too long comm range)
        img_mask = torch.where(img_voxel != 0, ones_mask, zeros_mask)
        img_mask, _ = torch.max(img_mask, dim=1)
        img_mask = img_mask.unsqueeze(1)

        fused_voxel = (
            pc_mask
            * img_mask
            * self.multifuse(
                torch.cat(
                    [self.act(self.multigate(pc_voxel)) * img_voxel, pc_voxel], dim=1
                )
            )
        )
        fused_voxel = (
            fused_voxel
            + pc_voxel * pc_mask * (1 - img_mask)
            + img_voxel
            * self.img_fusion(img_voxel, pc_voxel)
            * (1 - pc_mask)
            * img_mask
        )

        thres_map = (
            pc_mask * img_mask * 0
            + pc_mask * (1 - img_mask) * 0.5
            + (1 - pc_mask) * img_mask * 0.5
            + (1 - pc_mask) * (1 - img_mask) * 0.5
        )
        mask = (
            pc_mask * img_mask
            + pc_mask * (1 - img_mask) * 2
            + (1 - pc_mask) * img_mask * 3
            + (1 - pc_mask) * (1 - img_mask) * 4
        )
        mask1 = pc_mask
        mask2 = img_mask
        # size = [B, 1, Z, Y, X]
        thres_map, _ = torch.min(
            thres_map, dim=2
        )  # collapse Z-axis, dim=4 size = [B, 1, Y, X]
        mask1, _ = torch.max(mask1, dim=2)  # collapse Z-axis, dim=4 size = [B, 1, Y, X]
        mask2, _ = torch.max(mask2, dim=2)  # collapse Z-axis, dim=4 size = [B, 1, Y, X]

        pc_dict["spatial_features"] = fused_voxel.view(B, C * Z, Y, X)
        return (
            pc_dict,
            thres_map,
            torch.min(mask, dim=2)[0],
            torch.stack([mask1, mask2]),
        )


class LidarCamBM2CPEncoder(nn.Module):
    def __init__(self, args, agent_type="vehicle"):
        super(LidarCamBM2CPEncoder, self).__init__()
        # cuda选择
        self.device = args["device"] if "device" in args else "cpu"
        self.supervise_single = (
            args["supervise_single"] if "supervise_single" in args else False
        )

        self.agent_type = agent_type
        agent_args = args[agent_type]
        # camera branch
        img_args = agent_args["img_params"]
        self.grid_conf = img_args["grid_conf"]
        self.data_aug_conf = img_args["data_aug_conf"]
        self.downsample = img_args["img_downsample"]
        self.bevC = img_args["bev_dim"]
        self.use_quickcumsum = True

        dx, bx, nx = gen_dx_bx(
            self.grid_conf["xbound"],
            self.grid_conf["ybound"],
            self.grid_conf["zbound"],
        )
        self.dx = (
            dx.clone().detach().requires_grad_(False).to(torch.device(self.device))
        )
        self.bx = (
            bx.clone().detach().requires_grad_(False).to(torch.device(self.device))
        )
        self.nx = (
            nx.clone().detach().requires_grad_(False).to(torch.device(self.device))
        )
        self.frustum = (
            self.create_frustum()
            .clone()
            .detach()
            .requires_grad_(False)
            .to(torch.device(self.device))
        )  # frustum: DxfHxfWx3
        self.D, _, _, _ = self.frustum.shape
        print("total depth levels: ", self.D)
        self.camencode = ImgCamEncode(
            self.D,
            self.bevC,
            self.downsample,
            self.grid_conf["ddiscr"],
            self.grid_conf["mode"],
            img_args["use_depth_gt"],
            img_args["depth_supervision"],
        )
        # print(
        #     "Number of parameter CamEncode: %d"
        #     % (sum([param.nelement() for param in self.camencode.parameters()]))
        # )

        # lidar branch
        pc_args = agent_args["pc_params"]
        self.pillar_vfe = PillarVFE(
            pc_args["pillar_vfe"],
            num_point_features=4,
            voxel_size=pc_args["voxel_size"],
            point_cloud_range=pc_args["lidar_range"],
        )
        # print(
        #     "Number of parameter pillar_vfe: %d"
        #     % (sum([param.nelement() for param in self.pillar_vfe.parameters()]))
        # )
        self.scatter = PointPillarScatter(pc_args["point_pillar_scatter"])
        # print(
        #     "Number of parameter scatter: %d"
        #     % (sum([param.nelement() for param in self.scatter.parameters()]))
        # )
        self.intra_fusion = MultiModalFusion(args[agent_type]["img_params"]["bev_dim"])

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf["final_dim"]
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = (
            torch.tensor(
                depth_discretization(*self.grid_conf["ddiscr"], self.grid_conf["mode"]),
                dtype=torch.float,
            )
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )

        D, _, _ = ds.shape
        xs = (
            torch.linspace(0, ogfW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, ogfH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return frustum

    def forward(self, data_dict):  # loss: 5.91->0.76
        # get two types data
        image_inputs_dict = data_dict[self.agent_type]["batch_merged_cam_inputs"]
        pc_inputs_dict = data_dict[self.agent_type]["batch_merged_lidar_features_torch"]
        record_len = data_dict[self.agent_type]["record_len"]

        batch_dict = {
            "voxel_features": pc_inputs_dict["voxel_features"],
            "voxel_coords": pc_inputs_dict["voxel_coords"],
            "voxel_num_points": pc_inputs_dict["voxel_num_points"],
            # "record_len": record_len,
        }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        # batch_dict = self.backbone(batch_dict)
        # spatial_features_2d = batch_dict['spatial_features_2d']

        # process image to get bev
        geom = self.get_geometry(image_inputs_dict)

        x = image_inputs_dict["imgs"]
        B, N, C, imH, imW = x.shape  # torch.Size([4, 1, 3, 320, 480])
        x = x.view(B * N, C, imH, imW)
        _, x = self.camencode(
            x, data_dict[self.agent_type]["depth_maps_torch"], record_len
        )
        # x = x.view(B, N, self.bevC, self.D, imH//self.downsample, imW//self.downsample)
        x = rearrange(x, "(b l) c d h w -> b l c d h w", b=B, l=N)
        x = x.permute(
            0, 1, 3, 4, 5, 2
        )  # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64)

        # FIXME(YH): voxel pooling leads to (B, C, 1, 192, 704)
        x = self.voxel_pooling(geom, x)  # x: 4 x 64 x 240 x 240
        # # collapse Z
        # x = torch.cat(x.unbind(dim=2), 1)  # [B, C, H, W]

        # intra fusion
        pc_dict, thres_map, mask, each_mask = self.intra_fusion(x, batch_dict)
        output_dict = {
            **pc_dict,
            "thres_map": thres_map,
            "mask": mask,
            "each_mask": each_mask,
        }
        return output_dict
        # return x, batch_dict

    def get_geometry(self, image_inputs_dict):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        # process image to get bev
        rots, trans, intrins, post_rots, post_trans = (
            image_inputs_dict["rots"],
            image_inputs_dict["trans"],
            image_inputs_dict["intrinsics"],
            image_inputs_dict["post_rots"],
            image_inputs_dict["post_trans"],
        )

        B, N, _ = trans.shape  # B:4(batchsize)

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        if post_rots.device != "cpu":
            inv_post_rots = torch.inverse(post_rots.to("cpu")).to(post_rots.device)
        else:
            inv_post_rots = torch.inverse(post_rots)
        points = inv_post_rots.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

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
        )

        if intrins.device != "cpu":
            inv_intrins = torch.inverse(intrins.to("cpu")).to(intrins.device)
        else:
            inv_intrins = torch.inverse(intrins)

        combine = rots.matmul(inv_intrins)
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points  # B x N x D x H x W x 3 (4 x 1 x 41 x 16 x 22 x 3)

    def voxel_pooling(self, geom_feats, x):
        # geom_feats: B x N x D x H x W x 3 (4 x 6 x 41 x 16 x 22 x 3), D is discretization in "UD" or "LID"
        # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64), D is num_bins

        B, N, D, H, W, C = x.shape  # B: 4  N: 6  D: 41  H: 16  W: 22  C: 64
        Nprime = B * N * D * H * W  # Nprime

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
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
        )
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        # final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  # final: 4 x 64 x Z x X x Y
        # final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

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
        ] = x

        # # collapse Z
        # collapsed_final = torch.cat(final.unbind(dim=2), 1)

        # return collapsed_final  # final: 4 x 64 x 192 x 704  # B, C, H, W
        return final


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
        self.intra_fusion = MultiModalFusion(args[agent_type]["bev_dim"])
        # self.shrink_flag = False
        # if "shrink_header" in args:
        #     self.shrink_flag = True
        #     self.shrink_conv = DownsampleConv(args["shrink_header"])

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

        # for p in self.parameters():
        #     p.requires_grad = False
        # for p in self.camencode.depth_head.parameters():
        #     p.requires_grad = True
        #     print("freeze ",p)

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
        depth_items, x = self.camencode(
            x
        )  # 进行图像编码  x: B*N x C x D x fH x fW(24 x 64 x 41 x 16 x 22)
        x = x.view(
            B, N, self.camC, self.D, imH // self.downsample, imW // self.downsample
        )  # 将前两维拆开 x: B x N x C x D x fH x fW(4 x 6 x 64 x 41 x 16 x 22)
        x = x.permute(
            0, 1, 3, 4, 5, 2
        )  # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64)

        return x, depth_items

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

        x_img, depth_items = self.get_cam_feats(
            x
        )  # 提取图像特征并预测深度编码 x: B x N x D x fH x fW x C()

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

        x = self.bevencode(x)  # 用resnet18提取特征  x: 4 x C x 240 x 240
        x_3d = x.unsqueeze(2)
        num_drones = x.shape[0]
        mock_lidar = skylink_utils.mock_lidar_for_drone(num_drones, x.device)

        # intra fusion
        pc_dict, thres_map, mask, each_mask = self.intra_fusion(x_3d, mock_lidar)
        output_dict = {
            **pc_dict,
            "thres_map": thres_map,
            "mask": mask,
            "each_mask": each_mask,
        }

        return output_dict
