"""
# Author: Binyu Zhao <byzhao@stu.hit.edu.cn>
"""

import torch
from einops import rearrange
from numpy import record
from torch import nn
from torchvision.models.resnet import resnet18

from opencood.models.bm2cp_modules.attentioncomm import (
    AttenComm,
    ScaledDotProductAttention,
)
from opencood.models.bm2cp_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.bm2cp_modules.sensor_blocks import ImgCamEncode
from opencood.models.common_modules.base_bev_backbone import (
    BaseBEVBackbone as PCBaseBEVBackbone,
)
from opencood.models.common_modules.downsample_conv import DownsampleConv
from opencood.models.common_modules.naive_compress import NaiveCompressor
from opencood.models.common_modules.pillar_vfe import PillarVFE
from opencood.models.common_modules.point_pillar_scatter import PointPillarScatter
from opencood.utils.camera_utils import (
    QuickCumsum,
    cumsum_trick,
    depth_discretization,
    gen_dx_bx,
)


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


class PointPillarBM2CP(nn.Module):
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

    def __init__(self, args):
        super(PointPillarBM2CP, self).__init__()
        # cuda选择
        self.device = args["device"] if "device" in args else "cpu"
        self.supervise_single = (
            args["supervise_single"] if "supervise_single" in args else False
        )

        # camera branch
        img_args = args["img_params"]
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
        )  # [0.4,0.4,20]
        self.bx = (
            bx.clone().detach().requires_grad_(False).to(torch.device(self.device))
        )  # [-49.8,-49.8,0]
        self.nx = (
            nx.clone().detach().requires_grad_(False).to(torch.device(self.device))
        )  # [250,250,1]
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
        print(
            "Number of parameter CamEncode: %d"
            % (sum([param.nelement() for param in self.camencode.parameters()]))
        )

        # lidar branch
        pc_args = args["pc_params"]
        self.pillar_vfe = PillarVFE(
            pc_args["pillar_vfe"],
            num_point_features=4,
            voxel_size=pc_args["voxel_size"],
            point_cloud_range=pc_args["lidar_range"],
        )
        print(
            "Number of parameter pillar_vfe: %d"
            % (sum([param.nelement() for param in self.pillar_vfe.parameters()]))
        )
        self.scatter = PointPillarScatter(pc_args["point_pillar_scatter"])
        print(
            "Number of parameter scatter: %d"
            % (sum([param.nelement() for param in self.scatter.parameters()]))
        )

        # multi-modal fusion
        modality_args = args["modality_fusion"]
        self.modal_multi_scale = modality_args["bev_backbone"]["multi_scale"]
        self.num_levels = len(modality_args["bev_backbone"]["layer_nums"])
        assert img_args["bev_dim"] == pc_args["point_pillar_scatter"]["num_features"]
        self.fusion = MultiModalFusion(img_args["bev_dim"])
        print(
            "Number of parameter modal fusion: %d"
            % (sum([param.nelement() for param in self.fusion.parameters()]))
        )
        self.backbone = ResNetBEVBackbone(
            modality_args["bev_backbone"],
            input_channels=pc_args["point_pillar_scatter"]["num_features"],
        )
        print(
            "Number of parameter bevbackbone: %d"
            % (sum([param.nelement() for param in self.backbone.parameters()]))
        )

        self.shrink_flag = False
        if "shrink_header" in modality_args:
            self.shrink_flag = modality_args["shrink_header"]["use"]
            self.shrink_conv = DownsampleConv(modality_args["shrink_header"])
            print(
                "Number of parameter shrink_conv: %d"
                % (sum([param.nelement() for param in self.shrink_conv.parameters()]))
            )

        self.compression = False
        if "compression" in modality_args and modality_args["compression"] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, modality_args["compression"])
        # map_channels = input_dim//compress_raito if self.compression else modality_args['shrink_header']['dim'][0] if self.shrink_flag else sum(modality_args['bev_backbone']['num_upsample_filter'])

        # collaborative fusion network
        self.multi_scale = args["collaborative_fusion"]["multi_scale"]
        self.fusion_net = AttenComm(args["collaborative_fusion"])
        print(
            "Number of fusion_net parameter: %d"
            % (sum([param.nelement() for param in self.fusion_net.parameters()]))
        )

        self.outC = args["outC"]
        self.cls_head = nn.Conv2d(self.outC, args["anchor_number"], kernel_size=1)
        self.reg_head = nn.Conv2d(self.outC, 7 * args["anchor_number"], kernel_size=1)
        if "dir_args" in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(
                self.outC,
                args["dir_args"]["num_bins"] * args["anchor_number"],
                kernel_size=1,
            )  # BIN_NUM = 2
        else:
            self.use_dir = False

        # freeze
        if args["backbone_fix"]:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.bevbackbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, data_dict):  # loss: 5.91->0.76
        # get two types data
        image_inputs_dict = data_dict["image_inputs"]
        pc_inputs_dict = data_dict["processed_lidar"]
        record_len = data_dict["record_len"]

        batch_dict = {
            "voxel_features": pc_inputs_dict["voxel_features"],
            "voxel_coords": pc_inputs_dict["voxel_coords"],
            "voxel_num_points": pc_inputs_dict["voxel_num_points"],
            "record_len": record_len,
        }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        # batch_dict = self.backbone(batch_dict)
        # spatial_features_2d = batch_dict['spatial_features_2d']

        # process image to get bev
        # x, rots, trans, intrins, post_rots, post_trans, depth_map = image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'], image_inputs_dict['post_rots'], image_inputs_dict['post_trans'], image_inputs_dict['depth_map']
        # geom: ([8, 1, 48, 40, 60, 3]), x: torch.Size([8, 1, 48, 40, 60, 64])
        geom = self.get_geometry(image_inputs_dict)

        x = image_inputs_dict["imgs"]
        B, N, C, imH, imW = x.shape  # torch.Size([4, 1, 3, 320, 480])
        x = x.view(B * N, C, imH, imW)
        _, x = self.camencode(x, image_inputs_dict["depth_map"], record_len)
        # x = x.view(B, N, self.bevC, self.D, imH//self.downsample, imW//self.downsample)
        x = rearrange(x, "(b l) c d h w -> b l c d h w", b=B, l=N)
        x = x.permute(
            0, 1, 3, 4, 5, 2
        )  # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64)

        x = self.voxel_pooling(geom, x)  # x: 4 x 64 x 240 x 240
        # collapse Z
        # x = torch.cat(x.unbind(dim=2), 1)

        # modal fusion in voxel space. img: B*C*Z*Y*X; pc: B*C*Z*Y*X
        batch_dict, thres_map, mask, each_mask = self.fusion(x, batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict["spatial_features_2d"]
        if self.shrink_flag:  # downsample feature to reduce memory
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        if self.compression:  # compressor
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        # collaborative fusion
        pairwise_t_matrix = data_dict["pairwise_t_matrix"]

        if self.multi_scale:
            fused_feature, communication_rates, result_dict = self.fusion_net(
                batch_dict["spatial_features"],
                self.cls_head(spatial_features_2d),
                thres_map,
                record_len,
                pairwise_t_matrix,
                self.backbone,
                [self.shrink_conv, self.cls_head, self.reg_head],
            )
            # downsample feature to reduce memory
            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)
        else:
            fused_feature, communication_rates, result_dict = self.fusion_net(
                spatial_features_2d,
                self.cls_head(spatial_features_2d),
                thres_map,
                record_len,
                pairwise_t_matrix,
            )

        # decode head
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        # update output dict
        output_dict = {"psm": psm, "rm": rm}

        if self.use_dir:
            dm = self.dir_head(fused_feature)
            output_dict.update({"dm": dm})

        output_dict.update(
            {"mask": mask, "each_mask": each_mask, "comm_rate": communication_rates}
        )

        if not self.supervise_single:
            return output_dict

        # single decode head
        psm_single = self.cls_head(spatial_features_2d)
        rm_single = self.reg_head(spatial_features_2d)
        split_psm_single = self.regroup(psm_single, record_len)
        split_rm_single = self.regroup(rm_single, record_len)
        psm_single_v = []
        psm_single_i = []
        rm_single_v = []
        rm_single_i = []

        for b in range(len(split_psm_single)):
            psm_single_v.append(split_psm_single[b][0:1])
            psm_single_i.append(split_psm_single[b][1:2])
            rm_single_v.append(split_rm_single[b][0:1])
            rm_single_i.append(split_rm_single[b][1:2])
        psm_single_v = torch.cat(psm_single_v, dim=0)
        psm_single_i = torch.cat(psm_single_i, dim=0)
        rm_single_v = torch.cat(rm_single_v, dim=0)
        rm_single_i = torch.cat(rm_single_i, dim=0)
        output_dict.update(
            {
                "psm_single_v": psm_single_v,
                "psm_single_i": psm_single_i,
                "rm_single_v": rm_single_v,
                "rm_single_i": rm_single_i,
                "mask": mask,
                "comm_rate": communication_rates,
            }
        )

        return output_dict

    def get_geometry(self, image_inputs_dict):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        # process image to get bev
        rots, trans, intrins, post_rots, post_trans = (
            image_inputs_dict["rots"],
            image_inputs_dict["trans"],
            image_inputs_dict["intrins"],
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

        # collapse Z
        # collapsed_final = torch.cat(final.unbind(dim=2), 1)

        # return collapsed_final#, x  # final: 4 x 64 x 240 x 240  # B, C, H, W
        return final
