from typing import Any, Callable, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from opencood.models.common_modules.resblock import BasicBlock, Bottleneck, conv1x1


class ResNetModified(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        layer_strides: List[int],
        num_filters: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(ResNetModified, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 128
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(
            block, num_filters[0], layers[0], stride=layer_strides[0]
        )
        self.layer2 = self._make_layer(
            block,
            num_filters[1],
            layers[1],
            stride=layer_strides[1],
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            num_filters[2],
            layers[2],
            stride=layer_strides[2],
            dilate=replace_stride_with_dilation[1],
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, return_interm: bool = True):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        if return_interm:
            return (x1, x2, x3)
        return x3

    def forward(self, x: Tensor):
        return self._forward_impl(x)


class ResNetBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if "layer_nums" in self.model_cfg:
            assert (
                len(self.model_cfg["layer_nums"])
                == len(self.model_cfg["layer_strides"])
                == len(self.model_cfg["num_filters"])
            )

            layer_nums = self.model_cfg["layer_nums"]
            layer_strides = self.model_cfg["layer_strides"]
            num_filters = self.model_cfg["num_filters"]
        else:
            layer_nums = layer_strides = num_filters = []

        if "upsample_strides" in self.model_cfg:
            assert len(self.model_cfg["upsample_strides"]) == len(
                self.model_cfg["num_upsample_filter"]
            )

            num_upsample_filters = self.model_cfg["num_upsample_filter"]
            upsample_strides = self.model_cfg["upsample_strides"]

        else:
            upsample_strides = num_upsample_filters = []

        self.resnet = ResNetModified(BasicBlock, layer_nums, layer_strides, num_filters)

        num_levels = len(layer_nums)
        self.num_levels = len(layer_nums)
        self.deblocks = nn.ModuleList()

        for idx in range(num_levels):
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(
                        nn.Sequential(
                            nn.ConvTranspose2d(
                                num_filters[idx],
                                num_upsample_filters[idx],
                                upsample_strides[idx],
                                stride=upsample_strides[idx],
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_upsample_filters[idx], eps=1e-3, momentum=0.01
                            ),
                            nn.ReLU(),
                        )
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_filters[idx],
                                num_upsample_filters[idx],
                                stride,
                                stride=stride,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_upsample_filters[idx], eps=1e-3, momentum=0.01
                            ),
                            nn.ReLU(),
                        )
                    )

        c_in = sum(num_upsample_filters)
        self.deblocks.append(
            nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in // 6, 2, 2, bias=False),
                nn.BatchNorm2d(c_in // 6, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
        )

        self.num_bev_features = c_in

    def forward(self, spatial_features):
        x = self.resnet(spatial_features)
        ups = []

        for i in range(self.num_levels):
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x[i]))
            else:
                ups.append(x[i])

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)

        return x


class ReduceInfTC(nn.Module):
    def __init__(self, channel, mode="DFF"):
        super(ReduceInfTC, self).__init__()
        self.conv1_2 = nn.Conv2d(
            channel // 2, channel // 4, kernel_size=3, stride=2, padding=0
        )
        self.bn1_2 = nn.BatchNorm2d(channel // 4, track_running_stats=True)
        self.conv1_3 = nn.Conv2d(
            channel // 4, channel // 8, kernel_size=3, stride=2, padding=0
        )
        self.bn1_3 = nn.BatchNorm2d(channel // 8, track_running_stats=True)
        self.conv1_4 = nn.Conv2d(
            channel // 8, channel // 64, kernel_size=3, stride=2, padding=1
        )
        self.bn1_4 = nn.BatchNorm2d(channel // 64, track_running_stats=True)

        self.deconv2_1 = nn.ConvTranspose2d(
            channel // 64, channel // 8, kernel_size=3, stride=2, padding=1
        )
        self.bn2_1 = nn.BatchNorm2d(channel // 8, track_running_stats=True)
        self.deconv2_2 = nn.ConvTranspose2d(
            channel // 8, channel // 4, kernel_size=3, stride=2, padding=0
        )
        self.bn2_2 = nn.BatchNorm2d(channel // 4, track_running_stats=True)
        self.deconv2_3 = nn.ConvTranspose2d(
            channel // 4,
            channel // 2,
            kernel_size=3,
            stride=2,
            padding=0,
            output_padding=1,
        )
        self.bn2_3 = nn.BatchNorm2d(channel // 2, track_running_stats=True)

        self.mode = mode

        if self.mode == "FFNet":
            self.conv_flow = nn.Conv2d(
                channel // 2, channel // 2, kernel_size=3, stride=1, padding=1
            )
            self.conv_uncertain = nn.Conv2d(
                channel // 2, channel // 2, kernel_size=3, stride=1, padding=1
            )
        elif self.mode == "DFF":
            self.conv_flow = nn.Conv2d(
                channel // 2, 2, kernel_size=3, stride=1, padding=1
            )
            self.conv_scale = nn.Conv2d(
                channel // 2, 1, kernel_size=1, stride=1, padding=0, bias=False
            )
            torch.nn.init.zeros_(self.conv_scale.weight)

    def forward(self, x):
        outputsize = x.shape
        out = F.relu(self.bn1_2(self.conv1_2(x)))
        out = F.relu(self.bn1_3(self.conv1_3(out)))
        out = F.relu(self.bn1_4(self.conv1_4(out)))

        out = F.relu(self.bn2_1(self.deconv2_1(out)))
        out = F.relu(self.bn2_2(self.deconv2_2(out)))
        x_1 = F.relu(self.bn2_3(self.deconv2_3(out)))

        if self.mode == "FFNet":
            flow = self.conv_flow(x_1)
            uncertainty = self.conv_uncertain(x_1)
            uncertainty = torch.sigmoid(uncertainty)
            return flow, uncertainty
        elif self.mode == "DFF":
            offset = self.conv_flow(x_1)
            scale = self.conv_scale(x_1)
            scale = scale + torch.ones_like(scale)
            return offset, scale


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownsampleConv(nn.Module):
    def __init__(self, config):
        super(DownsampleConv, self).__init__()
        self.layers = nn.ModuleList([])
        input_dim = config["input_dim"]

        for ksize, dim, stride, padding in zip(
            config["kernal_size"], config["dim"], config["stride"], config["padding"]
        ):
            self.layers.append(
                DoubleConv(
                    input_dim, dim, kernel_size=ksize, stride=stride, padding=padding
                )
            )
            input_dim = dim

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x


class FlowGenerator(nn.Module):
    def __init__(self, args):
        super(FlowGenerator, self).__init__()
        self.channel = 64
        self.backbone = ResNetBEVBackbone(args["base_bev_backbone"], self.channel)
        self.pre_encoder = ReduceInfTC(128)
        self.mse_loss = nn.MSELoss()

    def get_grid(self, flow):
        m, n = flow.shape[-2:]
        shifts_x = torch.arange(0, n, 1, dtype=torch.float32, device=flow.device)
        shifts_y = torch.arange(0, m, 1, dtype=torch.float32, device=flow.device)
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x)

        grid_dst = torch.stack((shifts_x, shifts_y)).unsqueeze(0)
        workspace = (
            torch.tensor([(n - 1) / 2, (m - 1) / 2]).view(1, 2, 1, 1).to(flow.device)
        )

        flow_grid = ((flow + grid_dst) / workspace - 1).permute(0, 2, 3, 1)

        return flow_grid

    def flow_warp_feats(self, feats, flow):
        flow_grid = self.get_grid(flow)
        warped_feats = F.grid_sample(
            feats, flow_grid, mode="bilinear", padding_mode="border"
        )

        return warped_feats

    def forward(self, feat_list):
        total_loss = torch.zeros(1).to(feat_list[0][0].device)
        flow_offset_list = []
        final_list = []
        for bs in range(len(feat_list)):
            time_list = feat_list[bs]
            time_list.reverse()

            fusion_feature = torch.cat(time_list, dim=1)
            ego_feat = fusion_feature[:1][:, self.channel :, :, :]
            if fusion_feature.shape[0] == 1:
                final_list.append(ego_feat)
                continue

            colla_feat = fusion_feature[1:]
            colla_fusion = self.backbone(colla_feat)

            feat_source = colla_feat[:, : self.channel, :, :]
            feat_target = colla_feat[:, self.channel :, :, :]

            offset, scale = self.pre_encoder(colla_fusion)

            feat_estimate_target = self.flow_warp_feats(feat_source, offset)
            feat_estimate_target = feat_estimate_target * scale
            final_list.append(torch.cat([ego_feat, feat_estimate_target], dim=0))
            similarity = torch.cosine_similarity(
                torch.flatten(feat_target, start_dim=1, end_dim=3),
                torch.flatten(feat_estimate_target, start_dim=1, end_dim=3),
                dim=1,
            )
            label = torch.ones(1, requires_grad=False).to(device=feat_target.device)
            loss = self.mse_loss(similarity, label)
            total_loss += loss

        final_feat = torch.cat(final_list, dim=0)
        return final_feat, total_loss
