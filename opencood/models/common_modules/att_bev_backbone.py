import numpy as np
import torch
import torch.nn as nn

from opencood.models.common_modules.auto_encoder import AutoEncoder
from opencood.models.common_modules.self_attn import AttFusion


class AttBEVBackbone_MGF(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.compress = False

        if "compression" in model_cfg and model_cfg["compression"] > 0:
            self.compress = True
            self.compress_layer = model_cfg["compression"]
        self.Conv_LG = nn.ModuleList()

        self.Conv_RG = nn.ModuleList()
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

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]

        self.blocks = nn.ModuleList()
        self.fuse_modules = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        if self.compress:
            self.compression_modules = nn.ModuleList()

        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx],
                    num_filters[idx],
                    kernel_size=3,
                    stride=layer_strides[idx],
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]

            fuse_network = AttFusion(num_filters[idx])
            self.fuse_modules.append(fuse_network)
            if self.compress and self.compress_layer - idx > 0:
                self.compression_modules.append(
                    AutoEncoder(num_filters[idx], self.compress_layer - idx)
                )

            for k in range(layer_nums[idx]):
                cur_layers.extend(
                    [
                        nn.Conv2d(
                            num_filters[idx],
                            num_filters[idx],
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    ]
                )

            self.blocks.append(nn.Sequential(*cur_layers))
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
        c_in_list = [input_channels // 2, *num_filters[:-1]]
        self.L_blocks = nn.ModuleList()
        self.L_deblocks = nn.ModuleList()
        self.R_blocks = nn.ModuleList()
        self.R_deblocks = nn.ModuleList()
        for idx in range(num_levels):
            self.Conv_RG.append(
                nn.Sequential(
                    nn.Conv2d(
                        num_filters[idx],
                        num_filters[idx],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.Sigmoid(),
                )
            )
            self.Conv_LG.append(
                nn.Sequential(
                    nn.Conv2d(
                        num_filters[idx],
                        num_filters[idx],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.Sigmoid(),
                )
            )
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx],
                    num_filters[idx],
                    kernel_size=3,
                    stride=layer_strides[idx],
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend(
                    [
                        nn.Conv2d(
                            num_filters[idx],
                            num_filters[idx],
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    ]
                )
            self.L_blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or (
                    stride == 1
                    and not self.model_cfg.get("USE_CONV_FOR_NO_STRIDE", False)
                ):
                    self.L_deblocks.append(
                        nn.Sequential(
                            nn.ConvTranspose2d(
                                num_filters[idx],
                                num_upsample_filters[idx] // 2,
                                upsample_strides[idx],
                                stride=upsample_strides[idx],
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_upsample_filters[idx] // 2, eps=1e-3, momentum=0.01
                            ),
                            nn.ReLU(),
                        )
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.L_deblocks.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_filters[idx],
                                num_upsample_filters[idx] // 2,
                                stride,
                                stride=stride,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_upsample_filters[idx] // 2, eps=1e-3, momentum=0.01
                            ),
                            nn.ReLU(),
                        )
                    )
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx],
                    num_filters[idx],
                    kernel_size=3,
                    stride=layer_strides[idx],
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend(
                    [
                        nn.Conv2d(
                            num_filters[idx],
                            num_filters[idx],
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    ]
                )
            self.R_blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or (
                    stride == 1
                    and not self.model_cfg.get("USE_CONV_FOR_NO_STRIDE", False)
                ):
                    self.R_deblocks.append(
                        nn.Sequential(
                            nn.ConvTranspose2d(
                                num_filters[idx],
                                num_upsample_filters[idx] // 2,
                                upsample_strides[idx],
                                stride=upsample_strides[idx],
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_upsample_filters[idx] // 2, eps=1e-3, momentum=0.01
                            ),
                            nn.ReLU(),
                        )
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.R_deblocks.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_filters[idx],
                                num_upsample_filters[idx] // 2,
                                stride,
                                stride=stride,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_upsample_filters[idx] // 2, eps=1e-3, momentum=0.01
                            ),
                            nn.ReLU(),
                        )
                    )

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        c_in,
                        c_in,
                        upsample_strides[-1],
                        stride=upsample_strides[-1],
                        bias=False,
                    ),
                    nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                )
            )
        self.cnt = 0
        self.num_bev_features = c_in * 2

    def forward(self, data_dict):
        record_len = data_dict["record_len"]
        lidar_x = data_dict["lidar_spatial_features"]
        radar_x = data_dict["radar_spatial_features"]
        ups = []
        ret_dict = {}
        spatial_features = torch.cat([lidar_x, radar_x], dim=1)

        x = spatial_features

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            lidar_x = self.L_blocks[i](lidar_x)
            radar_x = self.R_blocks[i](radar_x)
            lidar_x = self.Conv_LG[i](lidar_x) * lidar_x
            radar_x = self.Conv_RG[i](radar_x) * radar_x
            if self.compress and i < len(self.compression_modules):
                x = self.compression_modules[i](x)
            x_fuse = self.fuse_modules[i](x, record_len)
            lidar_x_fuse = self.fuse_modules[i](lidar_x, record_len)
            radar_x_fuse = self.fuse_modules[i](radar_x, record_len)
            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict["spatial_features_%dx" % stride] = x

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x_fuse))
                ups.append(self.L_deblocks[i](lidar_x_fuse))
                ups.append(self.R_deblocks[i](radar_x_fuse))
            else:
                ups.append(x_fuse)
                ups.append(lidar_x_fuse)
                ups.append(radar_x_fuse)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict["spatial_features_2d"] = x

        return data_dict


class AttBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.compress = False

        if "compression" in model_cfg and model_cfg["compression"] > 0:
            self.compress = True
            self.compress_layer = model_cfg["compression"]

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

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]

        self.blocks = nn.ModuleList()
        self.fuse_modules = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        if self.compress:
            self.compression_modules = nn.ModuleList()

        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx],
                    num_filters[idx],
                    kernel_size=3,
                    stride=layer_strides[idx],
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]

            fuse_network = AttFusion(num_filters[idx])
            self.fuse_modules.append(fuse_network)
            if self.compress and self.compress_layer - idx > 0:
                self.compression_modules.append(
                    AutoEncoder(num_filters[idx], self.compress_layer - idx)
                )

            for k in range(layer_nums[idx]):
                cur_layers.extend(
                    [
                        nn.Conv2d(
                            num_filters[idx],
                            num_filters[idx],
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    ]
                )

            self.blocks.append(nn.Sequential(*cur_layers))
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
        if len(upsample_strides) > num_levels:
            self.deblocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        c_in,
                        c_in,
                        upsample_strides[-1],
                        stride=upsample_strides[-1],
                        bias=False,
                    ),
                    nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                )
            )

        self.num_bev_features = c_in

    def forward(self, data_dict):
        spatial_features = data_dict["spatial_features"]
        record_len = data_dict["record_len"]

        ups = []
        ret_dict = {}
        x = spatial_features

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if self.compress and i < len(self.compression_modules):
                x = self.compression_modules[i](x)
            x_fuse = self.fuse_modules[i](x, record_len)
            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict["spatial_features_%dx" % stride] = x

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x_fuse))
            else:
                ups.append(x_fuse)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict["spatial_features_2d"] = x

        return data_dict
