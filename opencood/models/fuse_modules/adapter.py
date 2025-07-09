# -*- coding: utf-8 -*-
# Author: Xiangbo Gao <xiangbogaobarry@gmail.com>
# License: MIT License

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from opencood.models.sub_modules.feature_alignnet_modules import (
    SCAligner,
    Res1x1Aligner,
    Res3x3Aligner,
    Res3x3Aligner,
    CBAM,
    ConvNeXt,
    # AttBlock,
    FANet,
    SDTAAgliner,
)
# from opencood.models.sub_modules.deformable_attention import (
#     deformable_attn_pytorch,
#     LearnedPositionalEncoding,
#     constant_init,
#     xavier_init,
# )
# from opencood.models.sub_modules.deformable_attention import (
#     compute_mixed_cis,
#     compute_axial_cis,
#     init_2d_freqs,
#     init_t_xy,
#     apply_rotary_emb,
#     init_random_2d_freqs,
# )
# from positional_encodings.torch_encodings import PositionalEncoding2D, PositionalEncodingPermute2D, Summer

import warnings
import numpy as np


class BaseAdapter(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        in_cav_lidar_range,
        out_cav_lidar_range,
        in_feature_shape,
        out_feature_shape,
        **kwargs,
    ):
        # TODO: For now, we ignore the z axis, not sure if we need to consider it.
        # We also assume that the agent is always at the center of the lidar range
        super(BaseAdapter, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_cav_lidar_range = in_cav_lidar_range
        self.out_cav_lidar_range = out_cav_lidar_range
        self.in_feature_shape = in_feature_shape
        self.out_feature_shape = out_feature_shape

        in_range_lidar = np.array(
            [in_cav_lidar_range[3] - in_cav_lidar_range[0], in_cav_lidar_range[4] - in_cav_lidar_range[1]]
        )
        out_range_lidar = np.array(
            [out_cav_lidar_range[3] - out_cav_lidar_range[0], out_cav_lidar_range[4] - out_cav_lidar_range[1]]
        )
        self.ratio = out_range_lidar / in_range_lidar

        in_range_feature = np.array([in_feature_shape[1], in_feature_shape[0]])
        out_range_feature = np.array([out_feature_shape[1], out_feature_shape[0]])

        in_ratio = in_range_feature / in_range_lidar
        out_ratio = out_range_feature / out_range_lidar
        self.feat_ratio = out_ratio / in_ratio

        left_new = in_cav_lidar_range[0] * in_ratio[0] * self.feat_ratio[0]
        right_new = in_cav_lidar_range[3] * in_ratio[0] * self.feat_ratio[0]
        top_new = in_cav_lidar_range[1] * in_ratio[1] * self.feat_ratio[1]
        bottom_new = in_cav_lidar_range[4] * in_ratio[1] * self.feat_ratio[1]

        left_target = out_cav_lidar_range[0] * out_ratio[0]
        right_target = out_cav_lidar_range[3] * out_ratio[0]
        top_target = out_cav_lidar_range[1] * out_ratio[1]
        bottom_target = out_cav_lidar_range[4] * out_ratio[1]

        left_diff = left_new - left_target
        right_diff = right_target - right_new
        top_diff = top_new - top_target
        bottom_diff = bottom_target - bottom_new

        self.pad = nn.ZeroPad2d((round(left_diff), round(right_diff), round(top_diff), round(bottom_diff)))

        self.init_adapter()

    def init_adapter(self):
        raise NotImplementedError

    def forward(self, ego_feature, protocol_feature):
        raise NotImplementedError


class AdapterIdentity(BaseAdapter):
    def __init__(self, **kwargs):
        super(AdapterIdentity, self).__init__(**kwargs)

    def init_adapter(self):
        self.resize = nn.Upsample(
            scale_factor=(self.out_channels / self.in_channels, self.feat_ratio[0], self.feat_ratio[1]),
            mode="trilinear",
        )

    def forward(self, ego_feature):
        ego_feature = self.resize(ego_feature.unsqueeze(1))
        ego_feature = ego_feature.squeeze(1)
        return ego_feature



class AdapterConvNext(BaseAdapter):
    def __init__(self, submodule_args, **kwargs):
        self.submodule_args = submodule_args
        super(AdapterConvNext, self).__init__(**kwargs)

    def init_adapter(self):
        
        self.resize = nn.Upsample(scale_factor=(self.feat_ratio[0], self.feat_ratio[1]), mode="bilinear")
        hiddle_channel = self.submodule_args.get("dim", 64)
        self.channel_convert1 = nn.Conv2d(self.in_channels, hiddle_channel, kernel_size=1)
        self.conv = ConvNeXt(self.submodule_args)
        self.channel_convert2 = nn.Conv2d(hiddle_channel, self.out_channels, kernel_size=1)
        self.smoothing = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, ego_feature):
        ego_feature = ego_feature * self.submodule_args.get("early_scale", 1.0)
        if not self.submodule_args.get("late_upsample", False):
            ego_feature = self.resize(ego_feature)
        ego_feature = self.channel_convert1(ego_feature)
        protocol_feature = self.conv(ego_feature)
        protocol_feature = self.channel_convert2(protocol_feature)
        if self.submodule_args.get("late_upsample", False):
            protocol_feature = self.resize(protocol_feature)

        return protocol_feature



class AdapterAtt(BaseAdapter):
    def __init__(self, submodule_args, **kwargs):
        self.submodule_args = submodule_args
        super(AdapterAtt, self).__init__(**kwargs)

    def init_adapter(self):
        
        self.resize = nn.Upsample(scale_factor=(self.feat_ratio[0], self.feat_ratio[1]), mode="bilinear")
        hiddle_channel = self.submodule_args.get("dim", 64)
        self.channel_convert1 = nn.Conv2d(self.in_channels, hiddle_channel, kernel_size=1)
        self.channel_convert2 = nn.Conv2d(hiddle_channel, self.out_channels, kernel_size=1)
        self.smoothing = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        
        self.patch_size = self.submodule_args.get("patch_size", 16)
        stride = self.patch_size
        if self.submodule_args.get("late_upsample", False):
            H = self.in_feature_shape[0]
            W = self.in_feature_shape[1]
        else:
            H = self.out_feature_shape[0]
            W = self.out_feature_shape[1]
        num_heads = self.submodule_args.get("num_heads", 4)
        depth = self.submodule_args.get("depth", 3)
        
        self.patch_embed = nn.Conv2d(hiddle_channel, hiddle_channel, kernel_size=self.patch_size, stride=stride)
        self.pos_embed = nn.Parameter(torch.zeros(1, H // self.patch_size * W // self.patch_size, hiddle_channel))
        self.blocks = nn.ModuleList([
            AttBlock(hiddle_channel, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hiddle_channel)
        

    def forward(self, ego_feature):
        ego_feature = ego_feature * self.submodule_args.get("early_scale", 1.0)
        if not self.submodule_args.get("late_upsample", False):
            ego_feature = self.resize(ego_feature)
        ego_feature = self.channel_convert1(ego_feature)
        # protocol_feature = self.conv(ego_feature)
        
        
        B, C, H, W = ego_feature.shape
        # Patch embedding
        x = self.patch_embed(ego_feature)  # shape: (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # shape: (B, num_patches, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for blk in self.blocks: x = blk(x)
        
        x = self.norm(x)
        
        # Reshape back to image-like tensor
        # import pdb; pdb.set_trace()
        x = x.transpose(1, 2).reshape(B, C, H // self.patch_size, W // self.patch_size)
        protocol_feature = nn.functional.interpolate(x, scale_factor=self.patch_size, mode='bilinear', align_corners=False)
        
        protocol_feature = self.channel_convert2(protocol_feature)
        if self.submodule_args.get("late_upsample", False):
            protocol_feature = self.resize(protocol_feature)
        # protocol_feature = apply_gaussian_smoothing(protocol_feature, 5, 1.0)
        # protocol_feature = self.pad(protocol_feature)

        return protocol_feature







class AdapterConv(BaseAdapter):
    """
    Conv Adapter
    Upsample the protocol feature to the same size of ego feature with a convolutional layer.

    """

    def __init__(self, **kwargs):
        super(AdapterConv, self).__init__(**kwargs)

    def init_adapter(self):
        self.resize = nn.Upsample(scale_factor=(self.feat_ratio[0], self.feat_ratio[1]), mode="bilinear")
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        nn.init.kaiming_normal_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, ego_feature):

        ego_feature = self.resize(ego_feature)
        protocol_feature = self.conv(ego_feature)

        protocol_feature = self.pad(protocol_feature)

        return protocol_feature


class AdapterFC(BaseAdapter):
    """
    Conv Adapter
    Upsample the protocol feature to the same size of ego feature with a convolutional layer.

    eg:
    in_H_voxel = 128
    in_W_voxel = 128
    in_H = 512
    in_W = 512
    out_H_voxel = 192
    out_W_voxel = 192
    out_H = 1536
    out_W = 1536

    in_factor_H = 512 / 128 = 4
    (out_H_voxel - in_H_voxel) / 2 = (192 - 128) / 2 = 32

    """

    def __init__(self, **kwargs):
        super(AdapterFC, self).__init__(**kwargs)

    def init_adapter(self):
        self.resize = nn.Upsample(scale_factor=(self.feat_ratio[0], self.feat_ratio[1]), mode="bilinear")

        self.weights = nn.Parameter(
            torch.Tensor(self.in_feature_shape[0], self.in_feature_shape[1], self.in_channels, self.out_channels)
        )
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        self.biases = nn.Parameter(torch.zeros(self.in_feature_shape[0], self.in_feature_shape[1], self.out_channels))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.biases, -bound, bound)

    def forward(self, ego_feature):
        ego_feature = self.resize(ego_feature)
        B, C, H, W = ego_feature.shape
        ego_feature = ego_feature.reshape(B, C, H, W)

        # Perform pixel-wise fully connected operation
        protocol_feature = torch.einsum("bchw,hwco->bhwo", ego_feature, self.weights) + self.biases.view(
            H, W, self.out_channels
        )
        protocol_feature = protocol_feature.permute(0, 3, 1, 2)

        protocol_feature = self.pad(protocol_feature)
        # TODO: make sure this is on the correct dimension
        # protocol_feature = protocol_feature[:, :, self.crop_top:self.crop_bottom, self.crop_left:self.crop_right]

        return protocol_feature


class DeformableSpatialAttentionLayer(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        num_heads=8,
        num_points=4,
        dropout=0.1,
        scale_ratio=1.0,
    ):
        super(DeformableSpatialAttentionLayer, self).__init__()
        if out_channel % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, " f"but got {out_channel} and {num_heads}")
        self.dim_per_head = out_channel // num_heads

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        # however, CUDA is not available in this implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(self.dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                "MultiScaleDeformAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
                "However, CUDA is not available in this implementation."
            )

        assert self.dim_per_head % 2 == 0, "embed_dims must be divisible by 2"

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_heads = num_heads
        self.num_points = num_points
        self.dropout = nn.Dropout(dropout)
        self.sampling_offsets = nn.Linear(self.in_channel, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(self.in_channel, num_heads * num_points)
        self.value_proj = nn.Linear(self.in_channel, self.out_channel)
        self.output_proj = nn.Linear(self.out_channel, self.out_channel)
        self.scale_ratio = scale_ratio
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, 1, self.num_points, 1)
        )

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        # TODO: Remove the hard coded half precision
        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        identity=None,
        device="cuda",
        dtype=torch.half,
        spatial_shapes=None,
    ):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                (bs, num_query, embed_dims).
            value (Tensor): The value tensor with shape
                (bs, num_query, embed_dims).
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            spatial_shapes (tuple): Spatial shape of features (h, w).

        Returns:
             Tensor: forwarded results with shape [bs, num_query, embed_dims].
        """

        bs, num_query, embed_dims = query.shape
        h, w = spatial_shapes

        if identity is None:
            identity = query

        if query_pos is not None:
            query = query + query_pos
        value = self.value_proj(value)
        # if key_padding_mask is not None:
        #     value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.reshape(
            bs, num_query, self.num_heads, self.dim_per_head
        )  # bs, num_query, num_heads, embed_dims//num_heads
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, self.num_points, 2
        )  # bs, num_query, num_heads, num_points, 2
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_points
        )  # bs, num_query, num_heads, num_points
        attention_weights = attention_weights.softmax(-1).to(
            dtype
        )  # TODO: attention_weights.softmax(-1) changed attention_weights from half to float

        reference_points = self.get_reference_points(
            h, w, bs=bs, scale_ratio=self.scale_ratio, device=device, dtype=dtype
        )  # bs, num_query, 2
        offset_normalizer = torch.Tensor([w, h]).to(device).to(dtype)
        sampling_locations = reference_points[:, :, None, None, :] + sampling_offsets / offset_normalizer

        output = self.output_proj(deformable_attn_pytorch(value, (h, w), sampling_locations, attention_weights))
        # return self.dropout(output) + identity
        return self.dropout(output) + identity

    def get_reference_points(self, H, W, bs=1, scale_ratio=1.0, device="cuda", dtype=torch.half):
        if type(scale_ratio) != tuple:
            scale_ratio = (scale_ratio, scale_ratio)

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
        )
        # TODO: make sure the x y dimension is correct
        ref_y = ref_y.reshape(-1)[None] / H * scale_ratio[0]
        ref_x = ref_x.reshape(-1)[None] / W * scale_ratio[1]
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1)
        return ref_2d


class AdapterDSA(BaseAdapter):
    """Deformable Spatial Attention."""

    def __init__(
        self,
        in_channels,
        out_channels,
        in_cav_lidar_range,
        out_cav_lidar_range,
        in_feature_shape,
        out_feature_shape,
        submodule_args,
        **kwargs,
    ):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.H, self.W = in_feature_shape
        self.outH = out_feature_shape[0]
        self.outW = out_feature_shape[1]
        self.n_layers = submodule_args.get("n_layers", 8)
        self.num_heads = submodule_args.get("num_heads", 8)
        self.num_points = submodule_args.get("num_points", 4)
        self.dropout = submodule_args.get("dropout", 0.1)
        self.rope_mixed = submodule_args.get("rope_mixed", True)  # TODO: False does not work
        self.rope_theta = submodule_args.get("rope_theta", 10.0)
        self.dim_per_head = out_channels // self.num_heads

        super(AdapterDSA, self).__init__(
            in_channels,
            out_channels,
            in_cav_lidar_range,
            out_cav_lidar_range,
            in_feature_shape,
            out_feature_shape,
            **kwargs,
        )

    def init_adapter(self):
        # self.bev_embedding = nn.Embedding(self.outH * self.outW, self.out_channels)
        self.resize = nn.Upsample(scale_factor=(self.feat_ratio[0], self.feat_ratio[1]), mode="bilinear")
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        # self.positional_encoding = LearnedPositionalEncoding(self.out_channels // 2, self.outH, self.outW)

        if self.rope_mixed:
            self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)

            freqs = []
            for _ in range(self.n_layers):
                freqs.append(
                    init_random_2d_freqs(dim=self.dim_per_head, num_heads=self.num_heads, theta=self.rope_theta)
                )
            freqs = torch.stack(freqs, dim=1).view(2, self.n_layers, -1)
            self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)

            t_x, t_y = init_t_xy(end_x=self.outH, end_y=self.outW)
            # self.register_buffer('freqs_t_x', t_x)
            # self.register_buffer('freqs_t_y', t_y)
            self.freqs_t_x = t_x
            self.freqs_t_y = t_y
        else:
            self.compute_cis = partial(compute_axial_cis, dim=self.dim_per_head, theta=self.rope_theta)

            freqs_cis = self.compute_cis(end_x=self.outH, end_y=self.outW)
            self.freqs_cis = freqs_cis

        self.attention_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.attention_layers.append(
                DeformableSpatialAttentionLayer(
                    self.in_channels, self.out_channels, self.num_heads, self.num_points, self.dropout, self.ratio
                )
            )

    def forward(self, ego_feature):
        B, C, H, W = ego_feature.shape
        key = ego_feature.view(B, C, H * W).transpose(1, 2)  # B, H*W, C
        device, dtype = key.device, key.dtype
        # query = self.bev_embedding.weight.to(device).reshape(1, self.outH*self.outW, self.out_channels).repeat(B, 1, 1) # B, outH*outW, outC
        # query = self.resize(ego_feature).view(B, C, self.outH*self.outW).transpose(1, 2) # B, H*W, C
        query = (
            self.conv(self.resize(ego_feature)).view(B, self.out_channels, self.outH * self.outW).transpose(1, 2)
        )  # B, H*W, C
        # pos_mask = torch.zeros((B, self.outH, self.outW), device=device).to(dtype)
        # query_pos = self.positional_encoding(pos_mask).to(dtype).flatten(2).transpose(1,2) # B, H*W, C

        # ###### Calculate freqs_cis
        if self.rope_mixed:
            if self.freqs_t_x.shape[0] != self.outH * self.W - 1:
                t_x, t_y = init_t_xy(end_x=self.outH, end_y=self.outW)
                t_x, t_y = t_x.to(device), t_y.to(device)
            else:
                t_x, t_y = self.freqs_t_x, self.freqs_t_y
                t_x, t_y = t_x.to(device), t_y.to(device)
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)

        else:
            if self.freqs_cis.shape[0] != self.outH * self.W - 1:
                freqs_cis = self.compute_cis(end_x=self.outH, end_y=self.outW)
            else:
                freqs_cis = self.freqs_cis
            freqs_cis = freqs_cis.to(device)

        output = query.contiguous()
        for layer in range(self.n_layers):

            ###### Apply rotary position embedding
            # Figure out why [1:] is used
            # if self.rope_mixed:
            #     output[:,1:], _ = apply_rotary_emb(output[:,1:], None, freqs_cis=freqs_cis[layer])
            # else:
            #     output[:,1:], _ = apply_rotary_emb(output[:,1:], None, freqs_cis=freqs_cis)
            if self.rope_mixed:
                output, _ = apply_rotary_emb(output, None, freqs_cis=freqs_cis[layer])
            else:
                output, _ = apply_rotary_emb(output, None, freqs_cis=freqs_cis)
            #########

            output = self.attention_layers[layer](
                query=output,
                key=None,  # key is not used
                value=key,
                identity=output,
                # query_pos=query_pos, # query_pos is not used for rotary position embedding
                device=device,
                dtype=dtype,
                spatial_shapes=(self.outH, self.outW),
            )

        output = output.transpose(1, 2).reshape(B, self.out_channels, self.outH, self.outW)
        protocol_feature = self.pad(output)
        return protocol_feature


class AdapterDSA(BaseAdapter):
    """Deformable Spatial Attention."""

    def __init__(
        self,
        in_channels,
        out_channels,
        in_cav_lidar_range,
        out_cav_lidar_range,
        in_feature_shape,
        out_feature_shape,
        submodule_args,
        **kwargs,
    ):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.H, self.W = in_feature_shape
        self.outH = out_feature_shape[0]
        self.outW = out_feature_shape[1]
        self.n_layers = submodule_args.get("n_layers", 8)
        self.num_heads = submodule_args.get("num_heads", 8)
        self.num_points = submodule_args.get("num_points", 4)
        self.dropout = submodule_args.get("dropout", 0.1)
        self.rope_mixed = submodule_args.get("rope_mixed", True)  # TODO: False does not work
        self.rope_theta = submodule_args.get("rope_theta", 10.0)
        self.dim_per_head = out_channels // self.num_heads

        super(AdapterDSA, self).__init__(
            in_channels,
            out_channels,
            in_cav_lidar_range,
            out_cav_lidar_range,
            in_feature_shape,
            out_feature_shape,
            **kwargs,
        )

    def init_adapter(self):
        # self.bev_embedding = nn.Embedding(self.outH * self.outW, self.out_channels)
        self.resize = nn.Upsample(scale_factor=(self.feat_ratio[0], self.feat_ratio[1]), mode="bilinear")
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        # self.positional_encoding = LearnedPositionalEncoding(self.out_channels // 2, self.outH, self.outW)
        self.in_pos_embed_sinusoidal = PositionalEncodingPermute2D(self.in_channels)
        self.in_pos_scale_factor = nn.Parameter(torch.ones(1) / 30.0)
        self.out_pos_embed_sinusoidal = PositionalEncoding2D(self.out_channels)
        self.out_pos_scale_factor = nn.Parameter(torch.ones(1) / 30.0)

        self.attention_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.attention_layers.append(
                DeformableSpatialAttentionLayer(
                    self.in_channels, self.out_channels, self.num_heads, self.num_points, self.dropout, self.ratio
                )
            )

    def forward(self, ego_feature):
        B, C, H, W = ego_feature.shape
        # ego_feature = self.in_pos_embed_sinusoidal(ego_feature)
        ego_feature_embed = self.in_pos_embed_sinusoidal(ego_feature)
        ego_feature = ego_feature_embed * self.in_pos_scale_factor + ego_feature
        key = ego_feature.view(B, C, H * W).transpose(1, 2)  # B, H*W, C
        device, dtype = key.device, key.dtype
        # query = self.bev_embedding.weight.to(device).reshape(1, self.outH*self.outW, self.out_channels).repeat(B, 1, 1) # B, outH*outW, outC
        # query = self.resize(ego_feature).view(B, C, self.outH*self.outW).transpose(1, 2) # B, H*W, C
        query = (
            self.conv(self.resize(ego_feature)).view(B, self.out_channels, self.outH * self.outW).transpose(1, 2)
        )  # B, H*W, C
        # pos_mask = torch.zeros((B, self.outH, self.outW), device=device).to(dtype)
        # query_pos = self.positional_encoding(pos_mask).to(dtype).flatten(2).transpose(1,2) # B, H*W, C

        output = query

        print(self.in_pos_scale_factor, self.out_pos_scale_factor)

        for layer in range(self.n_layers):

            output = output.reshape(B, self.outH, self.outW, self.out_channels)
            output_embed = self.out_pos_embed_sinusoidal(output)
            output = output_embed * self.out_pos_scale_factor + output
            output = output.reshape(B, self.outH * self.outW, self.out_channels)

            output = self.attention_layers[layer](
                query=output,
                key=None,  # key is not used
                value=key,
                identity=output,
                # query_pos=query_pos, # query_pos is not used for rotary position embedding
                device=device,
                dtype=dtype,
                spatial_shapes=(self.outH, self.outW),
            )

        output = output.transpose(1, 2).reshape(B, self.out_channels, self.outH, self.outW)
        protocol_feature = self.pad(output)
        return protocol_feature


# class AdapterDSA(BaseAdapter):
#     """Deformable Spatial Attention."""

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  in_cav_lidar_range,
#                  out_cav_lidar_range,
#                  in_feature_shape,
#                  out_feature_shape,
#                  submodule_args,
#                  **kwargs):

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.H, self.W = in_feature_shape
#         self.outH = out_feature_shape[0]
#         self.outW = out_feature_shape[1]
#         self.n_layers = submodule_args.get('n_layers', 8)
#         self.num_heads = submodule_args.get('num_heads', 8)
#         self.num_points = submodule_args.get('num_points', 4)
#         self.dropout = submodule_args.get('dropout', 0.1)
#         self.rope_mixed = submodule_args.get('rope_mixed', True) # TODO: False does not work
#         self.rope_theta = submodule_args.get('rope_theta', 10.0)
#         self.dim_per_head = out_channels // self.num_heads


#         super(AdapterDSA, self).__init__(in_channels,
#                                           out_channels,
#                                           in_cav_lidar_range,
#                                           out_cav_lidar_range,
#                                           in_feature_shape,
#                                           out_feature_shape,
#                                           **kwargs)


#     def init_adapter(self):
#         self.resize = nn.Upsample(scale_factor=self.ratio, mode='bilinear')
#         self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
#         # self.positional_encoding = LearnedPositionalEncoding(self.out_channels // 2, self.outH, self.outW)


#         self.attention_layers = nn.ModuleList()
#         for _ in range(self.n_layers):
#             self.attention_layers.append(DeformableSpatialAttentionLayer(self.in_channels,
#                                                                         self.out_channels,
#                                                                         self.num_heads,
#                                                                         self.num_points,
#                                                                         self.dropout,
#                                                                         self.ratio))


#     def forward(self, ego_feature):
#         B, C, H, W = ego_feature.shape
#         key = ego_feature.view(B, C, H*W).transpose(1, 2) # B, H*W, C
#         device, dtype = key.device, key.dtype
#         query = self.conv(self.resize(ego_feature)).view(B, self.out_channels, self.outH*self.outW).transpose(1, 2) # B, H*W, C
#         # pos_mask = torch.zeros((B, self.outH, self.outW), device=device).to(dtype)
#         # query_pos = self.positional_encoding(pos_mask).to(dtype).flatten(2).transpose(1,2) # B, H*W, C

#         output = query


#         for layer in range(self.n_layers):

#             output = self.attention_layers[layer](
#                                                 query=output,
#                                                 key=None, # key is not used
#                                                 value=key,
#                                                 identity=output,
#                                                 # query_pos=query_pos, # query_pos is not used for rotary position embedding
#                                                 device=device,
#                                                 dtype=dtype,
#                                                 spatial_shapes=(self.outH, self.outW))

#         output = output.transpose(1, 2).reshape(B, self.out_channels, self.outH, self.outW)
#         protocol_feature = self.pad(output)
#         return protocol_feature


class Adapter(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_name = args["core_method"]

        if model_name == "adapterfc":
            self.adapter = AdapterFC(**args["args"])
        elif model_name == "adapterconv":
            self.adapter = AdapterConv(**args["args"])
        elif model_name == "adapterconvnext":
            self.adapter = AdapterConvNext(**args["args"])
        elif model_name == "adapterdsa":
            self.adapter = AdapterDSA(**args["args"])
        elif model_name == "identity":
            self.adapter = AdapterIdentity(**args["args"])
        elif model_name == "adapteratt":
            self.adapter = AdapterAtt(**args["args"])
        else:
            raise NotImplementedError(f"Adapter {model_name} not implemented")

    def forward(self, x):
        return self.adapter(x)


class Reverter(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_name = args["core_method"]

        if model_name == "adapterfc":
            self.reverter = AdapterFC(**args["args"])
        elif model_name == "adapterconv":
            self.reverter = AdapterConv(**args["args"])
        elif model_name == "adapterconvnext":
            self.reverter = AdapterConvNext(**args["args"])
        elif model_name == "adapterdsa":
            self.reverter = AdapterDSA(**args["args"])
        elif model_name == "identity":
            self.reverter = AdapterIdentity(**args["args"])
        elif model_name == "adapteratt":
            self.reverter = AdapterAtt(**args["args"])
        else:
            raise NotImplementedError(f"Reverter {model_name} not implemented")

    def forward(self, x):
        return self.reverter(x)
