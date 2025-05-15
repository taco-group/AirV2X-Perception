import math
import time

import MultiScaleDeformableAttention as MSDA
import numpy as np
import torch
from einops import rearrange, repeat
from torch import batch_norm, einsum, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn import functional as F
from torch.nn.init import constant_, xavier_uniform_


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            ctx.im2col_step,
        )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = MSDA.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            ctx.im2col_step,
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


class MSDeformAttn(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_head=64,
        n_levels=2,
        n_heads=8,
        n_points=4,
        out_sample_loc=False,
    ):
        super().__init__()

        self.im2col_step = 64
        n_levels = 5
        self.d_model = d_model
        self.d_head = d_head
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.out_sample_loc = out_sample_loc

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_head * n_heads)
        self.output_proj = nn.Linear(d_head * n_heads, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_head)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            ).to(sampling_offsets)

            a = reference_points[:, :, None, :, None, :]
            b = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )

        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )
        output = MSDeformAttnFunction.apply(
            value,
            input_spatial_shapes,
            input_level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )
        output = self.output_proj(output)
        if self.out_sample_loc:
            return output, torch.cat(
                (sampling_locations, attention_weights[:, :, :, :, :, None]), dim=-1
            )
        else:
            return output, None


class DeformableTransformerCrossAttention(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_head=64,
        dropout=0.3,
        n_levels=2,
        n_heads=6,
        n_points=9,
        out_sample_loc=False,
    ):
        super().__init__()

        self.cross_attn = MSDeformAttn(
            d_model, d_head, n_levels, n_heads, n_points, out_sample_loc=out_sample_loc
        )
        self.dropout = nn.Dropout(dropout)
        self.out_sample_loc = out_sample_loc

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        src,
        query_pos=None,
        reference_points=None,
        src_spatial_shapes=None,
        level_start_index=None,
        src_padding_mask=None,
    ):
        tgt2, sampling_locations = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask,
        )
        tgt = self.dropout(tgt2)

        if self.out_sample_loc:
            return tgt, sampling_locations
        else:
            return tgt


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class GELU(nn.Module):
    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
            )
        )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm_CA(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, y, **kwargs):
        return self.fn(self.norm(x), self.norm(y), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, out_attention=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.out_attention = out_attention

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = self.attend(dots)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        if self.out_attention:
            return self.to_out(out), attn
        else:
            return self.to_out(out)


class Deform_Transformer(nn.Module):
    def __init__(
        self,
        dim,
        levels=2,
        depth=2,
        heads=4,
        dim_head=32,
        mlp_dim=256,
        dropout=0.0,
        out_attention=False,
        n_points=9,
    ):
        super().__init__()
        self.out_attention = out_attention
        self.layers = nn.ModuleList([])
        self.depth = depth
        self.levels = levels
        self.n_points = n_points

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                out_attention=self.out_attention,
                            ),
                        ),
                        PreNorm_CA(
                            dim,
                            DeformableTransformerCrossAttention(
                                dim,
                                dim_head,
                                n_levels=levels,
                                n_heads=heads,
                                dropout=dropout,
                                n_points=n_points,
                                out_sample_loc=self.out_attention,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(
        self, x, pos_embedding, src, src_spatial_shapes, level_start_index, center_pos
    ):
        if self.out_attention:
            out_cross_attention_list = []
            out_self_attention_list = []
        if pos_embedding is not None:
            center_pos_embedding = pos_embedding(center_pos)
        reference_points = center_pos[:, :, None, :].repeat(1, 1, self.levels, 1)
        for i, (self_attn, cross_attn, ff) in enumerate(self.layers):
            if self.out_attention:
                if center_pos_embedding is not None:
                    x_att, self_att = self_attn(x + center_pos_embedding)
                    x = x_att + x
                    x_att, cross_att = cross_attn(
                        x,
                        src,
                        query_pos=center_pos_embedding,
                        reference_points=reference_points,
                        src_spatial_shapes=src_spatial_shapes,
                        level_start_index=level_start_index,
                    )
                else:
                    x_att, self_att = self_attn(x)
                    x = x_att + x
                    x_att, cross_att = cross_attn(
                        x,
                        src,
                        query_pos=None,
                        reference_points=reference_points,
                        src_spatial_shapes=src_spatial_shapes,
                        level_start_index=level_start_index,
                    )
                out_cross_attention_list.append(cross_att)
            else:
                if center_pos_embedding is not None:
                    x_att = cross_attn(
                        x,
                        src,
                        query_pos=center_pos_embedding,
                        reference_points=reference_points,
                        src_spatial_shapes=src_spatial_shapes,
                        level_start_index=level_start_index,
                    )
                else:
                    x_att = cross_attn(
                        x,
                        src,
                        query_pos=None,
                        reference_points=reference_points,
                        src_spatial_shapes=src_spatial_shapes,
                        level_start_index=level_start_index,
                    )

            x = x_att + x
            x = ff(x) + x

        out_dict = {"ct_feat": x}
        if self.out_attention:
            out_dict.update(
                {"out_attention": torch.stack(out_cross_attention_list, dim=2)}
            )
        return out_dict


class RPN_transformer_deformable_mtf_singlescale(nn.Module):
    def __init__(self, channel):
        super(RPN_transformer_deformable_mtf_singlescale, self).__init__()
        self.channels = channel
        self.depth = 1
        self.heads = 8
        self.agent_num = 5
        self.dim_head = 64
        self.mlp_dim = 256
        self.dp_rate = 0.3
        self.out_att = False
        self.n_points = 9
        self.obj_num = 500
        self.thres = 0.001

        self.transformer_layer = Deform_Transformer(
            self.channels,
            depth=self.depth,
            heads=self.heads,
            levels=self.agent_num,
            dim_head=self.dim_head,
            mlp_dim=self.mlp_dim,
            dropout=self.dp_rate,
            out_attention=self.out_att,
            n_points=self.n_points,
        )
        self.pos_embedding = nn.Linear(2, self.channels)
        self.scale = [1, 0.5, 0.25]

    def forward(self, x, psm, level):
        cav_num, C, H, W = x.shape
        x_ego = x[0:1, :]
        x_agent = x[1:, :]
        if level > 0:
            psm = F.interpolate(psm, scale_factor=self.scale[level])
        psm_sum = sum([f.unsqueeze(0) for f in psm])
        scores, labels = torch.max(psm_sum.reshape(1, 1, H * W), dim=1)

        sort = scores.sort(1, descending=True)
        score_value = sort[0]
        order = sort[1]
        obj_num = len(torch.where(score_value[0] > self.thres)[0])
        obj_num = max(obj_num, self.obj_num)

        order = order[:, :obj_num]
        batch_id = torch.from_numpy(np.indices((1, obj_num))[0]).to(labels)
        re_order = torch.cat([order for _ in range(C)], dim=0)

        scores = torch.gather(scores, 1, order)
        labels = torch.gather(labels, 1, order)

        temp = x_ego.reshape(1, -1, H * W).transpose(2, 1).contiguous()
        ct_feat = (
            x_ego.reshape(1, -1, H * W).transpose(2, 1).contiguous()[batch_id, order]
        )

        y_coor = order // W
        x_coor = order - y_coor * W
        y_coor, x_coor = y_coor.to(ct_feat), x_coor.to(ct_feat)
        y_coor, x_coor = y_coor / H, x_coor / W
        pos_features = torch.stack([x_coor, y_coor], dim=2)

        src_list = [x_ego.reshape(1, -1, H * W).transpose(2, 1).contiguous()]
        for num in range(x_agent.shape[0]):
            src_list.append(
                x_agent[num].reshape(1, -1, (H * W)).transpose(2, 1).contiguous()
            )
        src = torch.cat(src_list, dim=1)
        spatial_list = [(H, W), (H, W), (H, W), (H, W), (H, W)]
        spatial_shapes = torch.as_tensor(
            spatial_list, dtype=torch.long, device=ct_feat.device
        )
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )

        transformer_out = self.transformer_layer(
            ct_feat,
            self.pos_embedding,
            src,
            spatial_shapes,
            level_start_index,
            center_pos=pos_features,
        )

        ct_feat = transformer_out["ct_feat"].transpose(2, 1).contiguous()

        ct_feat = ct_feat[0]
        x_ego = x_ego.reshape(C, -1)
        x_ego.scatter_(1, re_order, ct_feat)
        out = x_ego.reshape(C, H, W)

        return out
