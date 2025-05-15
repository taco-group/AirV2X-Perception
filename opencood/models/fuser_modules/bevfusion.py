import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVFusion(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        """
        Args:
            in_channels (int): Number of channels in each input feature map.
            out_channels (int): Desired output channel dimension.
            fusion_type (str): One of ['add', 'concat', 'conv'].
        """
        super(BEVFusion, self).__init__()

        self.fusion_conv = nn.Sequential(
            # as in original BEVFusion paper
            # TODO(YH): incorperate HanLab implementation
            nn.Conv2d(
                in_channels * 2, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_tensor_list):
        x = torch.cat(input_tensor_list, dim=1)  # [B, n*C, H, W]
        fused = self.fusion_conv(x)  # [B, C, H, W]
        return fused
