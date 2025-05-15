import numpy as np
import torch
import torch.nn as nn


class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()

        self.smooth = False
        self.thre = args["thre"]
        self.compressed_dim = args["compressed_dim"]
        if "gaussian_smooth" in args:
            self.smooth = True
            kernel_size = args["gaussian_smooth"]["k_size"]
            c_sigma = args["gaussian_smooth"]["c_sigma"]
            self.gaussian_filter = nn.Conv2d(
                1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2
            )
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False

    def init_gaussian_filter(self, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
            g = (
                1
                / (2 * np.pi * sigma)
                * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            )
            return g

        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        self.gaussian_filter.weight.data = (
            torch.Tensor(gaussian_kernel)
            .to(self.gaussian_filter.weight.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.gaussian_filter.bias.data.zero_()

    def forward(self, psm):
        B = len(psm)
        _, _, H, W = psm[0].shape

        private_confidence_maps = []
        private_communication_masks = []
        communication_rates = []

        for b in range(B):
            ori_private_communication_maps = psm[b].sigmoid().max(dim=1)[0].unsqueeze(1)

            if self.smooth:
                private_communication_maps = self.gaussian_filter(
                    ori_private_communication_maps
                )
            else:
                private_communication_maps = ori_private_communication_maps
            private_confidence_maps.append(private_communication_maps)

            ones_mask = torch.ones_like(private_communication_maps).to(
                private_communication_maps.device
            )
            zeros_mask = torch.zeros_like(private_communication_maps).to(
                private_communication_maps
            )

            private_mask = torch.where(
                private_communication_maps > self.thre, ones_mask, zeros_mask
            )
            cav_num = private_mask.shape[0]
            private_rate = private_mask[1:].sum() / ((cav_num - 1) * H * W)

            private_mask_nodiag = private_mask.clone()
            ones_mask = torch.ones_like(private_mask).to(private_mask.device)
            private_mask_nodiag[::2] = ones_mask[::2]
            private_communication_masks.append(private_mask_nodiag)
            communication_rates.append(private_rate)

        communication_rates = sum(communication_rates) / B
        private_mask = torch.cat(private_communication_masks, dim=0)

        return private_mask, communication_rates, private_confidence_maps
