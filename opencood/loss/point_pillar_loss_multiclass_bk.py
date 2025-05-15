# -*- coding: utf-8 -*-
# Author: OpenPCDet, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """

    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n**2 / beta, n - 0.5 * beta)

        return loss

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None
    ):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert (
                weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            )
            loss = loss * weights.unsqueeze(-1)

        return loss


class PointPillarLossMultiClass(nn.Module):
    def __init__(self, args):
        super(PointPillarLossMultiClass, self).__init__()
        self.reg_loss_func = WeightedSmoothL1Loss()
        self.alpha = 0.25
        self.gamma = 2.0

        self.cls_weight = args["cls_weight"]
        self.reg_coe = args["reg"]
        self.flow_weight = args["flow_weight"] if "flow_weight" in args else 1.0
        self.loss_dict = {}
        self.use_dir = False
        self.obj_pos_weight = args["obj_pos_weight"]
        self.obj_neg_weight = args["obj_neg_weight"]

        self.cls_num = args["num_class"]

    def forward(self, output_dict, target_dict, prefix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict

        cls_label -> one_hot label
        """
        rm = output_dict["rm{}".format(prefix)]  # [B, #anchor*7, H, 1W]
        psm = output_dict["psm{}".format(prefix)]  # [B, #anchor*#class, H, W]
        obj = output_dict["obj{}".format(prefix)]  # [B, #anchor, H, W]
        targets = target_dict["targets"]

        cls_preds = psm.permute(0, 2, 3, 1).contiguous()  # N, C, H, W -> N, H, W, C
        obj_preds = obj.permute(0, 2, 3, 1).contiguous()  # (B, H, W, A)

        cls_labels = target_dict["class_ids"]  # [B, H, W, A] 0 = background
        pos_mask = target_dict["pos_equal_one"] > 0  # [B, H, W, A]
        num_pos = pos_mask.sum()

        if num_pos > 0:
            B, H, W, AC = cls_preds.shape
            C = self.cls_num
            A = AC // C
            cls_preds = cls_preds.view(B, H, W, A, C)
            # NOTE(YH): remove BG
            cls_labels_fg = cls_labels[pos_mask] - 1
            cls_preds_fg = cls_preds[pos_mask]
            one_hot_targets = F.one_hot(cls_labels_fg.long(), num_classes=C).float()

            pred_sigmoid = torch.sigmoid(cls_preds_fg)
            alpha_weight = one_hot_targets * self.alpha + (1.0 - one_hot_targets) * (
                1.0 - self.alpha
            )
            pt = (
                one_hot_targets * (1 - pred_sigmoid)
                + (1.0 - one_hot_targets) * pred_sigmoid
            )
            focal_weight = alpha_weight * torch.pow(pt, self.gamma)
            bce_loss = F.binary_cross_entropy_with_logits(
                cls_preds_fg, one_hot_targets, reduction="none"
            )
            cls_loss = (focal_weight * bce_loss).sum() / num_pos.clamp(min=1.0)
        else:
            cls_loss = torch.tensor(0.0, dtype=cls_preds.dtype, device=cls_preds.device)

        conf_loss = cls_loss * self.cls_weight

        # reg
        rm = rm.permute(0, 2, 3, 1).contiguous()
        rm = rm.view(rm.shape[0], -1, 7)
        targets = targets.view(targets.shape[0], -1, 7)
        box_preds_sin, reg_targets_sin = self.add_sin_difference(rm, targets)

        reg_weights = pos_mask.view(rm.shape[0], -1).float()
        pos_normalizer = reg_weights.sum(1, keepdim=True)
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, reg_weights)
        reg_loss = loc_loss_src.sum() / rm.shape[0]
        reg_loss *= self.reg_coe

        # obj
        pos_mask = pos_mask.float()
        neg_mask = 1.0 - pos_mask

        obj_preds_sigmoid = torch.sigmoid(obj_preds)

        pos_loss = -pos_mask * torch.log(obj_preds_sigmoid + 1e-6)
        neg_loss = -neg_mask * torch.log(1.0 - obj_preds_sigmoid + 1e-6)

        num_pos = pos_mask.sum()
        num_neg = neg_mask.sum()

        pos_loss = pos_loss.sum() / torch.clamp(num_pos, min=1.0)
        neg_loss = neg_loss.sum() / torch.clamp(num_neg, min=1.0)
        obj_loss = (
            self.obj_pos_weight * pos_loss + self.obj_neg_weight * neg_loss
        )  # or weight according to your preference

        # implementation 3
        # obj_alpha = 0.75
        # obj_preds_sigmoid = torch.sigmoid(obj_preds)
        # pt = obj_preds_sigmoid * pos_mask.float() + (1 - obj_preds_sigmoid) * (1 - pos_mask.float())
        # alpha_factor = obj_alpha * pos_mask.float() + (1 - obj_alpha) * (1 - pos_mask.float())
        # focal_weight = alpha_factor * (1 - pt) ** self.gamma

        # bce_loss = F.binary_cross_entropy(obj_preds_sigmoid, pos_mask.float(), reduction='none')
        # obj_loss = (focal_weight * bce_loss).mean()

        # box_cls_labels = target_dict["pos_equal_one"]  # [B, H, W, A]
        # box_cls_labels = box_cls_labels.view(psm.shape[0], -1).contiguous()

        # positives = box_cls_labels > 0
        # negatives = box_cls_labels == 0
        # negative_cls_weights = negatives * 1.0
        # cls_weights = (negative_cls_weights + 1.0 * positives).float()
        # reg_weights = positives.float()
        # # added
        # pos_mask = target_dict["pos_equal_one"]  # (B, H, W, A)
        # neg_mask = target_dict["neg_equal_one"]  # (B, H, W, A)

        # pos_normalizer = positives.sum(1, keepdim=True).float()
        # reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        # cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        # cls_labels = target_dict["class_ids"]  # [B, H, W, A]
        # cls_targets = cls_labels
        # one_hot_targets = torch.zeros(
        #     *list(cls_targets.shape),
        #     self.cls_num,
        #     dtype=cls_preds.dtype,
        #     device=cls_targets.device,
        # )
        # one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        # cls_labels = one_hot_targets.view(
        #     cls_targets.shape[0], cls_targets.shape[1], cls_targets.shape[2], -1
        # )

        # assert cls_labels.shape == cls_preds.shape, (
        #     f"cls label {cls_labels.shape}, cls_preds shape {cls_preds.shape}"
        # )
        # cls_loss_src = self.cls_loss_func(
        #     cls_preds, cls_labels, weights=cls_weights
        # )  # [N, M]
        # cls_loss = cls_loss_src.sum() / psm.shape[0]
        # conf_loss = cls_loss * self.cls_weight

        # # regression
        # rm = rm.permute(0, 2, 3, 1).contiguous()
        # rm = rm.view(rm.size(0), -1, 7)
        # targets = targets.view(targets.size(0), -1, 7)
        # box_preds_sin, reg_targets_sin = self.add_sin_difference(rm, targets)
        # loc_loss_src = self.reg_loss_func(
        #     box_preds_sin, reg_targets_sin, weights=reg_weights
        # )

        # reg_loss = loc_loss_src.sum() / rm.shape[0]
        # reg_loss *= self.reg_coe

        # # TODO(YH): check obj loss, consider how to weight this
        # obj_preds_sigmoid = torch.sigmoid(obj_preds)
        # bce = -(
        #     pos_mask * torch.log(obj_preds_sigmoid + 1e-6)
        #     + (1 - pos_mask) * torch.log(1 - obj_preds_sigmoid + 1e-6)
        # )
        # obj_loss = bce.mean()

        # # flow feature
        # if "flow" in output_dict:
        #     ego_flow = output_dict["flow"]
        #     ego_flow_loss = self.smooth_l1_loss(ego_flow, 0)
        #     ego_flow_loss = ego_flow_loss.mean() / ego_flow.shape[0]
        #     ego_flow_loss *= self.flow_weight
        # else:
        #     ego_flow_loss = 0

        total_loss = reg_loss + conf_loss + obj_loss
        # total_loss = reg_loss + conf_loss

        # print('psm: ', psm.shape, cls_preds.shape)
        # print('rm: ', rm.shape, box_preds_sin.shape)

        self.loss_dict.update(
            {
                "total_loss{}".format(prefix): total_loss.item(),
                "reg_loss{}".format(prefix): reg_loss.item(),
                "conf_loss{}".format(prefix): conf_loss.item(),
                "obj_loss{}".format(prefix): obj_loss.item(),
            }
        )

        return total_loss

    def cls_loss_func(
        self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
    ):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        # TODO(YH): ensure correctness
        B, H, W, AC = input.shape
        C = self.cls_num  # num_classes (can also be set as self.num_classes)
        A = AC // C

        weights = weights.reshape(B, H, W, A, 1).expand(-1, -1, -1, -1, C)
        weights = weights.contiguous().view(B, H, W, AC)
        assert weights.shape == loss.shape, (
            f"weights shape {weights.shape} must match loss shape {loss.shape}"
        )

        return loss * weights

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = (
            torch.clamp(input, min=0)
            - input * target
            + torch.log1p(torch.exp(-torch.abs(input)))
        )
        return loss

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim : dim + 1]) * torch.cos(
            boxes2[..., dim : dim + 1]
        )
        rad_tg_encoding = torch.cos(boxes1[..., dim : dim + 1]) * torch.sin(
            boxes2[..., dim : dim + 1]
        )

        boxes1 = torch.cat(
            [boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1 :]], dim=-1
        )
        boxes2 = torch.cat(
            [boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1 :]], dim=-1
        )
        return boxes1, boxes2

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n**2 / beta, n - 0.5 * beta)
        return loss

    def logging(self, epoch, batch_id, batch_len, writer=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = [v for k, v in self.loss_dict.items() if "total_loss" in k]
        if len(total_loss) > 1:
            total_loss = sum(total_loss)
        else:
            total_loss = total_loss[0]

        print_msg = "[epoch {}][{}/{}], || Loss: {:.2f} ||".format(
            epoch, batch_id + 1, batch_len, total_loss
        )
        for k, v in self.loss_dict.items():
            print_msg += "{}: {:.2f} | ".format(
                k.replace("_loss", "").replace("_single", ""), v
            )

        if not writer is None:
            for k, v in self.loss_dict.items():
                writer.add_scalar(k, v, epoch * batch_len + batch_id)

        return print_msg

    def _forward(self, output_dict, target_dict, prefix=""):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict

        cls_label -> one_hot label
        """
        rm = output_dict["rm{}".format(prefix)]  # [B, 14, 50, 176]
        psm = output_dict["psm{}".format(prefix)]  # [B, 2, 50, 176]
        targets = target_dict["targets"]

        cls_preds = psm.permute(0, 2, 3, 1).contiguous()  # N, C, H, W -> N, H, W, C

        box_cls_labels = target_dict["pos_equal_one"]  # [B, 50, 176, 2]
        box_cls_labels = box_cls_labels.view(psm.shape[0], -1).contiguous()

        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape),
            2,
            dtype=cls_preds.dtype,
            device=cls_targets.device,
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(psm.shape[0], -1, 1)
        one_hot_targets = one_hot_targets[..., 1:]

        cls_loss_src = self.cls_loss_func(
            cls_preds, one_hot_targets, weights=cls_weights
        )  # [N, M]
        cls_loss = cls_loss_src.sum() / psm.shape[0]
        conf_loss = cls_loss * self.cls_weight

        # regression
        rm = rm.permute(0, 2, 3, 1).contiguous()
        rm = rm.view(rm.size(0), -1, 7)
        targets = targets.view(targets.size(0), -1, 7)
        box_preds_sin, reg_targets_sin = self.add_sin_difference(rm, targets)
        loc_loss_src = self.reg_loss_func(
            box_preds_sin, reg_targets_sin, weights=reg_weights
        )

        reg_loss = loc_loss_src.sum() / rm.shape[0]
        reg_loss *= self.reg_coe

        total_loss = conf_loss
        # total_loss = reg_loss + conf_loss

        # print('psm: ', psm.shape, cls_preds.shape)
        # print('rm: ', rm.shape, box_preds_sin.shape)

        self.loss_dict.update(
            {
                "total_loss{}".format(prefix): total_loss,
                #'reg_loss{}'.format(prefix): reg_loss,
                "conf_loss{}".format(prefix): conf_loss,
            }
        )

        return total_loss

    def _logging(self, epoch, batch_id, batch_len, writer=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = [v.item() for k, v in self.loss_dict.items() if "total_loss" in k]
        if len(total_loss) > 1:
            total_loss = sum(total_loss)
        else:
            total_loss = total_loss[0]
        # reg_loss = self.loss_dict['reg_loss']
        conf_loss = self.loss_dict["conf_loss"]

        print_msg = "[epoch {}][{}/{}], || Loss: {:.2f} ||".format(
            epoch, batch_id + 1, batch_len, total_loss
        )
        for k, v in self.loss_dict.items():
            print_msg += "{}: {:.2f} | ".format(
                k.replace("_loss", "").replace("_single", ""), v.item()
            )

        # print_msg = ("[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
        #             " || Loc Loss: %.4f" % (
        #                 epoch, batch_id + 1, batch_len,
        #                 total_loss.item(), conf_loss.item(), reg_loss.item()))

        if self.use_dir:
            dir_loss = self.loss_dict["dir_loss"]
            print_msg += " || Dir Loss: %.4f" % dir_loss.item()

        # print(print_msg)

        if not writer is None:
            for k, v in self.loss_dict.items():
                writer.add_scalar(k, v.item(), epoch * batch_len + batch_id)
            # writer.add_scalar('Regression_loss', reg_loss.item(),
            #                 epoch*batch_len + batch_id)
            # writer.add_scalar('Confidence_loss', conf_loss.item(),
            #                 epoch*batch_len + batch_id)

            if self.use_dir:
                writer.add_scalar(
                    "dir_loss", dir_loss.item(), epoch * batch_len + batch_id
                )

        return print_msg
