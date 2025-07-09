import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalSegLossSiCP(nn.Module):
    def __init__(self, args):
        """
        Multi-class Focal Loss for segmentation.

        Parameters:
        - gamma: focusing parameter.
        - alpha: class weighting factor, tensor of shape (C,) or scalar.
        - reduction: 'mean', 'sum', or 'none'.
        - ignore_index: optional, index to ignore during loss computation.
        """
        super(FocalSegLossSiCP, self).__init__()
        self.gamma = args["gamma"]
        self.alpha = args["alpha"]
        self.reduction = args["reduction"]
        self.ignore_index = args.get("ignore_index", None)
        self.loss_dict = {}
        self.use_ce = args.get("use_ce", False)
        if self.use_ce:
            self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, output_dict, target_dict):
        dynamic_seg_logits = output_dict["dynamic_seg"]
        static_seg_logits = output_dict["static_seg"]

        dynamic_seg_label = target_dict["dynamic_seg_label"]
        static_seg_label = target_dict["static_seg_label"]

        if self.use_ce:
            dynamic_loss = self.ce_loss(
                dynamic_seg_logits, dynamic_seg_label.long()
            )
            static_loss = self.ce_loss(
                static_seg_logits, static_seg_label.long()
            )
            total_loss = dynamic_loss + static_loss
        else:
            dynamic_loss = self.single_seg_forward(
                {"seg_logits": dynamic_seg_logits}, {"seg_label": dynamic_seg_label}
            )
            static_loss = self.single_seg_forward(
                {"seg_logits": static_seg_logits}, {"seg_label": static_seg_label}
            )
            total_loss = dynamic_loss + static_loss

        self.loss_dict.update({"total_loss": total_loss.item(), "dynamic_loss": dynamic_loss.item(), "static_loss": static_loss.item()})
        return total_loss

    def single_seg_forward(self, output_dict, target_dict):
        """
        logits: (B, C, H, W) — raw predictions (not softmaxed)
        target: (B, H, W) — ground-truth class indices
        """
        logits = output_dict["seg_logits"]  # (B, C, H, W)
        target = target_dict["seg_label"]
        target = target.long()

        logpt = F.log_softmax(logits, dim=1)  # (B, C, H, W)
        pt = torch.exp(logpt)
        pt = torch.clamp(pt, min=1e-7, max=1.0)

        logpt = logpt.gather(1, target.unsqueeze(1))
        pt = pt.gather(1, target.unsqueeze(1))

        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha = torch.tensor(self.alpha, device=logits.device)
            elif isinstance(self.alpha, float):
                alpha = torch.ones(logits.shape[1], device=logits.device)
                alpha[1:] = self.alpha
            else:
                alpha = self.alpha.to(logits.device)

            at = alpha.gather(0, target.view(-1)).view_as(target).unsqueeze(1)
            logpt = logpt * at

        loss = -((1 - pt) ** self.gamma) * logpt

        if self.ignore_index is not None:
            mask = (target != self.ignore_index).unsqueeze(1)
            loss = loss * mask
            if self.reduction == "mean":
                loss = loss.sum() / mask.sum()
            elif self.reduction == "sum":
                loss = loss.sum()
        else:
            if self.reduction == "mean":
                loss = loss.mean()
            elif self.reduction == "sum":
                loss = loss.sum()

        # self.loss_dict.update({"total_loss": loss.item()})
        return loss

    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
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
        total_loss = self.loss_dict["total_loss"]
        dynamic_loss = self.loss_dict["dynamic_loss"]
        static_loss = self.loss_dict["static_loss"] 

        if pbar is None:
            print_msg = (
                "[epoch %d][%d/%d], || Total Loss: %.4f || Dynamic Loss: %.4f || Static Loss: %.4f ||"
                % (epoch, batch_id + 1, batch_len, total_loss, dynamic_loss, static_loss)
            )
            print(print_msg)
        
        else:
            pbar.set_description(
                "[epoch %d][%d/%d], || Total Loss: %.4f || Dynamic Loss: %.4f || Static Loss: %.4f ||"
                % (epoch, batch_id + 1, batch_len, total_loss, dynamic_loss, static_loss)
            )

        if writer is not None:
            writer.add_scalar("total_loss", total_loss, epoch * batch_len + batch_id)
            writer.add_scalar("dynamic_loss", dynamic_loss, epoch * batch_len + batch_id)
            writer.add_scalar("static_loss", static_loss, epoch * batch_len + batch_id)
