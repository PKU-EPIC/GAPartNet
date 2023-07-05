import torch
from kornia.metrics import mean_iou as _mean_iou


@torch.no_grad()
def pixel_accuracy(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """
    Compute pixel accuracy.
    """

    if gt_mask.numel() > 0:
        accuracy = (pred_mask == gt_mask).sum() / gt_mask.numel()
        accuracy = accuracy.item()
    else:
        accuracy = 0.
    return accuracy


@torch.no_grad()
def mean_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, num_classes: int) -> float:
    """
    Compute mIoU.
    """

    valid_mask = gt_mask >= 0
    miou = _mean_iou(
        pred_mask[valid_mask][None], gt_mask[valid_mask][None], num_classes=num_classes
    ).mean()
    return miou
