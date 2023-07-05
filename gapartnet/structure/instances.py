from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Instances:
    valid_mask: Optional[torch.Tensor] = None
    sorted_indices: Optional[torch.Tensor] = None
    pt_xyz: Optional[torch.Tensor] = None

    batch_indices: Optional[torch.Tensor] = None
    proposal_offsets: Optional[torch.Tensor] = None
    proposal_indices: Optional[torch.Tensor] = None
    num_points_per_proposal: Optional[torch.Tensor] = None

    sem_preds: Optional[torch.Tensor] = None
    pt_sem_classes: Optional[torch.Tensor] = None
    score_preds: Optional[torch.Tensor] = None
    npcs_preds: Optional[torch.Tensor] = None

    sem_labels: Optional[torch.Tensor] = None
    instance_labels: Optional[torch.Tensor] = None
    instance_sem_labels: Optional[torch.Tensor] = None
    num_points_per_instance: Optional[torch.Tensor] = None
    gt_npcs: Optional[torch.Tensor] = None

    npcs_valid_mask: Optional[torch.Tensor] = None

    ious: Optional[torch.Tensor] = None

    cls_preds: Optional[torch.Tensor] = None
    cls_labels: Optional[torch.Tensor] = None
    
    name: Optional[str] = None

@dataclass
class Result:
    xyz: torch.Tensor
    rgb: torch.Tensor
    sem_preds: torch.Tensor
    ins_preds: torch.Tensor
    npcs_preds: torch.Tensor