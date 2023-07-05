from dataclasses import dataclass

import torch


@dataclass
class PartPC:
    scene_id: str
    cls_label: int

    points: torch.Tensor
    rgb: torch.Tensor
    npcs: torch.Tensor
