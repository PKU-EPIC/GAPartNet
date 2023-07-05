from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class Segmentation:
    batch_size: int

    sem_preds: torch.Tensor
    sem_labels: Optional[torch.Tensor] = None
    all_accu: Optional[torch.Tensor] = None
    pixel_accu: Optional[float] = None
