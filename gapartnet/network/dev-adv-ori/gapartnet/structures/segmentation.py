from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class Segmentation:
    batch_size: int = None
    num_points: List[int] = None

    sem_preds: torch.Tensor = None
    sem_labels: Optional[torch.Tensor] = None

