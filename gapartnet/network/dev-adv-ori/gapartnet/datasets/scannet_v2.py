import copy
from functools import partial
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchdata.datapipes as dp
from epic_ops.voxelize import voxelize
from torch.utils.data import DataLoader

from gapartnet.structures.point_cloud import PointCloud
from gapartnet.utils import data as data_utils


def apply_augmentations(
    pc: PointCloud, *, jitter: float = 0, flip_prob: float = 0, rotate: bool = False
) -> PointCloud:
    pc = copy.copy(pc)

    m = np.eye(3)
    if jitter > 0:
        m += np.random.randn(3, 3) * jitter

    if flip_prob > 0:
        if np.random.rand() < flip_prob:
            m[0, 0] = -m[0, 0]

    if rotate:
        theta = np.random.rand() * np.pi * 2
        m = m @ np.asarray([
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])

    pc.points = pc.points.copy()
    pc.points[:, :3] = pc.points[:, :3] @ m

    if jitter > 0:
        pc.points[:, 3:] += np.random.randn(1, 3) * jitter

    return pc


def downsample(pc: PointCloud, *, max_points: int = 250000) -> PointCloud:
    pc = copy.copy(pc)

    num_points = pc.points.shape[0]

    if num_points > max_points:
        indices = np.random.choice(num_points, max_points, replace=False)
        pc.points = pc.points[indices]
        pc.sem_labels = pc.sem_labels[indices]
        pc.instance_labels = pc.instance_labels[indices]

    return pc


def compact_instance_labels(pc: PointCloud) -> PointCloud:
    pc = copy.copy(pc)

    valid_mask = pc.instance_labels >= 0
    instance_labels = pc.instance_labels[valid_mask]
    _, instance_labels = np.unique(instance_labels, return_inverse=True)
    pc.instance_labels[valid_mask] = instance_labels

    return pc


def generate_inst_info(pc: PointCloud) -> PointCloud:
    pc = copy.copy(pc)

    num_points = pc.points.shape[0]

    num_instances = int(pc.instance_labels.max()) + 1
    instance_regions = np.zeros((num_points, 9), dtype=np.float32)
    num_points_per_instance = []

    for i in range(num_instances):
        indices = np.where(pc.instance_labels == i)[0]

        xyz_i = pc.points[indices, :3]
        min_i = xyz_i.min(0)
        max_i = xyz_i.max(0)
        mean_i = xyz_i.mean(0)
        instance_regions[indices, 0:3] = mean_i
        instance_regions[indices, 3:6] = min_i
        instance_regions[indices, 6:9] = max_i

        num_points_per_instance.append(indices.shape[0])

    pc.num_instances = num_instances
    pc.instance_regions = instance_regions
    pc.num_points_per_instance = np.asarray(num_points_per_instance, dtype=np.int32)

    return pc


def apply_voxelization(pc: PointCloud) -> PointCloud:
    pc = copy.copy(pc)

    num_points = pc.points.shape[0]
    points_range_min = pc.points[:, :3].min(0)[0] - 1e-4
    points_range_max = pc.points[:, :3].max(0)[0] + 1e-4
    voxel_features, voxel_coords, _, pc_voxel_id = voxelize(
        pc.points[:, :3], pc.points,
        batch_offsets=torch.as_tensor([0, num_points], dtype=torch.int64),
        voxel_size=[1 / 50, 1 / 50, 1 / 50],
        points_range_min=points_range_min.tolist(),
        points_range_max=points_range_max.tolist(),
        reduction="mean",
    )

    voxel_coords_range = (voxel_coords.max(0)[0] + 1).clamp(min=128, max=None)

    pc.voxel_features = voxel_features
    pc.voxel_coords = voxel_coords
    pc.voxel_coords_range = voxel_coords_range.tolist()
    pc.pc_voxel_id = pc_voxel_id

    return pc


def from_folder(
    root_dir: Union[str, Path] = "",
    shuffle: bool = False,
    max_points: int = 250000,
    augmentation: bool = False,
):
    pipe = dp.iter.FileLister(str(root_dir))
    pipe = pipe.filter(filter_fn=lambda x: x.endswith("_inst_nostuff.pth"))

    pipe = pipe.distributed_sharding_filter()
    if shuffle:
        pipe = pipe.shuffle()

    # Load data
    pipe = pipe.map(lambda x: torch.load(x))
    pipe = pipe.map(
        lambda x: PointCloud(
            points=np.concatenate([x[0], x[1]], axis=-1, dtype=np.float32),
            sem_labels=x[2].astype(np.int64),
            instance_labels=x[3].astype(np.int32),
        )
    )

    # Augmentations
    if augmentation:
        pipe = pipe.map(
            partial(apply_augmentations, jitter=0.1, flip_prob=0, rotate=False)
        )

    # Downsample
    pipe = pipe.map(partial(downsample, max_points=max_points))
    pipe = pipe.map(compact_instance_labels)

    # Generate instance info
    pipe = pipe.map(generate_inst_info)

    # To tensor
    pipe = pipe.map(lambda pc: pc.to_tensor())

    # Voxelization
    pipe = pipe.map(apply_voxelization)

    return pipe


class ScanNetV2Inst(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        max_points: int = 250000,
        train_batch_size: int = 4,
        val_batch_size: int = 4,
        test_batch_size: int = 4,
        num_workers: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.root_dir = root_dir
        self.max_points = max_points
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_data_pipe = from_folder(
                Path(self.root_dir) / "train",
                shuffle=True,
                max_points=self.max_points,
                augmentation=True,
            )

            self.val_data_pipe = from_folder(
                Path(self.root_dir) / "val",
                shuffle=False,
                max_points=self.max_points,
                augmentation=False,
            )

        if stage in (None, "test"):
            self.test_data_pipe = from_folder(
                Path(self.root_dir) / "test",
                shuffle=False,
                max_points=self.max_points,
                augmentation=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data_pipe,
            batch_size=self.train_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=data_utils.trivial_batch_collator,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data_pipe,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=data_utils.trivial_batch_collator,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data_pipe,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=data_utils.trivial_batch_collator,
            pin_memory=True,
            drop_last=False,
        )
