from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pyparsing import Opt
import spconv.pytorch as spconv
import torch


@dataclass
class PointCloud:
    scene_id: str
    obj_cat: int

    points: Union[torch.Tensor, np.ndarray]
    

    sem_labels: Optional[Union[torch.Tensor, np.ndarray]] = None
    instance_labels: Optional[Union[torch.Tensor, np.ndarray]] = None

    gt_npcs: Optional[Union[torch.Tensor, np.ndarray]] = None

    num_instances: Optional[int] = None
    instance_regions: Optional[Union[torch.Tensor, np.ndarray]] = None
    num_points_per_instance: Optional[Union[torch.Tensor, np.ndarray]] = None
    instance_sem_labels: Optional[torch.Tensor] = None

    voxel_features: Optional[torch.Tensor] = None
    voxel_coords: Optional[torch.Tensor] = None
    voxel_coords_range: Optional[List[int]] = None
    pc_voxel_id: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }

    def to_tensor(self) -> "PointCloud":
        return PointCloud(**{
            k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in self.to_dict().items()
        })

    def to(self, device: torch.device) -> "PointCloud":
        return PointCloud(**{
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in self.to_dict().items()
        })

    @staticmethod
    def collate(point_clouds: List["PointCloud"]):
        batch_size = len(point_clouds)
        device = point_clouds[0].points.device

        scene_ids = [pc.scene_id for pc in point_clouds]
        cls_labels = torch.tensor([pc.obj_cat for pc in point_clouds])
        num_points = [pc.points.shape[0] for pc in point_clouds]

        points = torch.cat([pc.points for pc in point_clouds], dim=0)
        batch_indices = torch.cat([
            torch.full((pc.points.shape[0],), i, dtype=torch.int32, device=device)
            for i, pc in enumerate(point_clouds)
        ], dim=0)

        if point_clouds[0].sem_labels is not None:
            sem_labels = torch.cat([pc.sem_labels for pc in point_clouds], dim=0)
        else:
            sem_labels = None

        if point_clouds[0].instance_labels is not None:
            instance_labels = torch.cat([pc.instance_labels for pc in point_clouds], dim=0)
        else:
            instance_labels = None

        if point_clouds[0].gt_npcs is not None:
            gt_npcs = torch.cat([pc.gt_npcs for pc in point_clouds], dim=0)
        else:
            gt_npcs = None

        if point_clouds[0].num_instances is not None:
            num_instances = [pc.num_instances for pc in point_clouds]
            max_num_instances = max(num_instances)
            num_points_per_instance = torch.zeros(
                batch_size, max_num_instances, dtype=torch.int32, device=device
            )
            instance_sem_labels = torch.full(
                (batch_size, max_num_instances), -1, dtype=torch.int32, device=device
            )
            for i, pc in enumerate(point_clouds):
                num_points_per_instance[i, :pc.num_instances] = pc.num_points_per_instance
                instance_sem_labels[i, :pc.num_instances] = pc.instance_sem_labels
        else:
            num_instances = None
            num_points_per_instance = None
            instance_sem_labels = None

        if point_clouds[0].instance_regions is not None:
            instance_regions = torch.cat([
                pc.instance_regions for pc in point_clouds
            ], dim=0)
        else:
            instance_regions = None

        voxel_batch_indices = torch.cat([
            torch.full((
                pc.voxel_coords.shape[0],), i, dtype=torch.int32, device=device
            )
            for i, pc in enumerate(point_clouds)
        ], dim=0)
        voxel_coords = torch.cat([
            pc.voxel_coords for pc in point_clouds
        ], dim=0)
        voxel_coords = torch.cat([
            voxel_batch_indices[:, None], voxel_coords
        ], dim=-1)
        voxel_features = torch.cat([
            pc.voxel_features for pc in point_clouds
        ], dim=0)

        voxel_coords_range = np.max([
            pc.voxel_coords_range for pc in point_clouds
        ], axis=0)
        voxel_tensor = spconv.SparseConvTensor(
            voxel_features, voxel_coords,
            spatial_shape=voxel_coords_range.tolist(),
            batch_size=len(point_clouds),
        )

        pc_voxel_id = []
        num_voxel_offset = 0
        for pc in point_clouds:
            pc.pc_voxel_id[pc.pc_voxel_id >= 0] += num_voxel_offset
            pc_voxel_id.append(pc.pc_voxel_id)
            num_voxel_offset += pc.voxel_coords.shape[0]
        pc_voxel_id = torch.cat(pc_voxel_id, dim=0)

        return (
            scene_ids, cls_labels, num_points, points, batch_indices, sem_labels, instance_labels, gt_npcs,
            num_instances, instance_regions, num_points_per_instance, instance_sem_labels,
            voxel_tensor, pc_voxel_id,
        )

if __name__ == "__main__":
    pc = PointCloud(np.ones((10000,3)), np.ones((10000)), np.ones(10000))
    print(pc.to_tensor().to("cuda:0"))
