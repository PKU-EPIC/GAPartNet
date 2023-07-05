import functools
from typing import List, Optional, Tuple
from unittest import result

import numpy as np
import pytorch_lightning as pl
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from epic_ops.iou import batch_instance_seg_iou
from epic_ops.reduce import segmented_maxpool
from gapartnet.metrics.segmentation import mean_iou, pixel_accuracy
from gapartnet.structures.instances import Instances, Result
from gapartnet.structures.point_cloud import PointCloud
from gapartnet.structures.segmentation import Segmentation
from gapartnet.utils.symmetry_matrix import get_symmetry_matrix
from gapartnet.losses.focal_loss import focal_loss
from gapartnet.losses.dice_loss import dice_loss

from .gapartnet_utils import (apply_nms, cluster_proposals, compute_ap,
                               compute_npcs_loss, filter_invalid_proposals,
                               get_gt_scores, segmented_voxelize)
from .sparse_unet import SparseUNet
from torch.autograd import Function
from .util_net import ReverseLayerF, Discriminator
from gapartnet.utils.info import OBJECT_NAME2ID, PART_ID2NAME, PART_NAME2ID

class InsSeg(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_obj_cats: int,
        channels: List[int],
        block_repeat: int = 2,
        learning_rate: float = 1e-3,
        ignore_sem_label: int = -100,
        ignore_instance_label: int = -100,
        ball_query_radius: float = 0.03,
        max_num_points_per_query: int = 50,
        max_num_points_per_query_shift: int = 300,
        min_num_points_per_proposal: int = 50,
        score_net_start_at: int = 100,
        score_fullscale: float = 14,
        score_scale: float = 50,
        npcs_net_start_at: int = 100,
        symmetry_indices: Optional[List[int]] = None,
        pretrained_model_path: Optional[str] = None,
        loss_sem_seg_weight: Optional[List[float]] = None,
        loss_prop_cls_weight: Optional[List[float]] = None,
        use_focal_loss: bool = False,
        use_dice_loss: bool = False,
        val_score_threshold: float = 0.09,
        val_min_num_points_per_proposal: int = 3,
        val_nms_iou_threshold: float = 0.3,
        val_ap_iou_threshold: float = 0.5,
        # cls_global_weight: float = 0.1,
        # cls_local_weight: float = 0.1,
        # cls_start_at: int = 50,
        reverse: bool = True,
        alpha: float = 0.5,
        discrimination_score_thresh: float = 0.3,
        discrimination_use_score: bool = True,
        ckpt: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_obj_cats = num_obj_cats
        self.channels = channels
        self.reverse = reverse
        self.alpha = alpha
        self.discrimination_score_thresh = discrimination_score_thresh
        self.discrimination_use_score = discrimination_use_score

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        self.unet = SparseUNet.build(in_channels, channels, block_repeat, norm_fn)
        self.sem_seg_head = nn.Linear(channels[0], num_classes)
        self.offset_head = nn.Sequential(
            nn.Linear(channels[0], channels[0]),
            norm_fn(channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(channels[0], 3),
        )

        self.score_unet = SparseUNet.build(
            channels[0], channels[:2], block_repeat, norm_fn, without_stem=True
        )
        self.score_head = nn.Linear(channels[0], num_classes - 1)
        
        self.npcs_unet = SparseUNet.build(
            channels[0], channels[:2], block_repeat, norm_fn, without_stem=True
        )
        self.npcs_head = nn.Linear(channels[0], 3 * (num_classes - 1))

        self.learning_rate = learning_rate
        self.ignore_sem_label = ignore_sem_label
        self.ignore_instance_label = ignore_instance_label
        self.ball_query_radius = ball_query_radius
        self.max_num_points_per_query = max_num_points_per_query
        self.max_num_points_per_query_shift = max_num_points_per_query_shift
        self.min_num_points_per_proposal = min_num_points_per_proposal

        self.score_net_start_at = score_net_start_at
        self.score_fullscale = score_fullscale
        self.score_scale = score_scale

        self.npcs_net_start_at = npcs_net_start_at
        self.register_buffer(
            "symmetry_indices", torch.as_tensor(symmetry_indices, dtype=torch.int64)
        )
        if symmetry_indices is not None:
            assert len(symmetry_indices) == num_classes, (symmetry_indices, num_classes)

            (
                symmetry_matrix_1, symmetry_matrix_2, symmetry_matrix_3
            ) = get_symmetry_matrix()
            self.register_buffer("symmetry_matrix_1", symmetry_matrix_1)
            self.register_buffer("symmetry_matrix_2", symmetry_matrix_2)
            self.register_buffer("symmetry_matrix_3", symmetry_matrix_3)

        if pretrained_model_path is not None:
            print("Loading pretrained model from:", pretrained_model_path)
            state_dict = torch.load(
                pretrained_model_path, map_location="cpu"
            )["state_dict"]
            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False,
            )
            if len(missing_keys) > 0:
                print("missing_keys:", missing_keys)
            if len(unexpected_keys) > 0:
                print("unexpected_keys:", unexpected_keys)

        if loss_sem_seg_weight is None:
            self.loss_sem_seg_weight = loss_sem_seg_weight
        else:
            assert len(loss_sem_seg_weight) == num_classes
            self.register_buffer(
                "loss_sem_seg_weight",
                torch.as_tensor(loss_sem_seg_weight, dtype=torch.float32),
                persistent=False,
            )
        if loss_prop_cls_weight is None:
            self.loss_prop_cls_weight = loss_prop_cls_weight
        else:
            assert len(loss_prop_cls_weight) == num_obj_cats
            self.register_buffer(
                "loss_prop_cls_weight",
                torch.as_tensor(loss_prop_cls_weight, dtype=torch.float32),
                persistent=False,
            )
        self.use_focal_loss = use_focal_loss
        self.use_dice_loss = use_dice_loss

        self.cluster_proposals_start_at = min(
            self.score_net_start_at, self.npcs_net_start_at
        )

        self.val_score_threshold = val_score_threshold
        self.val_min_num_points_per_proposal = val_min_num_points_per_proposal
        self.val_nms_iou_threshold = val_nms_iou_threshold
        self.val_ap_iou_threshold = val_ap_iou_threshold
        self.ckpt = ckpt


    def forward_sem_seg(
        self,
        voxel_tensor: spconv.SparseConvTensor,
        pc_voxel_id: torch.Tensor,
    ) -> Tuple[spconv.SparseConvTensor, torch.Tensor, torch.Tensor]:
        voxel_features = self.unet(voxel_tensor)
        sem_logits = self.sem_seg_head(voxel_features.features)

        pt_features = voxel_features.features[pc_voxel_id]
        sem_logits = sem_logits[pc_voxel_id]

        return voxel_features, pt_features, sem_logits

    def loss_sem_seg(
        self,
        sem_logits: torch.Tensor,
        sem_labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_focal_loss:
            loss = focal_loss(
                sem_logits, sem_labels,
                alpha=self.loss_sem_seg_weight,
                gamma=2.0,
                ignore_index=self.ignore_sem_label,
                reduction="mean",
            )
        else:
            loss = F.cross_entropy(
                sem_logits, sem_labels,
                weight=self.loss_sem_seg_weight,
                ignore_index=self.ignore_sem_label,
                reduction="mean",
            )

        if self.use_dice_loss:
            loss += dice_loss(
                sem_logits[:, :, None, None], sem_labels[:, None, None],
            )

        return loss

    def forward_pt_offset(
        self,
        voxel_features: spconv.SparseConvTensor,
        pc_voxel_id: torch.Tensor,
    ) -> torch.Tensor:
        pt_offsets = self.offset_head(voxel_features.features)
        return pt_offsets[pc_voxel_id]

    def loss_pt_offset(
        self,
        pt_offsets: torch.Tensor,
        gt_pt_offsets: torch.Tensor,
        sem_labels: torch.Tensor,
        instance_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid_instance_mask = (sem_labels > 0) & (instance_labels >= 0)

        pt_diff = pt_offsets - gt_pt_offsets
        pt_dist = torch.sum(pt_diff.abs(), dim=-1)
        loss_pt_offset_dist = pt_dist[valid_instance_mask].mean()

        gt_pt_offsets_norm = torch.norm(gt_pt_offsets, p=2, dim=-1)
        gt_pt_offsets = gt_pt_offsets / (gt_pt_offsets_norm[:, None] + 1e-8)

        pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=-1)
        pt_offsets = pt_offsets / (pt_offsets_norm[:, None] + 1e-8)

        dir_diff = -(gt_pt_offsets * pt_offsets).sum(-1)
        loss_pt_offset_dir = dir_diff[valid_instance_mask].mean()

        return loss_pt_offset_dist, loss_pt_offset_dir

    def cluster_proposals_and_revoxelize(
        self,
        pt_xyz: torch.Tensor,
        batch_indices: torch.Tensor,
        pt_features: torch.Tensor,
        sem_preds: torch.Tensor,
        pt_offsets: torch.Tensor,
        instance_labels: Optional[torch.Tensor],
        cls_labels: Optional[torch.Tensor],
    ):
        device = pt_xyz.device

        # get rid of stuff classes (e.g. wall)
        if instance_labels is not None:
            valid_mask = (sem_preds > 0) & (instance_labels >= 0)
        else:
            valid_mask = sem_preds > 0

        pt_xyz = pt_xyz[valid_mask]
        if pt_xyz.shape[0] == 0:
            return None, None, None

        batch_indices = batch_indices[valid_mask]
        pt_features = pt_features[valid_mask]
        sem_preds = sem_preds[valid_mask].int()
        pt_offsets = pt_offsets[valid_mask]
        if instance_labels is not None:
            instance_labels = instance_labels[valid_mask]

        # get batch offsets (csr) from batch indices
        _, batch_indices_compact, num_points_per_batch = torch.unique_consecutive(
            batch_indices, return_inverse=True, return_counts=True
        )
        batch_indices_compact = batch_indices_compact.int()
        batch_offsets = torch.zeros(
            (num_points_per_batch.shape[0] + 1,), dtype=torch.int32, device=device
        )
        batch_offsets[1:] = num_points_per_batch.cumsum(0)

        # cluster proposals
        sorted_cc_labels, sorted_indices = cluster_proposals(
            pt_xyz, batch_indices_compact, batch_offsets, sem_preds,
            self.ball_query_radius, self.max_num_points_per_query,
        )

        sorted_cc_labels_shift, sorted_indices_shift = cluster_proposals(
            pt_xyz + pt_offsets, batch_indices_compact, batch_offsets, sem_preds,
            self.ball_query_radius, self.max_num_points_per_query_shift,
        )

        # combine clusters
        sorted_cc_labels = torch.cat([
            sorted_cc_labels,
            sorted_cc_labels_shift + sorted_cc_labels.shape[0],
        ], dim=0)
        sorted_indices = torch.cat([sorted_indices, sorted_indices_shift], dim=0)

        # compact the proposal ids
        _, proposal_indices, num_points_per_proposal = torch.unique_consecutive(
            sorted_cc_labels, return_inverse=True, return_counts=True
        )

        # remove small proposals
        valid_proposal_mask = (
            num_points_per_proposal >= self.min_num_points_per_proposal
        )
        # proposal to point
        valid_point_mask = valid_proposal_mask[proposal_indices]

        sorted_indices = sorted_indices[valid_point_mask]
        if sorted_indices.shape[0] == 0:
            return None, None, None

        batch_indices = batch_indices[sorted_indices]
        pt_xyz = pt_xyz[sorted_indices]
        pt_features = pt_features[sorted_indices]
        sem_preds = sem_preds[sorted_indices]
        if instance_labels is not None:
            instance_labels = instance_labels[sorted_indices]

        # re-compact the proposal ids
        proposal_indices = proposal_indices[valid_point_mask]
        _, proposal_indices, num_points_per_proposal = torch.unique_consecutive(
            proposal_indices, return_inverse=True, return_counts=True
        )
        num_proposals = num_points_per_proposal.shape[0]

        # get proposal batch offsets
        proposal_offsets = torch.zeros(
            num_proposals + 1, dtype=torch.int32, device=device
        )
        proposal_offsets[1:] = num_points_per_proposal.cumsum(0)

        # voxelization
        voxel_features, voxel_coords, pc_voxel_id = segmented_voxelize(
            pt_xyz, pt_features,
            proposal_offsets, proposal_indices,
            num_points_per_proposal,
            self.score_fullscale, self.score_scale,
        )
        voxel_tensor = spconv.SparseConvTensor(
            voxel_features, voxel_coords.int(),
            spatial_shape=[self.score_fullscale] * 3,
            batch_size=num_proposals,
        )
        assert (pc_voxel_id >= 0).all()
        cls_labels = cls_labels.to(batch_indices.device)
        proposal_cls_labels = cls_labels[batch_indices.long()]
        proposals = Instances(
            valid_mask=valid_mask,
            sorted_indices=sorted_indices,
            pt_xyz=pt_xyz,
            batch_indices=batch_indices,
            proposal_offsets=proposal_offsets,
            proposal_indices=proposal_indices,
            num_points_per_proposal=num_points_per_proposal,
            sem_preds=sem_preds,
            instance_labels=instance_labels,
            cls_labels=proposal_cls_labels,
        )

        return voxel_tensor, pc_voxel_id, proposals

    def forward_proposal_score(
        self,
        voxel_tensor: spconv.SparseConvTensor,
        pc_voxel_id: torch.Tensor,
        proposals: Instances,
    ):
        proposal_offsets = proposals.proposal_offsets
        proposal_offsets_begin = proposal_offsets[:-1]
        proposal_offsets_end = proposal_offsets[1:]

        score_features = self.score_unet(voxel_tensor)
        score_features = score_features.features[pc_voxel_id]
        pooled_score_features, _ = segmented_maxpool(
            score_features, proposal_offsets_begin, proposal_offsets_end
        )
        score_logits = self.score_head(pooled_score_features)

        return score_logits

    def loss_proposal_score(
        self,
        score_logits: torch.Tensor,
        proposals: Instances,
        num_points_per_instance: torch.Tensor,
    ) -> torch.Tensor:
        ious = batch_instance_seg_iou(
            proposals.proposal_offsets,
            proposals.instance_labels,
            proposals.batch_indices,
            num_points_per_instance,
        )
        proposals.ious = ious
        proposals.num_points_per_instance = num_points_per_instance

        ious_max = ious.max(-1)[0]
        gt_scores = get_gt_scores(ious_max, 0.75, 0.25)

        return F.binary_cross_entropy_with_logits(score_logits, gt_scores)

    def forward_proposal_npcs(
        self,
        voxel_tensor: spconv.SparseConvTensor,
        pc_voxel_id: torch.Tensor,
    ) -> torch.Tensor:
        npcs_features = self.npcs_unet(voxel_tensor)
        npcs_logits = self.npcs_head(npcs_features.features)
        npcs_logits = npcs_logits[pc_voxel_id]

        return npcs_logits

    def loss_proposal_npcs(
        self,
        npcs_logits: torch.Tensor,
        gt_npcs: torch.Tensor,
        proposals: Instances,
    ) -> torch.Tensor:
        sem_preds, sem_labels = proposals.sem_preds, proposals.sem_labels
        proposal_indices = proposals.proposal_indices
        valid_mask = (sem_preds == sem_labels) & (gt_npcs != 0).any(dim=-1)

        npcs_logits = npcs_logits[valid_mask]
        gt_npcs = gt_npcs[valid_mask]
        sem_preds = sem_preds[valid_mask].long()
        sem_labels = sem_labels[valid_mask]
        proposal_indices = proposal_indices[valid_mask]

        npcs_logits = rearrange(npcs_logits, "n (k c) -> n k c", c=3)
        npcs_logits = npcs_logits.gather(
            1, index=repeat(sem_preds - 1, "n -> n one c", one=1, c=3)
        ).squeeze(1)

        proposals.npcs_preds = npcs_logits.detach()
        proposals.gt_npcs = gt_npcs
        proposals.npcs_valid_mask = valid_mask

        loss_npcs = 0

        symmetry_indices = self.symmetry_indices[sem_preds]
        # group #1
        group_1_mask = symmetry_indices < 3
        symmetry_indices_1 = symmetry_indices[group_1_mask]
        if symmetry_indices_1.shape[0] > 0:
            loss_npcs += compute_npcs_loss(
                npcs_logits[group_1_mask], gt_npcs[group_1_mask],
                proposal_indices[group_1_mask],
                self.symmetry_matrix_1[symmetry_indices_1]
            )

        # group #2
        group_2_mask = symmetry_indices == 3
        symmetry_indices_2 = symmetry_indices[group_2_mask]
        if symmetry_indices_2.shape[0] > 0:
            loss_npcs += compute_npcs_loss(
                npcs_logits[group_2_mask], gt_npcs[group_2_mask],
                proposal_indices[group_2_mask],
                self.symmetry_matrix_2[symmetry_indices_2 - 3]
            )

        # group #3
        group_3_mask = symmetry_indices == 4
        symmetry_indices_3 = symmetry_indices[group_3_mask]
        if symmetry_indices_3.shape[0] > 0:
            loss_npcs += compute_npcs_loss(
                npcs_logits[group_3_mask], gt_npcs[group_3_mask],
                proposal_indices[group_3_mask],
                self.symmetry_matrix_3[symmetry_indices_3 - 4]
            )

        return loss_npcs

    def _training_or_validation_step(
        self,
        point_clouds: List[PointCloud],
        batch_idx: int,
        running_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(point_clouds)
        (
            scene_ids, cls_labels, num_points, points, batch_indices, sem_labels, instance_labels, gt_npcs,
            num_instances, instance_regions, num_points_per_instance, instance_sem_labels,
            voxel_tensor, pc_voxel_id,
        ) = PointCloud.collate(point_clouds)
        
        pt_xyz = points[:, :3]
        cls_labels.to(pt_xyz.device)
        assert (pc_voxel_id >= 0).all()


        # semantic segmentation
        voxel_features, pt_features, sem_logits = self.forward_sem_seg(voxel_tensor, pc_voxel_id)
        
        sem_preds = torch.argmax(sem_logits.detach(), dim=-1)

        if sem_labels is not None:
            loss_sem_seg = self.loss_sem_seg(sem_logits, sem_labels)
        else:
            loss_sem_seg = 0.

        sem_seg = Segmentation(batch_size=batch_size,num_points=num_points,sem_preds=sem_preds,sem_labels=sem_labels,)
        
        # point offset
        pt_offsets = self.forward_pt_offset(voxel_features, pc_voxel_id)
        if instance_regions is not None:
            gt_pt_offsets = instance_regions[:, :3] - pt_xyz
            loss_pt_offset_dist, loss_pt_offset_dir = self.loss_pt_offset(
                pt_offsets, gt_pt_offsets, sem_labels, instance_labels,
            )
        else:
            loss_pt_offset_dist, loss_pt_offset_dir = 0., 0.

        if self.current_epoch >= self.cluster_proposals_start_at:
            (
                voxel_tensor, pc_voxel_id, proposals
            ) = self.cluster_proposals_and_revoxelize(
                pt_xyz, batch_indices, pt_features,
                sem_preds, pt_offsets, instance_labels, cls_labels
            )
            if sem_labels is not None and proposals is not None:
                proposals.sem_labels = sem_labels[proposals.valid_mask][
                    proposals.sorted_indices
                ]
            if proposals is not None:
                proposals.instance_sem_labels = instance_sem_labels
        else:
            voxel_tensor, pc_voxel_id, proposals = None, None, None

        # clustering and scoring
        if self.current_epoch >= self.score_net_start_at and voxel_tensor is not None and proposals is not None:
            score_logits = self.forward_proposal_score(
                voxel_tensor, pc_voxel_id, proposals
            )
            proposal_offsets_begin = proposals.proposal_offsets[:-1].long()

            if proposals.sem_labels is not None:
                proposal_sem_labels = proposals.sem_labels[proposal_offsets_begin].long()
            else:
                proposal_sem_labels = proposals.sem_preds[proposal_offsets_begin].long()
            score_logits = score_logits.gather(
                1, proposal_sem_labels[:, None] - 1
            ).squeeze(1)
            proposals.score_preds = score_logits.detach().sigmoid()
            if num_points_per_instance is not None:
                loss_prop_score = self.loss_proposal_score(
                    score_logits, proposals, num_points_per_instance,
                )
            else:
                loss_prop_score = 0.0

        else:
            loss_prop_score = 0.0

        if self.current_epoch >= self.npcs_net_start_at and voxel_tensor is not None:
            npcs_logits = self.forward_proposal_npcs(
                voxel_tensor, pc_voxel_id
            )
            if gt_npcs is not None:
                gt_npcs = gt_npcs[proposals.valid_mask][proposals.sorted_indices]
                loss_prop_npcs = self.loss_proposal_npcs(npcs_logits, gt_npcs, proposals)
            else:
                npcs_preds = npcs_logits.detach()
                npcs_preds = rearrange(npcs_preds, "n (k c) -> n k c", c=3)
                npcs_preds = npcs_preds.gather(
                    1, index=repeat(proposals.sem_preds.long() - 1, "n -> n one c", one=1, c=3)
                ).squeeze(1)
                proposals.npcs_preds = npcs_preds
                loss_prop_npcs = 0.0
        else:
            loss_prop_npcs = 0.0


        # total loss
        loss = loss_sem_seg + loss_pt_offset_dist + loss_pt_offset_dir
        loss += loss_prop_score + loss_prop_npcs #+ self.cls_local_weight * loss_local_cls + self.cls_global_weight * loss_global_cls

        if sem_labels is not None:
            instance_mask = sem_labels > 0
            pixel_acc = pixel_accuracy(sem_preds[instance_mask], sem_labels[instance_mask])
        else:
            pixel_acc = 0.0

        prefix = running_mode
        self.log(f"{prefix}/total_loss", loss, batch_size=batch_size)
        self.log(
            f"{prefix}/loss_sem_seg",
            loss_sem_seg,
            batch_size=batch_size,
        )
        self.log(
            f"{prefix}/loss_pt_offset_dist",
            loss_pt_offset_dist,
            batch_size=batch_size,
        )
        self.log(
            f"{prefix}/loss_pt_offset_dir",
            loss_pt_offset_dir,
            batch_size=batch_size,
        )
        self.log(
            f"{prefix}/loss_prop_score",
            loss_prop_score,
            batch_size=batch_size,
        )
        self.log(
            f"{prefix}/loss_prop_npcs",
            loss_prop_npcs,
            batch_size=batch_size,
        )
        self.log(
            f"{prefix}/pixel_acc",
            pixel_acc * 100,
            batch_size=batch_size,
        )

        return scene_ids, sem_seg, proposals, loss

    def training_step(self, point_clouds: List[PointCloud], batch_idx: int):
        _, _, _, loss = self._training_or_validation_step(
            point_clouds, batch_idx, "train"
        )

        return loss

    def validation_step(self, point_clouds: List[PointCloud], batch_idx: int, dataloader_idx: int = 0):
        split = ["val", "intra", "inter"]
        scene_ids, sem_seg, proposals, _ = self._training_or_validation_step(
            point_clouds, batch_idx, split[dataloader_idx]
        )

        if proposals is not None:
            proposals = filter_invalid_proposals(
                proposals,
                score_threshold=self.val_score_threshold,
                min_num_points_per_proposal=self.val_min_num_points_per_proposal
            )
            proposals = apply_nms(proposals, self.val_nms_iou_threshold)

        if proposals != None:
            proposals.pt_sem_classes = proposals.sem_preds[proposals.proposal_offsets[:-1].long()]
            proposals.valid_mask = None
            proposals.pt_xyz = None
            proposals.sem_preds = None
            proposals.npcs_preds = None
            proposals.sem_labels = None
            proposals.npcs_valid_mask = None
            proposals.gt_npcs = None
        return scene_ids, sem_seg, proposals

    def validation_epoch_end(self, validation_step_outputs_list):
        splits = ["val", "intra", "inter"]
        for i_, validation_step_outputs in enumerate(validation_step_outputs_list):
            split = splits[i_]
        
            batch_size = sum(x[1].batch_size for x in validation_step_outputs)

            proposals = [x[2] for x in validation_step_outputs]
            # torch.save(validation_step_outputs, "wandb/predictions_gap.pth")
            del validation_step_outputs

            # miou = mean_iou(sem_preds, sem_labels, num_classes=self.num_classes)
            # self.log(f"{split}/mean_iou", miou * 100, batch_size=batch_size)
            

            if proposals[0] is not None:

                aps = compute_ap(proposals, self.num_classes, self.val_ap_iou_threshold)

                for class_idx in range(1, self.num_classes):
                    partname = PART_ID2NAME[class_idx]
                    self.log(
                        f"{split}/AP@50_{partname}",
                        aps[class_idx - 1] * 100,
                        batch_size=batch_size,
                        
                    )
                self.log(f"{split}/AP@50", np.mean(aps) * 100, )
            else:
                self.log(f"{split}/AP@50", 0.0, )

    def test_step(self, point_clouds: List[PointCloud], batch_idx: int, dataloader_idx: int = 0):
        split = ["val", "intra", "inter"]
        scene_ids, sem_seg, proposals, _ = self._training_or_validation_step(
            point_clouds, batch_idx, split[dataloader_idx]
        )

        if proposals is not None:
            proposals = filter_invalid_proposals(
                proposals,
                score_threshold=self.val_score_threshold,
                min_num_points_per_proposal=self.val_min_num_points_per_proposal
            )
            proposals = apply_nms(proposals, self.val_nms_iou_threshold)

        if proposals != None:
            proposals.pt_sem_classes = proposals.sem_preds[proposals.proposal_offsets[:-1].long()]
            proposals.valid_mask = None
            proposals.pt_xyz = None
            proposals.sem_preds = None
            proposals.npcs_preds = None
            proposals.sem_labels = None
            proposals.npcs_valid_mask = None
            proposals.gt_npcs = None

        return scene_ids, sem_seg, proposals

    def test_epoch_end(self, validation_step_outputs_list):
        splits = ["val", "intra", "inter"]
        for i_, validation_step_outputs in enumerate(validation_step_outputs_list):
            split = splits[i_]
        
            batch_size = sum(x[1].batch_size for x in validation_step_outputs)

            proposals = [x[2] for x in validation_step_outputs] # if x[2] != None 
            
            del validation_step_outputs

            if proposals[0] is not None:

                aps = compute_ap(proposals, self.num_classes, self.val_ap_iou_threshold)

                for class_idx in range(1, self.num_classes):
                    partname = PART_ID2NAME[class_idx]
                    self.log(
                        f"{split}/AP@50_{partname}",
                        aps[class_idx - 1] * 100,
                        batch_size=batch_size,
                        
                    )
                self.log(f"{split}/AP@50", np.mean(aps) * 100, )

    def test_step_(self, point_clouds: List[PointCloud], batch_idx: int, dataloader_idx: int):
        split = ["val", "intra", "inter"]
        scene_ids, sem_seg, proposals, _ = self._training_or_validation_step(
            point_clouds, batch_idx, split[dataloader_idx]
        )

        if proposals is not None:
            proposals = filter_invalid_proposals(
                proposals,
                score_threshold=self.val_score_threshold,
                min_num_points_per_proposal=self.val_min_num_points_per_proposal
            )
            proposals = apply_nms(proposals, self.val_nms_iou_threshold)
            
        if proposals != None:
            instance_label = torch.zeros_like(sem_seg.sem_preds, device = sem_seg.sem_preds.device)
            instance_label[proposals.valid_mask][proposals.sorted_indices.long()] = proposals.proposal_indices + 1
            npcs_map =  torch.zeros_like(point_clouds[0].points[:,:3], device = sem_seg.sem_preds.device)
            npcs_map[proposals.valid_mask][proposals.sorted_indices.long()][proposals.npcs_valid_mask] = proposals.npcs_preds
            result = Result(xyz = point_clouds[0].points[:,:3], rgb=point_clouds[0].points[:,3:], sem_preds=sem_seg.sem_preds, ins_preds = instance_label, npcs_preds=npcs_map)
        else: 
            instance_label = torch.zeros_like(sem_seg.sem_preds, device = sem_seg.sem_preds.device)
            npcs_map =  torch.zeros_like(point_clouds[0].points[:,:3], device = sem_seg.sem_preds.device)
            result = Result(xyz = point_clouds[0].points[:,:3], rgb=point_clouds[0].points[:,3:], sem_preds=sem_seg.sem_preds, ins_preds = instance_label, npcs_preds=npcs_map)
        
            
        return scene_ids, sem_seg, result

    def test_epoch_end_(self, validation_step_outputs_list):
        splits = ["val", "intra", "inter"]
        for i_, validation_step_outputs in enumerate(validation_step_outputs_list):
            split = splits[i_]
            import pdb
            pdb.set_trace()
        
            batch_size = sum(x[1].batch_size for x in validation_step_outputs)

            proposals = [x[2] for x in validation_step_outputs]
            ids = [y for x in validation_step_outputs for y in x[0]]
            results = [y for x in validation_step_outputs for y in x[2]]
            del validation_step_outputs


            if proposals[0] is not None:
                aps = compute_ap(proposals, self.num_classes, self.val_ap_iou_threshold)

                for class_idx in range(1, self.num_classes):
                    partname = PART_ID2NAME[class_idx]
                    self.log(
                        f"{split}/AP@50_{partname}",
                        aps[class_idx - 1] * 100,
                        batch_size=batch_size,
                        
                    )
                self.log(f"{split}/AP@50", np.mean(aps) * 100, )

    def forward(self, point_clouds: List[PointCloud]):
        scene_ids, sem_seg, proposals, _ = self._training_or_validation_step(
            point_clouds, 0, "val"
        )

        if proposals is not None:
            proposals = filter_invalid_proposals(
                proposals,
                score_threshold=self.val_score_threshold,
                min_num_points_per_proposal=self.val_min_num_points_per_proposal
            )
            proposals = apply_nms(proposals, self.val_nms_iou_threshold)
        return scene_ids, sem_seg, proposals

    def configure_optimizers(self):
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )
