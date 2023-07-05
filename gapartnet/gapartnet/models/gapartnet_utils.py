from typing import List, Tuple

import torch
from epic_ops.ball_query import ball_query
from epic_ops.ccl import connected_components_labeling
from epic_ops.nms import nms
from epic_ops.reduce import segmented_reduce
from epic_ops.voxelize import voxelize

from gapartnet.structures.instances import Instances


@torch.jit.script
def compute_npcs_loss(
    npcs_preds: torch.Tensor,
    gt_npcs: torch.Tensor,
    proposal_indices: torch.Tensor,
    symmetry_matrix: torch.Tensor,
) -> torch.Tensor:
    _, num_points_per_proposal = torch.unique_consecutive(
        proposal_indices, return_counts=True
    )

    # gt_npcs: n, 3 -> n, 1, 1, 3
    # symmetry_matrix: n, m, 3, 3
    gt_npcs = gt_npcs[:, None, None, :] @ symmetry_matrix
    # n, m, 1, 3 -> n, m, 3
    gt_npcs = gt_npcs.squeeze(2)

    # npcs_preds: n, 3 -> n, 1, 3
    dist2 = (npcs_preds[:, None, :] - gt_npcs - 0.5) ** 2
    # n, m, 3 -> n, m
    dist2 = dist2.sum(dim=-1)

    loss = torch.where(
        dist2 <= 0.01,
        5 * dist2, torch.sqrt(dist2) - 0.05,
    )
    loss = torch.segment_reduce(
        loss, "mean", lengths=num_points_per_proposal
    )
    loss, _ = loss.min(dim=-1)
    return loss.mean()


@torch.jit.script
def segmented_voxelize(
    pt_xyz: torch.Tensor,
    pt_features: torch.Tensor,
    segment_offsets: torch.Tensor,
    segment_indices: torch.Tensor,
    num_points_per_segment: torch.Tensor,
    score_fullscale: float,
    score_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    segment_offsets_begin = segment_offsets[:-1]
    segment_offsets_end = segment_offsets[1:]

    segment_coords_mean = segmented_reduce(
        pt_xyz, segment_offsets_begin, segment_offsets_end, mode="sum"
    ) / num_points_per_segment[:, None]

    centered_points = pt_xyz - segment_coords_mean[segment_indices]

    segment_coords_min = segmented_reduce(
        centered_points, segment_offsets_begin, segment_offsets_end, mode="min"
    )
    segment_coords_max = segmented_reduce(
        centered_points, segment_offsets_begin, segment_offsets_end, mode="max"
    )

    score_fullscale = 28.
    score_scale = 50.
    segment_scales = 1. / (
        (segment_coords_max - segment_coords_min) / score_fullscale
    ).max(-1)[0] - 0.01
    segment_scales = torch.clamp(segment_scales, min=None, max=score_scale)

    min_xyz = segment_coords_min * segment_scales[..., None]
    max_xyz = segment_coords_max * segment_scales[..., None]

    segment_scales = segment_scales[segment_indices]
    scaled_points = centered_points * segment_scales[..., None]

    range_xyz = max_xyz - min_xyz
    offsets = -min_xyz + torch.clamp(
        score_fullscale - range_xyz - 0.001, min=0
    ) * torch.rand(3, dtype=min_xyz.dtype, device=min_xyz.device) + torch.clamp(
        score_fullscale - range_xyz + 0.001, max=0
    ) * torch.rand(3, dtype=min_xyz.dtype, device=min_xyz.device)
    scaled_points += offsets[segment_indices]

    voxel_features, voxel_coords, voxel_batch_indices, pc_voxel_id = voxelize(
        scaled_points,
        pt_features,
        batch_offsets=segment_offsets.long(),
        voxel_size=torch.as_tensor([1., 1., 1.]),
        points_range_min=torch.as_tensor([0., 0., 0.]),
        points_range_max=torch.as_tensor([score_fullscale, score_fullscale, score_fullscale]),
        reduction="mean",
    )
    voxel_coords = torch.cat([voxel_batch_indices[:, None], voxel_coords], dim=1)

    return voxel_features, voxel_coords, pc_voxel_id


@torch.jit.script
def cluster_proposals(
    pt_xyz: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_offsets: torch.Tensor,
    sem_preds: torch.Tensor,
    ball_query_radius: float,
    max_num_points_per_query: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = pt_xyz.device
    index_dtype = batch_indices.dtype

    clustered_indices, num_points_per_query = ball_query(
        pt_xyz,
        pt_xyz,
        batch_indices,
        batch_offsets,
        ball_query_radius,
        max_num_points_per_query,
        point_labels=sem_preds,
        query_labels=sem_preds,
    )

    ccl_indices_begin = torch.arange(
        pt_xyz.shape[0], dtype=index_dtype, device=device
    ) * max_num_points_per_query
    ccl_indices_end = ccl_indices_begin + num_points_per_query
    ccl_indices = torch.stack([ccl_indices_begin, ccl_indices_end], dim=1)
    cc_labels = connected_components_labeling(
        ccl_indices.view(-1), clustered_indices.view(-1), compacted=False
    )

    sorted_cc_labels, sorted_indices = torch.sort(cc_labels)
    return sorted_cc_labels, sorted_indices


@torch.jit.script
def get_gt_scores(
    ious: torch.Tensor, fg_thresh: float = 0.75, bg_thresh: float = 0.25
) -> torch.Tensor:
    fg_mask = ious > fg_thresh
    bg_mask = ious < bg_thresh
    intermidiate_mask = ~(fg_mask | bg_mask)

    gt_scores = fg_mask.float()
    k = 1 / (fg_thresh - bg_thresh)
    b = bg_thresh / (bg_thresh - fg_thresh)
    gt_scores[intermidiate_mask] = ious[intermidiate_mask] * k + b

    return gt_scores


def filter_invalid_proposals(
    proposals: Instances,
    score_threshold: float,
    min_num_points_per_proposal: int,
) -> Instances:
    score_preds = proposals.score_preds
    proposal_indices = proposals.proposal_indices
    num_points_per_proposal = proposals.num_points_per_proposal

    valid_proposals_mask = (
        score_preds > score_threshold
    ) & (num_points_per_proposal > min_num_points_per_proposal)
    valid_points_mask = valid_proposals_mask[proposal_indices]

    proposal_indices = proposal_indices[valid_points_mask]
    _, proposal_indices, num_points_per_proposal = torch.unique_consecutive(
        proposal_indices, return_inverse=True, return_counts=True
    )
    num_proposals = num_points_per_proposal.shape[0]

    proposal_offsets = torch.zeros(
        num_proposals + 1, dtype=torch.int32, device=proposal_indices.device
    )
    proposal_offsets[1:] = num_points_per_proposal.cumsum(0)

    if proposals.npcs_valid_mask is not None:
        valid_npcs_mask = valid_points_mask[proposals.npcs_valid_mask]
    else:
        valid_npcs_mask = valid_points_mask

    return Instances(
        valid_mask=proposals.valid_mask,
        sorted_indices=proposals.sorted_indices[valid_points_mask],
        pt_xyz=proposals.pt_xyz[valid_points_mask],
        batch_indices=proposals.batch_indices[valid_points_mask],
        proposal_offsets=proposal_offsets,
        proposal_indices=proposal_indices,
        num_points_per_proposal=num_points_per_proposal,
        sem_preds=proposals.sem_preds[valid_points_mask],
        score_preds=proposals.score_preds[valid_proposals_mask],
        npcs_preds=proposals.npcs_preds[
            valid_npcs_mask
        ] if proposals.npcs_preds is not None else None,
        sem_labels=proposals.sem_labels[
            valid_points_mask
        ] if proposals.sem_labels is not None else None,
        instance_labels=proposals.instance_labels[
            valid_points_mask
        ] if proposals.instance_labels is not None else None,
        instance_sem_labels=proposals.instance_sem_labels,
        num_points_per_instance=proposals.num_points_per_instance,
        gt_npcs=proposals.gt_npcs[
            valid_npcs_mask
        ] if proposals.gt_npcs is not None else None,
        npcs_valid_mask=proposals.npcs_valid_mask[valid_points_mask] \
            if proposals.npcs_valid_mask is not None else None,
        ious=proposals.ious[
            valid_proposals_mask
        ] if proposals.ious is not None else None,
    )


def apply_nms(
    proposals: Instances,
    iou_threshold: float = 0.3,
):
    score_preds = proposals.score_preds
    sorted_indices = proposals.sorted_indices
    proposal_offsets = proposals.proposal_offsets
    proposal_indices = proposals.proposal_indices
    num_points_per_proposal = proposals.num_points_per_proposal

    values = torch.ones(
        sorted_indices.shape[0], dtype=torch.float32, device=sorted_indices.device
    )
    csr = torch.sparse_csr_tensor(
        proposal_offsets.int(), sorted_indices.int(), values,
        dtype=torch.float32, device=sorted_indices.device,
    )
    intersection = csr @ csr.t()
    intersection = intersection.to_dense()
    union = num_points_per_proposal[:, None] + num_points_per_proposal[None, :]
    union = union - intersection

    ious = intersection / (union + 1e-8)
    keep = nms(ious.cuda(), score_preds.cuda(), iou_threshold)
    keep = keep.to(score_preds.device)

    valid_proposals_mask = torch.zeros(
        ious.shape[0], dtype=torch.bool, device=score_preds.device
    )
    valid_proposals_mask[keep] = True
    valid_points_mask = valid_proposals_mask[proposal_indices]

    proposal_indices = proposal_indices[valid_points_mask]
    _, proposal_indices, num_points_per_proposal = torch.unique_consecutive(
        proposal_indices, return_inverse=True, return_counts=True
    )
    num_proposals = num_points_per_proposal.shape[0]

    proposal_offsets = torch.zeros(
        num_proposals + 1, dtype=torch.int32, device=proposal_indices.device
    )
    proposal_offsets[1:] = num_points_per_proposal.cumsum(0)

    if proposals.npcs_valid_mask is not None:
        valid_npcs_mask = valid_points_mask[proposals.npcs_valid_mask]
    else:
        valid_npcs_mask = valid_points_mask

    return Instances(
        valid_mask=proposals.valid_mask,
        sorted_indices=proposals.sorted_indices[valid_points_mask],
        pt_xyz=proposals.pt_xyz[valid_points_mask],
        batch_indices=proposals.batch_indices[valid_points_mask],
        proposal_offsets=proposal_offsets,
        proposal_indices=proposal_indices,
        num_points_per_proposal=num_points_per_proposal,
        sem_preds=proposals.sem_preds[valid_points_mask],
        score_preds=proposals.score_preds[valid_proposals_mask],
        npcs_preds=proposals.npcs_preds[
            valid_npcs_mask
        ] if proposals.npcs_preds is not None else None,
        sem_labels=proposals.sem_labels[
            valid_points_mask
        ] if proposals.sem_labels is not None else None,
        instance_labels=proposals.instance_labels[
            valid_points_mask
        ] if proposals.instance_labels is not None else None,
        instance_sem_labels=proposals.instance_sem_labels,
        num_points_per_instance=proposals.num_points_per_instance,
        gt_npcs=proposals.gt_npcs[
            valid_npcs_mask
        ] if proposals.gt_npcs is not None else None,
        npcs_valid_mask=proposals.npcs_valid_mask[valid_points_mask] \
            if proposals.npcs_valid_mask is not None else None,
        ious=proposals.ious[
            valid_proposals_mask
        ] if proposals.ious is not None else None,
    )


@torch.jit.script
def voc_ap(
    rec: torch.Tensor,
    prec: torch.Tensor,
    use_07_metric: bool = False,
) -> float:
    if use_07_metric:
        # 11 point metric
        ap = torch.as_tensor(0, dtype=prec.dtype, device=prec.device)
        for t in range(0, 11, 1):
            t /= 10.0
            if torch.sum(rec >= t) == 0:
                p = torch.as_tensor(0, dtype=prec.dtype, device=prec.device)
            else:
                p = torch.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = torch.cat([
            torch.as_tensor([0.0], dtype=rec.dtype, device=rec.device),
            rec,
            torch.as_tensor([1.0], dtype=rec.dtype, device=rec.device),
        ], dim=0)
        mpre = torch.cat([
            torch.as_tensor([0.0], dtype=prec.dtype, device=prec.device),
            prec,
            torch.as_tensor([0.0], dtype=prec.dtype, device=prec.device),
        ], dim=0)

        # compute the precision envelope
        for i in range(mpre.shape[0] - 1, 0, -1):
            mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = torch.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = torch.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap.item())


@torch.jit.script
def _compute_ap_per_class(
    tp: torch.Tensor, fp: torch.Tensor, num_gt_instances: int
) -> float:
    if tp.shape[0] == 0:
        return 0.

    tp = tp.cumsum(0)
    fp = fp.cumsum(0)
    rec = tp / num_gt_instances
    prec = tp / (tp + fp + 1e-8)

    return voc_ap(rec, prec)


@torch.jit.script
def _compute_ap(
    confidence: torch.Tensor,
    classes: torch.Tensor,
    sorted_indices: torch.Tensor,
    batch_indices: torch.Tensor,
    sample_indices: torch.Tensor,
    proposal_indices: torch.Tensor,
    matched: List[torch.Tensor],
    instance_sem_labels: List[torch.Tensor],
    ious: List[torch.Tensor],
    num_classes: int,
    iou_threshold: float,
):
    sorted_indices_cpu = sorted_indices.cpu()

    num_proposals = confidence.shape[0]
    tp = torch.zeros(num_proposals, dtype=torch.float32)
    fp = torch.zeros(num_proposals, dtype=torch.float32)
    for i in range(num_proposals):
        idx = sorted_indices_cpu[i]

        class_idx = classes[idx]
        batch_idx = batch_indices[idx].item()
        sample_idx = sample_indices[idx]
        proposal_idx = proposal_indices[idx]

        instance_sem_labels_i = instance_sem_labels[batch_idx][sample_idx]
        invalid_instance_mask = instance_sem_labels_i != class_idx

        ious_i = ious[batch_idx][proposal_idx].clone()
        ious_i[invalid_instance_mask] = 0.
        if ious_i.shape[0] == 0:
            max_iou, max_idx = 0., 0
        else:
            max_iou, max_idx = ious_i.max(0)
            max_iou, max_idx = max_iou.item(), int(max_idx.item())

        if max_iou > iou_threshold:
            if not matched[batch_idx][sample_idx, max_idx].item():
                tp[i] = 1.0
                matched[batch_idx][sample_idx, max_idx] = True
            else:
                fp[i] = 1.0
        else:
            fp[i] = 1.0

    tp = tp.to(device=confidence.device)
    fp = fp.to(device=confidence.device)

    sorted_classes = classes[sorted_indices]
    gt_classes = torch.cat([x.view(-1) for x in instance_sem_labels], dim=0)
    aps: List[float] = []
    for c in range(1, num_classes):
        num_gt_instances = (gt_classes == c).sum()
        mask = sorted_classes == c
        ap = _compute_ap_per_class(tp[mask], fp[mask], num_gt_instances)
        aps.append(ap)
    return aps


def compute_ap(
    proposals: List[Instances],
    num_classes: int = 9,
    iou_threshold: float = 0.5,
    device="cpu",
):
    confidence = torch.cat([
        p.score_preds for p in proposals if p is not None
    ], dim=0).to(device=device)
    classes = torch.cat([
        p.pt_sem_classes
        for p in proposals if p is not None
    ], dim=0).to(device=device)
    sorted_indices = torch.argsort(confidence, descending=True)

    batch_indices = torch.cat([
        torch.full((p.score_preds.shape[0],), i, dtype=torch.int64)
        for i, p in enumerate(proposals)  if p is not None
    ], dim=0)
    sample_indices = torch.cat([
        p.batch_indices[p.proposal_offsets[:-1].long()].long()
        for p in proposals  if p is not None
    ], dim=0).cpu()
    proposal_indices = torch.cat([
        torch.arange(p.score_preds.shape[0], dtype=torch.int64)
        for p in proposals  if p is not None
    ], dim=0)

    matched = [
        torch.zeros_like(p.instance_sem_labels, dtype=torch.bool, device="cpu")
        for p in proposals  if p is not None
    ]

    return _compute_ap(
        confidence,
        classes,
        sorted_indices,
        batch_indices,
        sample_indices,
        proposal_indices,
        matched,
        [p.instance_sem_labels.to(device=device) for p in proposals if p is not None],
        [p.ious.to(device=device) for p in proposals if p is not None],
        num_classes,
        iou_threshold,
    )
