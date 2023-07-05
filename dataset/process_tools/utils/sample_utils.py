import os
import sys
from os.path import join as pjoin
import numpy as np
from numpy.random.mtrand import sample

import torch

CUDA = torch.cuda.is_available()
if CUDA:
    import pointnet_lib.pointnet2_utils as futils


def farthest_point_sample(xyz, npoint):
    """
    Copied from CAPTRA

    Input:
        xyz: pointcloud data, [B, N, 3], tensor
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    # return torch.randint(0, N, (B, npoint), dtype=torch.long).to(device)
    if CUDA:
        print('Use pointnet2_cuda!')
        idx = futils.furthest_point_sample(xyz, npoint).long()
        return idx

    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B, ), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def FPS(pcs, npoint):
    """
    Input:
        pcs: pointcloud data, [N, 3]
        npoint: number of samples
    Return:
        sampled_pcs: [npoint, 3]
        fps_idx: sampled pointcloud index, [npoint, ]
    """
    if pcs.shape[0] < npoint:
        print('Error! shape[0] of point cloud is less than npoint!')
        return None, None

    if pcs.shape[0] == npoint:
        return pcs, np.arange(pcs.shape[0])

    pcs_tensor = torch.from_numpy(np.expand_dims(pcs, 0)).float()
    fps_idx_tensor = farthest_point_sample(pcs_tensor, npoint)
    fps_idx = fps_idx_tensor.cpu().numpy()[0]
    sampled_pcs = pcs[fps_idx]
    return sampled_pcs, fps_idx


if __name__ == "__main__":
    pc = np.random.random((50000, 3))
    pc_sampled, idx = FPS(pc, 10000)
    print(pc_sampled)
    print(idx)