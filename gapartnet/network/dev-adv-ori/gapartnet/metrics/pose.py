import torch
import numpy as np
import itertools

def rot_diff_rad(rot1, rot2):
    mat_diff = np.dot(rot1, rot2.transpose(-1, -2))
    diff = mat_diff[..., 0, 0] + mat_diff[..., 1, 1] + mat_diff[..., 2, 2]
    diff = (diff - 1) / 2.0
    diff = np.clip(diff, -1.0, 1.0)
    return np.arccos(diff)

def z_rot_diff_rad(r1,r2):
    return np.arccos(np.dot(r1,r2.T)/(np.linalg.norm(r1)*np.linalg.norm(r2)))

def z_rot_diff_degree(r1,r2):
    return z_rot_diff_rad(r1,r2) / np.pi * 180.0   


def z_rot_norm_diff_rad(r1,r2):
    return np.arccos(np.linalg.norm(r1*r2)/(np.linalg.norm(r1)*np.linalg.norm(r2)))

def z_rot_norm_diff_degree(r1,r2):
    return z_rot_norm_diff_rad(r1,r2) / np.pi * 180.0  

def rot_diff_degree(rot1, rot2):
    return rot_diff_rad(rot1, rot2) / np.pi * 180.0


def trans_diff(trans1, trans2):
    return np.linalg.norm((trans1 - trans2))  # [..., 3, 1] -> [..., 3] -> [...]


def scale_diff(scale1, scale2):
    return np.absolute(scale1 - scale2)


def theta_diff(theta1, theta2):
    return np.absolute(theta1 - theta2)

def pts_inside_box(pts, bbox):
    # pts: N x 3
    u1 = bbox[1, :] - bbox[0, :]
    u2 = bbox[2, :] - bbox[0, :]
    u3 = bbox[3, :] - bbox[0, :]

    up = pts - np.reshape(bbox[0, :], (1, 3))
    p1 = np.matmul(up, u1.reshape((3, 1)))
    p2 = np.matmul(up, u2.reshape((3, 1)))
    p3 = np.matmul(up, u3.reshape((3, 1)))
    p1 = np.logical_and(p1>0, p1<np.dot(u1, u1))
    p2 = np.logical_and(p2>0, p2<np.dot(u2, u2))
    p3 = np.logical_and(p3>0, p3<np.dot(u3, u3))
    return np.logical_and(np.logical_and(p1, p2), p3)

# def trans_bbox(bbox):


def iou_3d(bbox1, bbox2, nres=50):
    bmin = np.min(np.concatenate((bbox1, bbox2), 0), 0)
    bmax = np.max(np.concatenate((bbox1, bbox2), 0), 0)
    xs = np.linspace(bmin[0], bmax[0], nres)
    ys = np.linspace(bmin[1], bmax[1], nres)
    zs = np.linspace(bmin[2], bmax[2], nres)
    pts = np.array([x for x in itertools.product(xs, ys, zs)])
    flag1 = pts_inside_box(pts, bbox1)
    flag2 = pts_inside_box(pts, bbox2)
    intersect = np.sum(np.logical_and(flag1, flag2))
    union = np.sum(np.logical_or(flag1, flag2))
    if union==0:
        return 1
    else:
        return intersect/float(union)

def dist_between_3d_lines(p1, e1, p2, e2):
    p1 = p1.reshape(-1)
    p2 = p2.reshape(-1)
    e1 = e1.reshape(-1)
    e2 = e2.reshape(-1)
    orth_vect = np.cross(e1, e2)
    product = np.sum(orth_vect * (p1 - p2))
    dist = product / np.linalg.norm(orth_vect)

    return np.abs(dist)

# print(dist_between_3d_lines(np.array([0,0,1]), np.array([0,0,1]), np.array([1,0,0]), np.array([-1,1,0])))
# r1 = np.array([0,0,1])
# r2 = np.array([0,0,-1])
# print(z_rot_diff_degree(r1,r2))
# print(z_rot_norm_diff_degree(r1,r2))