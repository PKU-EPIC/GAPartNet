'''
Convert the rendered data into the input format for the GAPartNet framework.

Output .pth format:
point_cloud: (N,3), float32, (x,y,z) in camera coordinate
per_point_rgb: (N,3), float32, ranging in [0,1] (R,G,B)
semantic_label: (N, ), int32, ranging in [0,nClass], 0 for others, [1, nClass] for part categories
instance_label: (N, ), int32, ranging in {-100} \cup [0,nInstance-1], -100 for others, [0, nInstance-1] for parts
NPCS: (N,3), float32, ranging in [-1,1] (x,y,z)
idx: (N,2), int32, (y,x) in the image coordinate
'''

import os
from os.path import join as pjoin
from argparse import ArgumentParser
import numpy as np
import torch
import open3d as o3d

from utils.read_utils import load_rgb_image, load_depth_map, load_anno_dict, load_meta
from utils.sample_utils import FPS

LOG_PATH = './log_sample.txt'

PARTNET_OBJECT_CATEGORIES = [
    'Box', 'Camera', 'CoffeeMachine', 'Dishwasher', 'KitchenPot', 'Microwave', 'Oven', 'Phone', 'Refrigerator',
    'Remote', 'Safe', 'StorageFurniture', 'Table', 'Toaster', 'TrashCan', 'WashingMachine', 'Keyboard', 'Laptop', 'Door', 'Printer',
    'Suitcase', 'Bucket', 'Toilet'
]
AKB48_OBJECT_CATEGORIES = [
    'Box', 'TrashCan', 'Bucket', 'Drawer'
]

MAX_INSTANCE_NUM = 1000

def log_string(file, s):
    file.write(s + '\n')
    print(s)


def get_point_cloud(rgb_image, depth_map, sem_seg_map, ins_seg_map, npcs_map, meta):
    width = meta['width']
    height = meta['height']
    K = np.array(meta['camera_intrinsic']).reshape(3, 3)

    point_cloud = []
    per_point_rgb = []
    per_point_sem_label = []
    per_point_ins_label = []
    per_point_npcs = []
    per_point_idx = []

    for y_ in range(height):
        for x_ in range(width):
            if sem_seg_map[y_, x_] == -2 or ins_seg_map[y_, x_] == -2:
                continue
            z_new = float(depth_map[y_, x_])
            x_new = (x_ - K[0, 2]) * z_new / K[0, 0]
            y_new = (y_ - K[1, 2]) * z_new / K[1, 1]
            point_cloud.append([x_new, y_new, z_new])
            per_point_rgb.append((rgb_image[y_, x_] / 255.0))
            per_point_sem_label.append(sem_seg_map[y_, x_])
            per_point_ins_label.append(ins_seg_map[y_, x_])
            per_point_npcs.append(npcs_map[y_, x_])
            per_point_idx.append([y_, x_])

    return np.array(point_cloud), np.array(per_point_rgb), np.array(per_point_sem_label), np.array(
        per_point_ins_label), np.array(per_point_npcs), np.array(per_point_idx)


def FindMaxDis(pointcloud):
    max_xyz = pointcloud.max(0)
    min_xyz = pointcloud.min(0)
    center = (max_xyz + min_xyz) / 2
    max_radius = ((((pointcloud - center)**2).sum(1))**0.5).max()
    return max_radius, center


def WorldSpaceToBallSpace(pointcloud):
    """
    change the raw pointcloud in world space to united vector ball space
    return: max_radius: the max_distance in raw pointcloud to center
            center: [x,y,z] of the raw center
    """
    max_radius, center = FindMaxDis(pointcloud)
    pointcloud_normalized = (pointcloud - center) / max_radius
    return pointcloud_normalized, max_radius, center


def sample_and_save(filename, data_path, save_path, num_points, visualize=False):
    
    pth_save_path = pjoin(save_path, 'pth')
    os.makedirs(pth_save_path, exist_ok=True)
    meta_save_path = pjoin(save_path, 'meta')
    os.makedirs(meta_save_path, exist_ok=True)
    gt_save_path = pjoin(save_path, 'gt')
    os.makedirs(gt_save_path, exist_ok=True)

    anno_dict = load_anno_dict(data_path, filename)
    metafile = load_meta(data_path, filename)
    rgb_image = load_rgb_image(data_path, filename)
    depth_map = load_depth_map(data_path, filename)
    
    # Get point cloud from back-projection
    pcs, pcs_rgb, pcs_sem, pcs_ins, pcs_npcs, pcs_idx = get_point_cloud(rgb_image,
                                                                        depth_map,
                                                                        anno_dict['semantic_segmentation'],
                                                                        anno_dict['instance_segmentation'],
                                                                        anno_dict['npcs_map'],
                                                                        metafile)
    
    assert ((pcs_sem == -1) == (pcs_ins == -1)).all(), 'Semantic and instance labels do not match!'

    # FPS sampling
    pcs_sampled, fps_idx = FPS(pcs, num_points)
    if pcs_sampled is None:
        return -1

    pcs_rgb_sampled = pcs_rgb[fps_idx]
    pcs_sem_sampled = pcs_sem[fps_idx]
    pcs_ins_sampled = pcs_ins[fps_idx]
    pcs_npcs_sampled = pcs_npcs[fps_idx]
    pcs_idx_sampled = pcs_idx[fps_idx]
    
    # normalize point cloud
    pcs_sampled_normalized, max_radius, center = WorldSpaceToBallSpace(pcs_sampled)
    scale_param = np.array([max_radius, center[0], center[1], center[2]])
    
    # convert semantic and instance labels
    # old label:
    # semantic label: -1 for others, [0, nClass-1] for part categories
    # instance label: -1 for others, [0, nInstance-1] for parts
    # new label:
    # semantic label: 0 for others, [1, nClass] for part categories
    # instance label: -100 for others, [0, nInstance-1] for parts
    pcs_sem_sampled_converted = pcs_sem_sampled + 1
    pcs_ins_sampled_converted = pcs_ins_sampled.copy()
    mask = pcs_ins_sampled_converted == -1
    pcs_ins_sampled_converted[mask] = -100
    
    # re-label instance label to be continuous (discontinuous because of FPS sampling)
    j = 0
    while (j < pcs_ins_sampled_converted.max()):
        if (len(np.where(pcs_ins_sampled_converted == j)[0]) == 0):
            mask = pcs_ins_sampled_converted == pcs_ins_sampled_converted.max()
            pcs_ins_sampled_converted[mask] = j
        j += 1
    
    # visualize
    if visualize:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcs_sampled_normalized)
        pcd.colors = o3d.utility.Vector3dVector(pcs_rgb_sampled)
        o3d.visualization.draw_geometries([pcd])

    torch.save((pcs_sampled_normalized.astype(np.float32), pcs_rgb_sampled.astype(
        np.float32), pcs_sem_sampled_converted.astype(np.int32), pcs_ins_sampled_converted.astype(
            np.int32), pcs_npcs_sampled.astype(np.float32), pcs_idx_sampled.astype(np.int32)), pjoin(pth_save_path, filename + '.pth'))
    np.savetxt(pjoin(meta_save_path, filename + '.txt'), scale_param, delimiter=',')
    
    # save gt for evaluation
    label_sem_ins = np.ones(pcs_ins_sampled_converted.shape, dtype=np.int32) * (-100)
    inst_num = int(pcs_ins_sampled_converted.max() + 1)
    for inst_id in range(inst_num):
        instance_mask = np.where(pcs_ins_sampled_converted == inst_id)[0]
        if instance_mask.shape[0] == 0:
            raise ValueError(f'{filename} has a part missing from point cloud, instance label is not continuous')
        semantic_label = int(pcs_sem_sampled_converted[instance_mask[0]])
        if semantic_label == 0:
            raise ValueError(f'{filename} has a part with semantic label [others]')
        label_sem_ins[instance_mask] = semantic_label * MAX_INSTANCE_NUM + inst_id

    np.savetxt(pjoin(gt_save_path, filename + '.txt'), label_sem_ins, fmt='%d')

    return 0


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='partnet', help='Specify the dataset to render')
    parser.add_argument('--data_path', type=str, default='./rendered_data', help='Specify the path to the rendered data')
    parser.add_argument('--save_path', type=str, default='./sampled_data', help='Specify the path to save the sampled data')
    parser.add_argument('--num_points', type=int, default=20000, help='Specify the number of points to sample')
    parser.add_argument('--visualize', type=bool, default=False, help='Whether to visualize the sampled point cloud')
    
    args = parser.parse_args()
    
    DATASET = args.dataset
    DATA_PATH = args.data_path
    SAVE_PATH = args.save_path
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    NUM_POINTS = args.num_points
    VISUALIZE = args.visualize
    if DATASET == 'partnet':
        OBJECT_CATEGORIES = PARTNET_OBJECT_CATEGORIES
    elif DATASET == 'akb48':
        OBJECT_CATEGORIES = AKB48_OBJECT_CATEGORIES
    else:
        raise ValueError(f'Unknown dataset {DATASET}')
    
    filename_list = sorted([x.split('.')[0] for x in os.listdir(pjoin(DATA_PATH, 'rgb'))])
    filename_dict = {x: [] for x in OBJECT_CATEGORIES}
    for fn in filename_list:
        for x in OBJECT_CATEGORIES:
            if fn.startswith(x):
                filename_dict[x].append(fn)
                break
    
    LOG_FILE = open(LOG_PATH, 'w')

    def log_writer(s):
        log_string(LOG_FILE, s)
    
    for category in filename_dict:
        log_writer(f'Start: {category}')

        fn_list = filename_dict[category]
        log_writer(f'{category} : {len(fn_list)}')
        
        for idx, fn in enumerate(fn_list):
            log_writer(f'Sampling {idx}/{len(fn_list)} {fn}')
            
            ret = sample_and_save(fn, DATA_PATH, SAVE_PATH, NUM_POINTS, VISUALIZE)
            if ret == -1:
                log_writer(f'Error in {fn} {category}, num of points less than NUM_POINTS!')
            else:
                log_writer(f'Finish: {fn}')

        log_writer(f'Finish: {category}')
    
    LOG_FILE.close()

    print('All finished!')

