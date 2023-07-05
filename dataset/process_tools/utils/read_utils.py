import os
from os.path import join as pjoin
import json
import numpy as np
from PIL import Image
import pickle


def load_rgb_image(save_path, filename):
    img = Image.open(pjoin(save_path, 'rgb', f'{filename}.png'))
    return np.array(img)


def load_depth_map(save_path, filename):
    depth_dict = np.load(pjoin(save_path, 'depth', f'{filename}.npz'))
    depth_map = depth_dict['depth_map']
    return depth_map


def load_anno_dict(save_path, filename):
    anno_dict = {}
    
    seg_path = pjoin(save_path, 'segmentation')
    bbox_path = pjoin(save_path, 'bbox')
    npcs_path = pjoin(save_path, 'npcs')

    seg_dict = np.load(pjoin(seg_path, f'{filename}.npz'))
    anno_dict['semantic_segmentation'] = seg_dict['semantic_segmentation']
    anno_dict['instance_segmentation'] = seg_dict['instance_segmentation']

    npcs_dict = np.load(pjoin(npcs_path, f'{filename}.npz'))
    anno_dict['npcs_map'] = npcs_dict['npcs_map']

    with open(pjoin(bbox_path, f'{filename}.pkl'), 'rb') as fd:
        bbox_dict = pickle.load(fd)
    anno_dict['bbox_pose_dict'] = bbox_dict['bbox_pose_dict']

    return anno_dict


def load_meta(save_path, filename):
    with open(pjoin(save_path, 'metafile', f'{filename}.json'), 'r') as fd:
        meta = json.load(fd)
    return meta

