import os
from os.path import join as pjoin
import xml.etree.ElementTree as ET
import json
import numpy as np
from PIL import Image
import pickle


def get_id_category(target_id, id_path):
    category = None
    with open(id_path, 'r') as fd:
        for line in fd:
            cat = line.rstrip('\n').split(' ')[0]
            id = int(line.rstrip('\n').split(' ')[1])
            if id == target_id:
                category = cat
                break
    return category


def read_joints_from_urdf_file(data_path, urdf_name):
    urdf_file = pjoin(data_path, urdf_name)
    tree_urdf = ET.parse(urdf_file)
    root_urdf = tree_urdf.getroot()
    
    joint_dict = {}
    for joint in root_urdf.iter('joint'):
        joint_name = joint.attrib['name']
        joint_type = joint.attrib['type']
        for child in joint.iter('child'):
            joint_child = child.attrib['link']
        for parent in joint.iter('parent'):
            joint_parent = parent.attrib['link']
        for origin in joint.iter('origin'):
            if 'xyz' in origin.attrib:
                joint_xyz = [float(x) for x in origin.attrib['xyz'].split()]
            else:
                joint_xyz = [0, 0, 0]
            if 'rpy' in origin.attrib:
                joint_rpy = [float(x) for x in origin.attrib['rpy'].split()]
            else:
                joint_rpy = [0, 0, 0]
        if joint_type == 'prismatic' or joint_type == 'revolute' or joint_type == 'continuous':
            for axis in joint.iter('axis'):
                joint_axis = [float(x) for x in axis.attrib['xyz'].split()]
        else:
            joint_axis = None
        if joint_type == 'prismatic' or joint_type == 'revolute':
            for limit in joint.iter('limit'):
                joint_limit = [float(limit.attrib['lower']), float(limit.attrib['upper'])]
        else:
            joint_limit = None
        
        joint_dict[joint_name] = {
            'type': joint_type,
            'parent': joint_parent,
            'child': joint_child,
            'xyz': joint_xyz,
            'rpy': joint_rpy,
            'axis': joint_axis,
            'limit': joint_limit
        }

    return joint_dict


def save_rgb_image(rgb_img, save_path, filename):
    rgb_path = pjoin(save_path, 'rgb')
    if not os.path.exists(rgb_path): os.mkdir(rgb_path)
    
    new_image = Image.fromarray(rgb_img)
    new_image.save(pjoin(rgb_path, f'{filename}.png'))


def save_depth_map(depth_map, save_path, filename):
    depth_path = pjoin(save_path, 'depth')
    if not os.path.exists(depth_path): os.mkdir(depth_path)
    
    np.savez_compressed(pjoin(depth_path, f'{filename}.npz'), depth_map=depth_map)


def save_anno_dict(anno_dict, save_path, filename):
    seg_path = pjoin(save_path, 'segmentation')
    bbox_path = pjoin(save_path, 'bbox')
    npcs_path = pjoin(save_path, 'npcs')

    if not os.path.exists(seg_path): os.mkdir(seg_path)
    if not os.path.exists(bbox_path): os.mkdir(bbox_path)
    if not os.path.exists(npcs_path): os.mkdir(npcs_path)

    np.savez_compressed(pjoin(seg_path, f'{filename}.npz'),
                        semantic_segmentation=anno_dict['semantic_segmentation'],
                        instance_segmentation=anno_dict['instance_segmentation'])

    np.savez_compressed(pjoin(npcs_path, f'{filename}.npz'), npcs_map=anno_dict['npcs_map'])

    with open(pjoin(bbox_path, f'{filename}.pkl'), 'wb') as fd:
        bbox_dict = {'bbox_pose_dict': anno_dict['bbox_pose_dict']}
        pickle.dump(bbox_dict, fd)


def save_meta(meta, save_path, filename):
    meta_path = pjoin(save_path, 'metafile')
    if not os.path.exists(meta_path): os.mkdir(meta_path)
    
    with open(pjoin(meta_path, f'{filename}.json'), 'w') as fd:
        json.dump(meta, fd)


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

