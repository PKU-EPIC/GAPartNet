from json import load
import os
from os.path import join as pjoin
from argparse import ArgumentParser

from utils.config_utils import SAVE_PATH, VISU_SAVE_PATH
from utils.read_utils import load_rgb_image, load_depth_map, load_anno_dict, load_meta
from utils.visu_utils import save_image, visu_depth_map, visu_2D_seg_map, visu_3D_bbox_semantic, visu_3D_bbox_pose_in_color, \
    visu_NPCS_map, visu_point_cloud_with_bbox_semantic, visu_point_cloud_with_bbox_pose_color, visu_NPCS_in_3D_with_bbox_pose_color


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--model_id', type=int)
    parser.add_argument('--category', type=str)
    parser.add_argument('--render_index', type=int, default=0)
    parser.add_argument('--camera_position_index', type=int, default=0)

    CONFS = parser.parse_args()

    MODEL_ID = CONFS.model_id
    CATEGORY = CONFS.category
    RENDER_INDEX = CONFS.render_index
    CAMERA_POSITION_INDEX = CONFS.camera_position_index

    filename = '{}_{}_{}_{}'.format(CATEGORY, MODEL_ID, CAMERA_POSITION_INDEX, RENDER_INDEX)
    save_path = pjoin(VISU_SAVE_PATH, filename)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    rgb_image = load_rgb_image(SAVE_PATH, filename)
    depth_map = load_depth_map(SAVE_PATH, filename)
    anno_dict = load_anno_dict(SAVE_PATH, filename)
    metafile = load_meta(SAVE_PATH, filename)
    
    # depth map
    colored_depth_image = visu_depth_map(depth_map)
    save_image(colored_depth_image, save_path, '{}_depth'.format(filename))

    # semantic segmentation
    sem_seg_image = visu_2D_seg_map(anno_dict['semantic_segmentation'])
    save_image(sem_seg_image, save_path, '{}_semseg'.format(filename))

    # instance segmentation
    ins_seg_image = visu_2D_seg_map(anno_dict['instance_segmentation'])
    save_image(ins_seg_image, save_path, '{}_insseg'.format(filename))

    # 3D bbox with category
    bbox_3D_image_category = visu_3D_bbox_semantic(rgb_image, anno_dict['bbox_pose_dict'], metafile)
    save_image(bbox_3D_image_category, save_path, '{}_bbox3Dcat'.format(filename))

    # 3D bbox with pose color
    bbox_3D_image_pose_color = visu_3D_bbox_pose_in_color(rgb_image, anno_dict['bbox_pose_dict'], metafile)
    save_image(bbox_3D_image_pose_color, save_path, '{}_bbox3Dposecolor'.format(filename))

    # NPCS image
    npcs_image = visu_NPCS_map(anno_dict['npcs_map'], anno_dict['instance_segmentation'])
    save_image(npcs_image, save_path, '{}_NPCS'.format(filename))

    # point cloud with 3D semantic bbox
    visu_point_cloud_with_bbox_semantic(rgb_image, depth_map, anno_dict['bbox_pose_dict'], metafile)

    # point cloud with 3D pose color bbox
    visu_point_cloud_with_bbox_pose_color(rgb_image, depth_map, anno_dict['bbox_pose_dict'], metafile)

    # point cloud of NPCS and 3D pose color bbox
    visu_NPCS_in_3D_with_bbox_pose_color(rgb_image, depth_map, anno_dict, metafile)
    
    print('Done!')
    