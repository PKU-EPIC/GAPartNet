import os
import sys
from os.path import join as pjoin
import numpy as np
from argparse import ArgumentParser

from utils.config_utils import ID_PATH, DATASET_PATH, CAMERA_POSITION_RANGE, TARGET_GAPARTS, BACKGROUND_RGB, SAVE_PATH
from utils.read_utils import get_id_category, read_joints_from_urdf_file, save_rgb_image, save_depth_map, save_anno_dict, save_meta
from utils.render_utils import get_cam_pos, set_all_scene, render_rgb_image, render_depth_map, \
    render_sem_ins_seg_map, add_background_color_for_image, get_camera_pos_mat, merge_joint_qpos
from utils.pose_utils import query_part_pose_from_joint_qpos, get_NPCS_map_from_oriented_bbox


def render_one_image(model_id, camera_idx, render_idx, height, width, use_raytracing=False, replace_texture=False):
    # 1. read the id list to get the category
    category = get_id_category(model_id, ID_PATH)
    if category is None:
        raise ValueError(f'Cannot find the category of model {model_id}')
    
    # 2. read the urdf file,  get the kinematic chain, and collect all the joints information
    data_path = pjoin(DATASET_PATH, str(model_id))
    joints_dict = read_joints_from_urdf_file(data_path, 'mobility_annotation_gapartnet.urdf')
    
    # 3. generate the joint qpos randomly in the limit range
    joint_qpos = {}
    for joint_name in joints_dict:
        joint_type = joints_dict[joint_name]['type']
        if joint_type == 'prismatic' or joint_type == 'revolute':
            joint_limit = joints_dict[joint_name]['limit']
            joint_qpos[joint_name] = np.random.uniform(joint_limit[0], joint_limit[1])
        elif joint_type == 'fixed':
            joint_qpos[joint_name] = 0.0  # ! the qpos of fixed joint must be 0.0
        elif joint_type == 'continuous':
            joint_qpos[joint_name] = np.random.uniform(-10000.0, 10000.0)
        else:
            raise ValueError(f'Unknown joint type {joint_type}')
    
    # 4. generate the camera pose randomly in the specified range
    camera_range = CAMERA_POSITION_RANGE[category][camera_idx]
    camera_pos = get_cam_pos(
        theta_min=camera_range['theta_min'], theta_max=camera_range['theta_max'],
        phi_min=camera_range['phi_min'], phi_max=camera_range['phi_max'],
        dis_min=camera_range['distance_min'], dis_max=camera_range['distance_max']
    )
    
    # 5. pass the joint qpos and the augmentation parameters to set up render environment and robot
    scene, camera, engine, robot = set_all_scene(data_path=data_path, 
                                        urdf_file='mobility_annotation_gapartnet.urdf',
                                        cam_pos=camera_pos,
                                        width=width, 
                                        height=height,
                                        use_raytracing=False,
                                        joint_qpos_dict=joint_qpos)
    
    # 6. use qpos to calculate the gapart poses
    link_pose_dict = query_part_pose_from_joint_qpos(data_path=data_path, anno_file='link_annotation_gapartnet.json', joint_qpos=joint_qpos, joints_dict=joints_dict, target_parts=TARGET_GAPARTS, robot=robot)
    
    # 7. render the rgb, depth, mask, valid(visible) gapart
    rgb_image = render_rgb_image(camera=camera)
    depth_map = render_depth_map(camera=camera)
    sem_seg_map, ins_seg_map, valid_linkName_to_instId = render_sem_ins_seg_map(scene=scene, camera=camera, link_pose_dict=link_pose_dict, depth_map=depth_map)
    valid_link_pose_dict = {link_name: link_pose_dict[link_name] for link_name in valid_linkName_to_instId.keys()}
    
    # 8. acquire camera intrinsic and extrinsic matrix
    camera_intrinsic, world2camera_rotation, camera2world_translation = get_camera_pos_mat(camera)
    
    # 9. calculate NPCS map
    valid_linkPose_RTS_dict, valid_NPCS_map = get_NPCS_map_from_oriented_bbox(depth_map, ins_seg_map, valid_linkName_to_instId, valid_link_pose_dict, camera_intrinsic, world2camera_rotation, camera2world_translation)
    
    # 10. (optional) use texture to render rgb to replace the previous rgb (texture issue during cutting the mesh)
    if replace_texture:
        texture_joints_dict = read_joints_from_urdf_file(data_path, 'mobility_texture_gapartnet.urdf')
        texture_joint_qpos = merge_joint_qpos(joint_qpos, joints_dict, texture_joints_dict)
        scene, camera, engine, robot = set_all_scene(data_path=data_path, 
                                        urdf_file='mobility_texture_gapartnet.urdf',
                                        cam_pos=camera_pos,
                                        width=width, 
                                        height=height,
                                        use_raytracing=use_raytracing,
                                        joint_qpos_dict=texture_joint_qpos,
                                        engine=engine)
        rgb_image = render_rgb_image(camera=camera)
    
    # 11. add background color
    rgb_image = add_background_color_for_image(rgb_image, depth_map, BACKGROUND_RGB)
    
    # 12. save the rendered results
    save_name = f"{category}_{model_id}_{camera_idx}_{render_idx}"
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    
    save_rgb_image(rgb_image, SAVE_PATH, save_name)
    
    save_depth_map(depth_map, SAVE_PATH, save_name)
    
    bbox_pose_dict = {}
    for link_name in valid_link_pose_dict:
        bbox_pose_dict[link_name] = {
            'bbox': valid_link_pose_dict[link_name]['bbox'],
            'category_id': valid_link_pose_dict[link_name]['category_id'],
            'instance_id': valid_linkName_to_instId[link_name],
            'pose_RTS_param': valid_linkPose_RTS_dict[link_name],
        }
    anno_dict = {
        'semantic_segmentation': sem_seg_map,
        'instance_segmentation': ins_seg_map,
        'npcs_map': valid_NPCS_map,
        'bbox_pose_dict': bbox_pose_dict,
    }
    save_anno_dict(anno_dict, SAVE_PATH, save_name)
    
    metafile = {
        'model_id': model_id,
        'category': category,
        'camera_idx': camera_idx,
        'render_idx': render_idx,
        'width': width,
        'height': height,
        'joint_qpos': joint_qpos,
        'camera_pos': camera_pos.reshape(-1).tolist(),
        'camera_intrinsic': camera_intrinsic.reshape(-1).tolist(),
        'world2camera_rotation': world2camera_rotation.reshape(-1).tolist(),
        'camera2world_translation': camera2world_translation.reshape(-1).tolist(),
        'target_gaparts': TARGET_GAPARTS,
        'use_raytracing': use_raytracing,
        'replace_texture': replace_texture,
    }
    save_meta(metafile, SAVE_PATH, save_name)
    
    print(f"Rendered {save_name} successfully!")
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_id', type=int, default=41083, help='Specify the model id to render')
    parser.add_argument('--camera_idx', type=int, default=0, help='Specify the camera range index to render')
    parser.add_argument('--render_idx', type=int, default=0, help='Specify the render index to render')
    parser.add_argument('--height', type=int, default=800, help='Specify the height of the rendered image')
    parser.add_argument('--width', type=int, default=800, help='Specify the width of the rendered image')
    parser.add_argument('--ray_tracing', type=bool, default=False, help='Specify whether to use ray tracing in rendering')
    parser.add_argument('--replace_texture', type=bool, default=False, help='Specify whether to replace the texture of the rendered image using the original model')
    
    args = parser.parse_args()
    
    render_one_image(args.model_id, args.camera_idx, args.render_idx, args.height, args.width, args.ray_tracing, args.replace_texture)
    
    print("Done!")

