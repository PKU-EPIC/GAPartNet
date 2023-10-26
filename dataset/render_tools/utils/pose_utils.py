from os.path import join as pjoin
import json
import math
import numpy as np
import transforms3d.euler as t
import transforms3d.axangles as tax
import sapien.core as sapien


def query_part_pose_from_joint_qpos(data_path, anno_file, joint_qpos, joints_dict, target_parts, base_link_name, robot: sapien.KinematicArticulation):
    anno_path = pjoin(data_path, anno_file)
    anno_list = json.load(open(anno_path, 'r'))
    
    target_links = {}
    for link_dict in anno_list:
        link_name = link_dict['link_name']
        is_gapart = link_dict['is_gapart']
        part_class = link_dict['category']
        bbox = link_dict['bbox']
        if is_gapart and part_class in target_parts:
            target_links[link_name] = {
                'category_id': target_parts.index(part_class),
                'bbox': np.array(bbox, dtype=np.float32).reshape(-1, 3)
            }
    
    joint_states = {}
    for joint in robot.get_joints():
        joint_name = joint.get_name()
        if joint_name in joints_dict:
            joint_pose = joint.get_parent_link().pose * joint.get_pose_in_parent()
            joint_states[joint_name] = {
                'origin': joint_pose.p,
                'axis': joint_pose.to_transformation_matrix()[:3,:3] @ [1,0,0]
            }
    
    child_link_to_joint_name = {}
    for joint_name, joint_dict in joints_dict.items():
        child_link_to_joint_name[joint_dict['child']] = joint_name
    
    result_dict = {}
    
    for link_name, link_dict in target_links.items():
        joint_names_to_base = []
        cur_name = link_name
        while cur_name in child_link_to_joint_name:
            joint_name = child_link_to_joint_name[cur_name]
            joint_names_to_base.append(joint_name)
            cur_name = joints_dict[joint_name]['parent']
        assert cur_name == base_link_name, 'link {} is not connected to base link {}'.format(link_name, base_link_name)
        joint_names_to_base = joint_names_to_base[:-1]
        
        bbox = link_dict['bbox']
        part_class = link_dict['category_id']
        for joint_name in joint_names_to_base[::-1]:
            joint_type = joints_dict[joint_name]['type']
            origin = joint_states[joint_name]['origin']
            axis = joint_states[joint_name]['axis']
            axis = axis / np.linalg.norm(axis)
            if joint_type == "fixed":
                continue
            elif joint_type == "prismatic":
                bbox = bbox + axis * joint_qpos[joint_name]
            elif joint_type == "revolute" or joint_type == "continuous":
                rotation_mat = t.axangle2mat(axis.reshape(-1).tolist(), joint_qpos[joint_name]).T
                bbox = np.dot(bbox - origin, rotation_mat) + origin
        
        result_dict[link_name] = {
            'category_id': part_class,
            'bbox': bbox
        }
    
    return result_dict


def backproject_depth_into_pointcloud(depth_map, ins_seg_map, valid_linkName_to_instId, camera_intrinsic, eps=1e-6):
    part_pcs_dict = {}
    
    for link_name, inst_id in valid_linkName_to_instId.items():
        mask = (ins_seg_map == inst_id).astype(np.int32)
        area = int(sum(sum(mask > 0)))
        assert area > 0, 'link {} has no area'.format(link_name)
        ys, xs = (mask > 0).nonzero()
        part_pcs = []
        for y, x in zip(ys, xs):
            if abs(depth_map[y][x]) < eps:
                continue
            z_proj = float(depth_map[y][x])
            x_proj = (float(x) - camera_intrinsic[0, 2]) * z_proj / camera_intrinsic[0, 0]
            y_proj = (float(y) - camera_intrinsic[1, 2]) * z_proj / camera_intrinsic[1, 1]
            part_pcs.append([x_proj, y_proj, z_proj])
        assert len(part_pcs) > 0, 'link {} has no valid point'.format(link_name)
        part_pcs_dict[link_name] = np.array(part_pcs).reshape(-1, 3)
    
    return part_pcs_dict


def compute_rotation_matrix(b1, b2):
    c1 = np.mean(b1, axis=0)
    c2 = np.mean(b2, axis=0)
    H = np.dot((b1 - c1).T, (b2 - c2))
    U, s, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    
    if np.linalg.det(R) < 0:
        R[0, :] *= -1

    return R.T


def get_NPCS_map_from_oriented_bbox(depth_map, inst_seg_map, linkName_to_instId, link_pose_dict, camera_intrinsic, world2camera_rotation, camera2world_translation):
    NPCS_RTS_dict = {}
    for link_name in linkName_to_instId.keys():
        bbox = link_pose_dict[link_name]['bbox']
        T = bbox.mean(axis=0)
        s_x = np.linalg.norm(bbox[1] - bbox[0])
        s_y = np.linalg.norm(bbox[1] - bbox[2])
        s_z = np.linalg.norm(bbox[0] - bbox[4])
        S = np.array([s_x, s_y, s_z])
        scaler = np.linalg.norm(S)
        bbox_scaled = (bbox - T) / scaler
        bbox_canon = np.array([
            [-s_x / 2, s_y / 2, s_z / 2],
            [s_x / 2, s_y / 2, s_z / 2],
            [s_x / 2, -s_y / 2, s_z / 2],
            [-s_x / 2, -s_y / 2, s_z / 2],
            [-s_x / 2, s_y / 2, -s_z / 2],
            [s_x / 2, s_y / 2, -s_z / 2],
            [s_x / 2, -s_y / 2, -s_z / 2],
            [-s_x / 2, -s_y / 2, -s_z / 2]
        ]) / scaler
        R = compute_rotation_matrix(bbox_canon, bbox_scaled)
        NPCS_RTS_dict[link_name] = {'R': R, 'T': T, 'S': S, 'scaler': scaler}
    
    height, width = depth_map.shape
    canon_position_map = np.zeros((height, width, 3), dtype=np.float32)
    
    instId_to_linkName = {v: k for k, v in linkName_to_instId.items()}
    assert len(instId_to_linkName) == len(linkName_to_instId)
    for y in range(height):
        for x in range(width):
            if inst_seg_map[y][x] < 0:
                continue
            z_proj = float(depth_map[y][x])
            x_proj = (float(x) - camera_intrinsic[0, 2]) * z_proj / camera_intrinsic[0, 0]
            y_proj = (float(y) - camera_intrinsic[1, 2]) * z_proj / camera_intrinsic[1, 1]
            pixel_camera_position = np.array([x_proj, y_proj, z_proj])
            pixel_world_position = pixel_camera_position @ world2camera_rotation.T + camera2world_translation
            RTS_param = NPCS_RTS_dict[instId_to_linkName[inst_seg_map[y][x]]]
            pixel_npcs_position = ((pixel_world_position - RTS_param['T']) / RTS_param['scaler']) @ RTS_param['R'].T
            canon_position_map[y][x] = pixel_npcs_position
    
    return NPCS_RTS_dict, canon_position_map

