import os
from os.path import join as pjoin
import math
import numpy as np
import sapien.core as sapien
import transforms3d.euler as t
import transforms3d.axangles as tax


def get_cam_pos(theta_min, theta_max, phi_min, phi_max, dis_min, dis_max):
    theta = np.random.uniform(low=theta_min, high=theta_max)
    phi = np.random.uniform(low=phi_min, high=phi_max)
    distance = np.random.uniform(low=dis_min, high=dis_max)
    x = math.sin(math.pi / 180 * theta) * math.cos(math.pi / 180 * phi) * distance
    y = math.sin(math.pi / 180 * theta) * math.sin(math.pi / 180 * phi) * distance
    z = math.cos(math.pi / 180 * theta) * distance
    return np.array([x, y, z])


def set_all_scene(data_path,
                  urdf_file,
                  cam_pos,
                  width,
                  height,
                  joint_qpos_dict,
                  engine=None,
                  use_raytracing=False):

    # set the sapien environment
    if engine is None:
        engine = sapien.Engine()
        if use_raytracing:
            config = sapien.KuafuConfig()
            config.spp = 256
            config.use_denoiser = True
            renderer = sapien.KuafuRenderer(config)
        else:
            renderer = sapien.VulkanRenderer(offscreen_only=True)
        engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    # load model
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    urdf_path = os.path.join(data_path, urdf_file)
    robot = loader.load_kinematic(urdf_path)
    assert robot, 'URDF not loaded.'
    
    joints = robot.get_joints()
    qpos = []
    for joint in joints:
        if joint.get_parent_link() is None:
            continue
        joint_name = joint.get_name()
        joint_type = joint.type
        if joint_type == 'revolute' or joint_type == 'prismatic' or joint_type == 'continuous':
            qpos.append(joint_qpos_dict[joint_name])
    qpos = np.array(qpos)
    assert qpos.shape[0] == robot.get_qpos().shape[0], 'qpos shape not match.'
    robot.set_qpos(qpos=qpos)

    # * different in server and local (sapien version issue)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)
    
    # rscene = scene.get_renderer_scene()
    # rscene.set_ambient_light([0.5, 0.5, 0.5])
    # rscene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    # rscene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
    # rscene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
    # rscene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

    camera_mount_actor = scene.create_actor_builder().build_kinematic()
    camera = scene.add_mounted_camera(
        name="camera",
        actor=camera_mount_actor,
        pose=sapien.Pose(),  # relative to the mounted actor
        width=width,
        height=height,
        fovx=np.deg2rad(35.0),
        fovy=np.deg2rad(35.0),
        near=0.1,
        far=100.0,
    )

    forward = -cam_pos / np.linalg.norm(cam_pos)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)

    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos
    camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))

    scene.step()
    scene.update_render()
    camera.take_picture()

    return scene, camera, engine, robot


def render_rgb_image(camera):
    rgba = camera.get_float_texture('Color')
    rgb = rgba[:, :, :3]
    rgb_img = (rgb * 255).clip(0, 255).astype("uint8")
    return rgb_img


def render_depth_map(camera):
    position = camera.get_float_texture('Position')
    depth_map = -position[..., 2]
    return depth_map


def get_visid2gapart_mapping_dict(scene: sapien.Scene, linkId2catName, target_parts_list):
    # map visual id to instance name
    visId2instName = {}
    # map instance name to category id(index+1, 0 for others)
    instName2catId = {}
    for articulation in scene.get_all_articulations():
        for link in articulation.get_links():
            link_name = link.get_name()
            if link_name == 'base':
                continue
            link_id = int(link_name.split('_')[-1]) + 1
            for visual in link.get_visual_bodies():
                visual_name = visual.get_name()
                if visual_name.find('handle') != -1 and linkId2catName[link_id].find('handle') == -1:
                    # visial name handle; link name not handle: fixed handle!
                    inst_name = link_name + ':' + linkId2catName[link_id] + '/' + visual_name.split(
                        '-')[0] + ':' + 'fixed_handle'
                    visual_id = visual.get_visual_id()
                    visId2instName[visual_id] = inst_name
                    if inst_name not in instName2catId.keys():
                        instName2catId[inst_name] = target_parts_list.index('fixed_handle') + 1
                elif linkId2catName[link_id] in target_parts_list:
                    inst_name = link_name + ':' + linkId2catName[link_id]
                    visual_id = visual.get_visual_id()
                    visId2instName[visual_id] = inst_name
                    if inst_name not in instName2catId.keys():
                        instName2catId[inst_name] = target_parts_list.index(linkId2catName[link_id]) + 1
                else:
                    inst_name = 'others'
                    visual_id = visual.get_visual_id()
                    visId2instName[visual_id] = inst_name
                    if inst_name not in instName2catId.keys():
                        instName2catId[inst_name] = 0
    return visId2instName, instName2catId


def render_sem_ins_seg_map(scene: sapien.Scene, camera, link_pose_dict, depth_map, eps=1e-6):
    vis_id_to_link_name = {}
    for articulation in scene.get_all_articulations():
        for link in articulation.get_links():
            link_name = link.get_name()
            if link_name not in link_pose_dict:
                continue
            for visual in link.get_visual_bodies():
                visual_id = visual.get_visual_id()
                vis_id_to_link_name[visual_id] = link_name
    
    seg_labels = camera.get_uint32_texture("Segmentation")
    seg_labels_by_visual_id = seg_labels[..., 0].astype(np.uint16)  # H x W, save each pixel's visual id
    height, width = seg_labels_by_visual_id.shape

    sem_seg_map = np.ones((height, width), dtype=np.int32) * (-1) # -2 for background, -1 for others, 0~N-1 for N categories
    ins_seg_map = np.ones((height, width), dtype=np.int32) * (-1) # -2 for background, -1 for others, 0~M-1 for M instances

    valid_linkName_to_instId_mapping = {}
    part_ins_cnt = 0
    for link_name in link_pose_dict.keys():
        mask = np.zeros((height, width), dtype=np.int32)
        for vis_id in vis_id_to_link_name.keys():
            if vis_id_to_link_name[vis_id] == link_name:
                mask += (seg_labels_by_visual_id == vis_id).astype(np.int32)
        area = int(sum(sum(mask > 0)))
        if area == 0:
            continue
        sem_seg_map[mask > 0] = link_pose_dict[link_name]['category_id']
        ins_seg_map[mask > 0] = part_ins_cnt
        valid_linkName_to_instId_mapping[link_name] = part_ins_cnt
        part_ins_cnt += 1
    
    empty_mask = abs(depth_map) < eps
    sem_seg_map[empty_mask] = -2
    ins_seg_map[empty_mask] = -2

    return sem_seg_map, ins_seg_map, valid_linkName_to_instId_mapping


def add_background_color_for_image(rgb_image, depth_map, background_rgb, eps=1e-6):
    background_mask = abs(depth_map) < eps
    rgb_image[background_mask] = background_rgb
    
    return rgb_image


def get_camera_pos_mat(camera):
    K = camera.get_camera_matrix()[:3, :3]
    Rtilt = camera.get_model_matrix()
    Rtilt_rot = Rtilt[:3, :3] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    Rtilt_trl = Rtilt[:3, 3]
    
    return K, Rtilt_rot, Rtilt_trl


def merge_joint_qpos(joint_qpos_dict, new_joint_dict, old_joint_dict):
    old_joint_qpos_dict = {}
    for joint_name in new_joint_dict:
        if joint_name not in old_joint_dict:
            assert new_joint_dict[joint_name]['type'] == 'fixed'
            continue
        old_joint_qpos_dict[joint_name] = joint_qpos_dict[joint_name]
    for joint_name in old_joint_dict:
        assert joint_name in old_joint_qpos_dict
    return old_joint_qpos_dict


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

