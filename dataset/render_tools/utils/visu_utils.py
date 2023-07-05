import os
from os.path import join as pjoin
import math
import numpy as np
import open3d as o3d
import transforms3d.euler as t
import matplotlib.pyplot as plt
import cv2
from PIL import Image


def save_image(img_array, save_path, filename):
    img = Image.fromarray(img_array)
    img.save(pjoin(save_path, '{}.png'.format(filename)))
    print('{} saved!'.format(filename))


def visu_depth_map(depth_map, eps=1e-6):
    object_mask = (abs(depth_map) >= eps)
    empty_mask = (abs(depth_map) < eps)
    new_map = depth_map - depth_map[object_mask].min()
    new_map = new_map / new_map.max()
    new_map = np.clip(new_map * 255, 0, 255).astype('uint8')
    colored_depth_map = cv2.applyColorMap(new_map, cv2.COLORMAP_JET)
    colored_depth_map[empty_mask] = np.array([0, 0, 0])
    return colored_depth_map


def visu_2D_seg_map(seg_map):
    H, W = seg_map.shape
    seg_image = np.zeros((H, W, 3)).astype("uint8")

    cmap = plt.cm.get_cmap('tab20', 20)
    cmap = cmap.colors[:, 0:3]
    cmap = (cmap * 255).clip(0, 255).astype("uint8")

    for y in range(0, H):
        for x in range(0, W):
            if seg_map[y, x] == -2:
                continue
            if seg_map[y, x] == -1:
                seg_image[y, x] = cmap[14]
            else:
                seg_image[y, x] = cmap[int(seg_map[y, x]) % 20]

    return seg_image


def visu_3D_bbox_semantic(rgb_image, bboxes_pose_dict, meta):
    image = np.copy(rgb_image)

    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    K = np.array(meta['camera_intrinsic']).reshape(3, 3)

    cmap = plt.cm.get_cmap('tab20', 20)
    cmap = cmap.colors[:, 0:3]
    cmap = (cmap * 255).clip(0, 255).astype("uint8")
    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]

    for link_name, part_dict in bboxes_pose_dict.items():
        category_id = part_dict['category_id']
        bbox = part_dict['bbox']
        bbox_camera = (bbox - Rtilt_trl) @ Rtilt_rot
        color = tuple(int(x) for x in cmap[category_id])
        for line in lines:
            x_start = int(bbox_camera[line[0], 0] * K[0][0] / bbox_camera[line[0], 2] + K[0][2])
            y_start = int(bbox_camera[line[0], 1] * K[1][1] / bbox_camera[line[0], 2] + K[1][2])
            x_end = int(bbox_camera[line[1], 0] * K[0][0] / bbox_camera[line[1], 2] + K[0][2])
            y_end = int(bbox_camera[line[1], 1] * K[1][1] / bbox_camera[line[1], 2] + K[1][2])
            start = (x_start, y_start)
            end = (x_end, y_end)
            thickness = 2
            linetype = 4
            cv2.line(image, start, end, color, thickness, linetype)
    return image


def visu_3D_bbox_pose_in_color(rgb_image, bboxes_pose_dict, meta):
    image = np.copy(rgb_image)

    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    K = np.array(meta['camera_intrinsic']).reshape(3, 3)

    cmap = plt.cm.get_cmap('tab20', 20)
    cmap = cmap.colors[:, 0:3]
    cmap = (cmap * 255).clip(0, 255).astype("uint8")
    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [
        cmap[0], cmap[2], cmap[4], cmap[6], cmap[8], cmap[10], cmap[12], cmap[16], cmap[14], cmap[14], cmap[14],
        cmap[14]
    ]

    for link_name, part_dict in bboxes_pose_dict.items():
        category_id = part_dict['category_id']
        bbox = part_dict['bbox']
        bbox_camera = (bbox - Rtilt_trl) @ Rtilt_rot
        for i, line in enumerate(lines):
            x_start = int(bbox_camera[line[0], 0] * K[0][0] / bbox_camera[line[0], 2] + K[0][2])
            y_start = int(bbox_camera[line[0], 1] * K[1][1] / bbox_camera[line[0], 2] + K[1][2])
            x_end = int(bbox_camera[line[1], 0] * K[0][0] / bbox_camera[line[1], 2] + K[0][2])
            y_end = int(bbox_camera[line[1], 1] * K[1][1] / bbox_camera[line[1], 2] + K[1][2])
            start = (x_start, y_start)
            end = (x_end, y_end)
            thickness = 2
            linetype = 4
            color = tuple(int(x) for x in colors[i])
            cv2.line(image, start, end, color, thickness, linetype)
    return image


def visu_NPCS_map(npcs_map, ins_seg_map):
    npcs_image = npcs_map + np.array([0.5, 0.5, 0.5])
    assert (npcs_image > 0).all(), 'NPCS map error!'
    assert (npcs_image < 1).all(), 'NPCS map error!'
    empty_mask = (ins_seg_map == -2)
    npcs_image[empty_mask] = np.array([0, 0, 0])
    npcs_image = (np.clip(npcs_image, 0, 1) * 255).astype('uint8')

    return npcs_image


def get_recovery_whole_point_cloud_camera(rgb_image, depth_map, meta, eps=1e-6):
    height, width = depth_map.shape
    K = meta['camera_intrinsic']
    K = np.array(K).reshape(3, 3)

    point_cloud = []
    per_point_rgb = []

    for y_ in range(height):
        for x_ in range(width):
            if abs(depth_map[y_][x_]) < eps:
                continue
            z_new = float(depth_map[y_][x_])
            x_new = (x_ - K[0][2]) * z_new / K[0][0]
            y_new = (y_ - K[1][2]) * z_new / K[1][1]
            point_cloud.append([x_new, y_new, z_new])
            per_point_rgb.append([
                float(rgb_image[y_][x_][0]) / 255,
                float(rgb_image[y_][x_][1]) / 255,
                float(rgb_image[y_][x_][2]) / 255
            ])

    point_cloud = np.array(point_cloud)
    per_point_rgb = np.array(per_point_rgb)

    return point_cloud, per_point_rgb


def get_recovery_part_point_cloud_camera(rgb_image, depth_map, mask, meta, eps=1e-6):
    height, width = depth_map.shape
    K = meta['camera_intrinsic']
    K = np.array(K).reshape(3, 3)

    point_cloud = []
    per_point_rgb = []

    for y_ in range(height):
        for x_ in range(width):
            if abs(depth_map[y_][x_]) < eps:
                continue
            if not mask[y_][x_]:
                continue
            z_new = float(depth_map[y_][x_])
            x_new = (x_ - K[0][2]) * z_new / K[0][0]
            y_new = (y_ - K[1][2]) * z_new / K[1][1]
            point_cloud.append([x_new, y_new, z_new])
            per_point_rgb.append([
                float(rgb_image[y_][x_][0]) / 255,
                float(rgb_image[y_][x_][1]) / 255,
                float(rgb_image[y_][x_][2]) / 255
            ])

    point_cloud = np.array(point_cloud)
    per_point_rgb = np.array(per_point_rgb)

    return point_cloud, per_point_rgb


def draw_bbox_in_3D_semantic(bbox, category_id):
    cmap = plt.cm.get_cmap('tab20', 20)
    cmap = cmap.colors[:, 0:3]

    points = []
    for i in range(bbox.shape[0]):
        points.append(bbox[i].reshape(-1).tolist())
    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    # Use the same color for all lines
    colors = [cmap[category_id] for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def draw_bbox_in_3D_pose_color(bbox):
    cmap = plt.cm.get_cmap('tab20', 20)
    cmap = cmap.colors[:, 0:3]

    points = []
    for i in range(bbox.shape[0]):
        points.append(bbox[i].reshape(-1).tolist())
    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    # Use the same color for all lines
    colors = [
        cmap[0], cmap[2], cmap[4], cmap[6], cmap[8], cmap[10], cmap[12], cmap[16], cmap[14], cmap[14], cmap[14],
        cmap[14]
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def visu_point_cloud_with_bbox_semantic(rgb_image, depth_map, bbox_pose_dict, meta):

    point_cloud, per_point_rgb = get_recovery_whole_point_cloud_camera(rgb_image, depth_map, meta)
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    point_cloud_world = point_cloud @ Rtilt_rot.T + Rtilt_trl

    vis_list = []
    for link_name, part_dict in bbox_pose_dict.items():
        category_id = part_dict['category_id']
        bbox = part_dict['bbox']
        bbox_t = draw_bbox_in_3D_semantic(bbox, category_id)
        vis_list.append(bbox_t)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_world)
    pcd.colors = o3d.utility.Vector3dVector(per_point_rgb)
    vis_list.append(pcd)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis_list.append(coord_frame)
    o3d.visualization.draw_geometries(vis_list)


def visu_point_cloud_with_bbox_pose_color(rgb_image, depth_map, bbox_pose_dict, meta):

    point_cloud, per_point_rgb = get_recovery_whole_point_cloud_camera(rgb_image, depth_map, meta)
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    point_cloud_world = point_cloud @ Rtilt_rot.T + Rtilt_trl

    vis_list = []
    for link_name, part_dict in bbox_pose_dict.items():
        category_id = part_dict['category_id']
        bbox = part_dict['bbox']
        bbox_t = draw_bbox_in_3D_pose_color(bbox)
        vis_list.append(bbox_t)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_world)
    pcd.colors = o3d.utility.Vector3dVector(per_point_rgb)
    vis_list.append(pcd)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis_list.append(coord_frame)
    o3d.visualization.draw_geometries(vis_list)


def visu_NPCS_in_3D_with_bbox_pose_color(rgb_image, depth_map, anno_dict, meta):
    ins_seg_map = anno_dict['instance_segmentation']
    bbox_pose_dict = anno_dict['bbox_pose_dict']
    npcs_map = anno_dict['npcs_map']
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)

    for link_name, part_dict in bbox_pose_dict.items():
        category_id = part_dict['category_id']
        instance_id = part_dict['instance_id']
        bbox_world = part_dict['bbox']
        mask = (ins_seg_map == instance_id)
        point_cloud, per_point_rgb = get_recovery_part_point_cloud_camera(rgb_image, depth_map, mask, meta)
        point_cloud_world = point_cloud @ Rtilt_rot.T + Rtilt_trl
        RTS_param = part_dict['pose_RTS_param']
        R, T, S, scaler = RTS_param['R'], RTS_param['T'], RTS_param['S'], RTS_param['scaler']
        point_cloud_canon = npcs_map[mask]
        bbox_canon = ((bbox_world - T) / scaler) @ R.T

        vis_list = []
        vis_list.append(draw_bbox_in_3D_pose_color(bbox_world))
        vis_list.append(draw_bbox_in_3D_pose_color(bbox_canon))

        pcd_1 = o3d.geometry.PointCloud()
        pcd_1.points = o3d.utility.Vector3dVector(point_cloud_world)
        pcd_1.colors = o3d.utility.Vector3dVector(per_point_rgb)
        vis_list.append(pcd_1)
        pcd_2 = o3d.geometry.PointCloud()
        pcd_2.points = o3d.utility.Vector3dVector(point_cloud_canon)
        pcd_2.colors = o3d.utility.Vector3dVector(per_point_rgb)
        vis_list.append(pcd_2)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        vis_list.append(coord_frame)
        o3d.visualization.draw_geometries(vis_list)

