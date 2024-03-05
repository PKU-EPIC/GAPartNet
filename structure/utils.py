import numpy as np
import glob
from einops import rearrange, repeat
import torch.nn.functional as Fun
import torch
import pickle
import cv2
# import open3d
from PIL import Image
import os
from sklearn.neighbors import KNeighborsClassifier
from os.path import join as pjoin
import importlib
# from gapartnet.structure.point_cloud import PointCloud
# from gapartnet.dataset.gapartnet import apply_voxelization
# from gapartnet.misc.pose_fitting import estimate_pose_from_npcs
# from gapartnet.tools.visu_utils import OBJfile2points, map2image, save_point_cloud_to_ply, \
#     WorldSpaceToBallSpace, FindMaxDis, draw_bbox_old, COLOR20, \
#     OTHER_COLOR, HEIGHT, WIDTH, EDGE, K, font, fontScale, fontColor,thickness, lineType 
from scipy.spatial.transform import Rotation as R

def draw_bbox_from_world(img, bbox_list, K, camera2world_translation, world2camera_rotation):
    point2images = []
    for i,bbox in enumerate(bbox_list):
        if len(bbox) == 0:
            continue
        bbox = np.array(bbox)
        if camera2world_translation is not None:
            assert world2camera_rotation is not None
            bbox = (bbox - camera2world_translation) @ world2camera_rotation
        point2image = np.concatenate(
            ((np.around(bbox[:,0] * K[0][0] / bbox[:,2] + K[0][2])).astype(dtype=int).reshape(-1,1),
            (np.around(bbox[:,1] * K[0][0] / bbox[:,2] + K[0][2])).astype(dtype=int).reshape(-1,1)),
        axis=1)
        point2images.append(point2image)
        cl = [255,0,255]

        cv2.line(img,point2image[0],point2image[1],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[0],point2image[3],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[0],point2image[4],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[1],point2image[2],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[1],point2image[5],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[2],point2image[3],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[2],point2image[6],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        # cv2.line(img,point2image[3],point2image[4],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[3],point2image[7],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[4],point2image[5],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[4],point2image[7],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[5],point2image[6],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[6],point2image[7],color=(0,0,255),thickness=3)
        cv2.line(img,point2image[3],point2image[7],color=(255,0,0),thickness=3)
        cv2.line(img,point2image[4],point2image[7],color=(0,255,0),thickness=3)
    return img, point2images

def draw_bbox(img, bbox_list, trans=None, K=None, camera2world_translation=None, world2camera_rotation=None):
    for i,bbox in enumerate(bbox_list):
        if len(bbox) == 0:
            continue
        bbox = np.array(bbox)
        if trans is not None:
            bbox = bbox * trans[0]+trans[1:4]
        if camera2world_translation is not None:
            assert world2camera_rotation is not None
            bbox = (bbox - camera2world_translation) @ world2camera_rotation
        point2image = []
        for pts in bbox:
            x = pts[0]
            y = pts[1]
            z = pts[2]
            x_new = (np.around(x * K[0][0] / z + K[0][2])).astype(dtype=int)
            y_new = (np.around(y * K[1][1] / z + K[1][2])).astype(dtype=int)
            point2image.append([x_new, y_new])
        cl = [255,0,255]

        cv2.line(img,point2image[0],point2image[1],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[0],point2image[2],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[0],point2image[3],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[1],point2image[4],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[1],point2image[5],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[2],point2image[6],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[6],point2image[3],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[4],point2image[7],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[5],point2image[7],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[3],point2image[5],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[2],point2image[4],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[6],point2image[7],color=(int(cl[0]),int(cl[1]),int(cl[2])),thickness=2)
        cv2.line(img,point2image[0],point2image[1],color=(0,0,255),thickness=3) # red
        cv2.line(img,point2image[0],point2image[3],color=(255,0,0),thickness=3) # green
        cv2.line(img,point2image[0],point2image[2],color=(0,255,0),thickness=3) # blue
    return img

def create_transformation_matrix(position, quaternion):
    # Create a 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    
    # Convert the quaternion to a rotation matrix
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    # rotation_matrix = R.from_quat(quaternion).inv().as_matrix()
    
    # Update the rotation part of the transformation matrix
    transformation_matrix[:3, :3] = rotation_matrix
    
    # Update the translation part of the transformation matrix
    transformation_matrix[:3, 3] = position
    
    return transformation_matrix

def transform_point_cloud(point_cloud, transformation_matrix):
    # Transform each point in the point cloud
    transformed_point_cloud = (transformation_matrix @ np.append(point_cloud, np.ones((1, point_cloud.shape[1])), axis=0)).T
    
    return transformed_point_cloud[:, :3]

def read_pcs_from_ply(path):
    pcd = open3d.io.read_point_cloud(path)
    return np.array(pcd.points), np.array(pcd.colors)

def _inference_perception_model(perception_model, points_list, name = "test",others = None, use_sam_masks = False):
    device = perception_model.device
    points_list = [torch.tensor(points, dtype=torch.float32) for points in points_list]
    pcs = []
    for points in points_list:
        # others["sam_masks"] = self.sam_pred_masks
        # others["sam_GAPart_ids"] = self.sam_GAPart_id_pred
        if use_sam_masks:
            assert others is not None
            pc_masks = others["sam_masks_pc"]
            mask_ids = others["sam_GAPart_ids_pc"]
            mask_labels = others["sam_GAPart_labels_pc"]
            pc = PointCloud(
                pc_id=name,
                points=points,
                pc_masks = [torch.tensor(m, device = device) for m in pc_masks],
                mask_ids = [id for id in mask_ids],
                mask_labels = mask_labels,
            )
        else:
            pc = PointCloud(
                pc_id=name,
                points=points,
                pc_masks = [],
                mask_ids = [],
                mask_labels = [],
            )
        pc = apply_voxelization(
            pc,  voxel_size=(1. / 100, 1. / 100, 1. / 100)
        )
        pc = pc.to(device=device)
        pcs.append(pc)

    with torch.no_grad():
        scene_ids, segmentations, proposals, _ = perception_model(pcs)
    npcs_maps = torch.ones(pcs[0].points.shape[0],3, device = pcs[0].points.device)* (230./255.)
    if proposals is not None:
        valid_index = torch.where(proposals.valid_mask==True)[0][proposals.sorted_indices.long()]
        npcs_preds = proposals.npcs_preds
        npcs_maps[valid_index] = npcs_preds

        proposal_sem_pred = proposals.pt_sem_classes
    sem_preds = segmentations.sem_preds
    if proposals is not None:
        batch_indices = proposals.batch_indices
        proposal_offsets = proposals.proposal_offsets
        num_points_per_proposal = proposals.num_points_per_proposal
        num_proposals = num_points_per_proposal.shape[0]
        score_preds= proposals.score_preds
        pt_xyz = proposals.pt_xyz

        indices = torch.arange(sem_preds.shape[0], dtype=torch.int64, device=device)
        proposal_indices = indices[proposals.valid_mask][proposals.sorted_indices]
        
    bboxes = [[] for _ in range(len(points_list))]
    if proposals is not None:
        for i in range(num_proposals):
            offset_begin = proposal_offsets[i].item()
            offset_end = proposal_offsets[i + 1].item()

            batch_idx = batch_indices[offset_begin]
            xyz_i = pt_xyz[offset_begin:offset_end]
            npcs_i = npcs_preds[offset_begin:offset_end]

            npcs_i = npcs_i - 0.5
            if xyz_i.shape[0]<=4:
                continue
            bbox_xyz, scale, rotation, translation, out_transform, best_inlier_idx = estimate_pose_from_npcs(xyz_i.cpu().numpy(), npcs_i.cpu().numpy())
            if scale[0] == None:
                continue
            bboxes[batch_idx].append(bbox_xyz.tolist())
    try:
        return bboxes, sem_preds, npcs_maps, proposal_indices, proposal_offsets, proposal_sem_pred
    except:
        return bboxes, sem_preds, npcs_maps, None, None, None
    
    
def _estimate_pose_with_masks(perception_model, points_list, name = "test",others = None):
    device = perception_model.device
    points_list = [torch.tensor(points, dtype=torch.float32) for points in points_list]
    pcs = []
    for points in points_list:
        # others["sam_masks"] = self.sam_pred_masks
        # others["sam_GAPart_ids"] = self.sam_GAPart_id_pred
        assert others is not None
        pc_masks = others["sam_masks_pc"]
        mask_ids = others["sam_GAPart_ids_pc"]
        mask_labels = others["sam_GAPart_labels_pc"]
        pc = PointCloud(
            pc_id=name,
            points=points,
            pc_masks = [torch.tensor(m, device = device) for m in pc_masks],
            mask_ids = [id for id in mask_ids],
            mask_labels = mask_labels,
        )

        pc = apply_voxelization(
            pc,  voxel_size=(1. / 100, 1. / 100, 1. / 100)
        )
        pc = pc.to(device=device)
        pcs.append(pc)

    with torch.no_grad():
        pc_ids, proposals = perception_model.estimate_pose_from_mask(pcs)
    # npcs_maps = torch.ones(pcs[0].points.shape[0],3, device = pcs[0].points.device)* (230./255.)
    if proposals is not None:
        # valid_index = torch.where(proposals.valid_mask==True)[0][proposals.sorted_indices.long()]
        npcs_preds = proposals.npcs_preds
        # npcs_maps[valid_index] = npcs_preds

        # proposal_sem_pred = proposals.pt_sem_classes
    # sem_preds = segmentations.sem_preds
    if proposals is not None:
        batch_indices = proposals.batch_indices
        proposal_offsets = proposals.proposal_offsets
        num_points_per_proposal = proposals.num_points_per_proposal
        num_proposals = num_points_per_proposal.shape[0]
    #     score_preds= proposals.score_preds
        pt_xyz = proposals.pt_xyz

    #     indices = torch.arange(sem_preds.shape[0], dtype=torch.int64, device=device)
    #     proposal_indices = indices[proposals.valid_mask][proposals.sorted_indices]
        
    bboxes = [[]]

    if proposals is not None:
        for i in range(num_proposals):
            offset_begin = proposal_offsets[i].item()
            offset_end = proposal_offsets[i + 1].item()

            batch_idx = 0
            xyz_i = pt_xyz[offset_begin:offset_end]
            npcs_i = npcs_preds[offset_begin:offset_end]

            npcs_i = npcs_i - 0.5
            if xyz_i.shape[0]<=4:
                continue
            bbox_xyz, scale, rotation, translation, out_transform, best_inlier_idx = estimate_pose_from_npcs(xyz_i.cpu().numpy(), npcs_i.cpu().numpy())
            if scale[0] == None:
                continue
            bboxes[batch_idx].append(bbox_xyz.tolist())
    try:
        return bboxes, None, proposal_offsets
    except:
        return bboxes, None, None
    


def _inference_perception_model_with_masks(perception_model, points_list, masks, labels):
    device = perception_model.device
    points_list = [torch.tensor(points, dtype=torch.float32) for points in points_list]
    pcs = []
    for points in points_list:
        pc = PointCloud(
            pc_id="test",
            points=points,
        )
        pc = apply_voxelization(
            pc,  voxel_size=(1. / 100, 1. / 100, 1. / 100)
        )
        pc = pc.to(device=device)
        pcs.append(pc)

    with torch.no_grad():
        scene_ids, segmentations, proposals, proposal_sem_labels = perception_model.forward_with_masks(pcs, masks, labels)
    npcs_maps = torch.ones(pcs[0].points.shape[0],3, device = pcs[0].points.device)* (230./255.)

    if proposals is not None:
        valid_index = torch.where(proposals.valid_mask==True)[0][proposals.sorted_indices.long()]
        npcs_preds = proposals.npcs_preds
        npcs_maps[valid_index] = npcs_preds

    sem_preds = segmentations.sem_preds
    if proposals is not None:
        pt_xyz = proposals.pt_xyz
        batch_indices = proposals.batch_indices
        proposal_offsets = proposals.proposal_offsets
        num_points_per_proposal = proposals.num_points_per_proposal
        num_proposals = num_points_per_proposal.shape[0]
        score_preds= proposals.score_preds

        indices = torch.arange(sem_preds.shape[0], dtype=torch.int64, device=device)
        proposal_indices = indices[proposals.valid_mask][proposals.sorted_indices]
        
    bboxes = [[] for _ in range(len(points_list))]
    if proposals is not None:
        for i in range(num_proposals):
            offset_begin = proposal_offsets[i].item()
            offset_end = proposal_offsets[i + 1].item()

            batch_idx = batch_indices[offset_begin]
            xyz_i = pt_xyz[offset_begin:offset_end]
            npcs_i = npcs_preds[offset_begin:offset_end]

            npcs_i = npcs_i - 0.5
            if xyz_i.shape[0]<=4:
                continue
            bbox_xyz, scale, rotation, translation, out_transform, best_inlier_idx = estimate_pose_from_npcs(xyz_i.cpu().numpy(), npcs_i.cpu().numpy())
            if scale[0] == None:
                continue
            bboxes[batch_idx].append(bbox_xyz.tolist())
    try:
        return bboxes, sem_preds, npcs_maps, proposal_indices, proposal_offsets, proposal_sem_labels
    except:
        return bboxes, sem_preds, npcs_maps, None, None, None

def _load_perception_model(
        ckpt_path = "gapartnet/ckpt/all_best_7816.ckpt",
        class_path = "gapartnet.network.model.GAPartNet",
        device = "cuda",
        USE_2D_FOR_PERCEPTION = False,
    ):
    module_name = ".".join(class_path.split(".")[:-1])
    class_name = class_path.split(".")[-1]

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    net = cls.load_from_checkpoint(ckpt_path)
    net.use_2d_masks = USE_2D_FOR_PERCEPTION
    net.training_schedule = [0,0]
    net.scorenet_npcsnet_new_backbone = False
    net.freeze()
    net.eval()
    net.to(device)

    return net

def farthest_point_sample(xyz, npoint, use_cuda = True):
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
    
    if use_cuda:
        # print('Use pointnet2_cuda!')
        from pointnet2_ops.pointnet2_utils import furthest_point_sample as fps_cuda
        sampled_points_ids = fps_cuda(torch.tensor(xyz).to("cuda"), npoint)
        return sampled_points_ids
    else:
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
        sampled_points_ids = centroids
    return sampled_points_ids

def map2image(pts, rgb, K, HEIGHT, WIDTH):
    # input为每个shape的info，取第idx行
    image_rgb = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
    # K = np.array([[1268.637939453125, 0, 400, 0], [0, 1268.637939453125, 400, 0],
    #              [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

    num_point = pts.shape[0]
    # print(num_point)
    # print(pts)
    # print(rgb.shape)

    point2image = {}
    for i in range(num_point):
        x = pts[i][0]
        y = pts[i][1]
        z = pts[i][2]
        x_new = (np.around(x * K[0][0] / z + K[0][2])).astype(dtype=int)
        y_new = (np.around(y * K[1][1] / z + K[1][2])).astype(dtype=int)
        point2image[i] = (y_new, x_new)

    # 还原原始的RGB图
    for i in range(num_point):
        # print(i, point2image[i][0], point2image[i][1])
        if point2image[i][0]+1 >= HEIGHT or point2image[i][0] < 0 or point2image[i][1]+1 >= WIDTH or point2image[i][1] < 0:
            continue
        image_rgb[point2image[i][0]][point2image[i][1]] = rgb[i]
        image_rgb[point2image[i][0]+1][point2image[i][1]] = rgb[i]
        image_rgb[point2image[i][0]+1][point2image[i][1]+1] = rgb[i]
        image_rgb[point2image[i][0]][point2image[i][1]+1] = rgb[i]

    # rgb_pil = Image.fromarray(image_rgb, mode='RGB')
    # rgb_pil.save(os.path.join(save_path, f'{instance_name}_{task}.png'))
    return image_rgb


def map2image_single(pts, rgb, K, HEIGHT, WIDTH):
    # input为每个shape的info，取第idx行
    image_rgb = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
    # K = np.array([[1268.637939453125, 0, 400, 0], [0, 1268.637939453125, 400, 0],
    #              [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

    num_point = pts.shape[0]
    # print(num_point)
    # print(pts)
    # print(rgb.shape)
    
    # pts_new = pts * trans[0] + trans[1:4]
    # x_new = np.round(np.around(pts_new[:,0] * K[0][0] / (pts_new[:,2]+1e-6) + K[0][2]))
    # y_new = np.round(np.around(pts_new[:,1] * K[1][1] / (pts_new[:,2]+1e-6) + K[1][2]))
    # point2image = np.ceil(np.concatenate([y_new.reshape(-1,1), x_new.reshape(-1,1)], axis = 1)).astype(np.int32)
    # image_rgb

    point2image = {}
    for i in range(num_point):
        x = pts[i][0]
        y = pts[i][1]
        z = pts[i][2]
        x_new = (np.around(x * K[0][0] / z + K[0][2])).astype(dtype=int)
        y_new = (np.around(y * K[1][1] / z + K[1][2])).astype(dtype=int)
        point2image[i] = (y_new, x_new)

    # 还原原始的RGB图
    for i in range(num_point):
        # print(i, point2image[i][0], point2image[i][1])
        if point2image[i][0]+1 >= HEIGHT or point2image[i][0] < 0 or point2image[i][1]+1 >= WIDTH or point2image[i][1] < 0:
            continue
        image_rgb[point2image[i][0]][point2image[i][1]] = rgb[i]
        image_rgb[point2image[i][0]+1][point2image[i][1]] = rgb[i]
        image_rgb[point2image[i][0]+1][point2image[i][1]+1] = rgb[i]
        image_rgb[point2image[i][0]][point2image[i][1]+1] = rgb[i]

    # rgb_pil = Image.fromarray(image_rgb, mode='RGB')
    # rgb_pil.save(os.path.join(save_path, f'{instance_name}_{task}.png'))
    return image_rgb


def get_point_cloud(rgb_image, depth_map, sem_seg_map, ins_seg_map, K):
    point_cloud = []
    per_point_rgb = []
    per_point_sem_label = []
    per_point_ins_label = []
    per_point_idx = []

    for y_ in range(rgb_image.shape[0]):
        for x_ in range(rgb_image.shape[1]):
            # if sem_seg_map[y_, x_] == -1:
            #     continue
            z_new = float(depth_map[y_, x_])
            x_new = (x_ - K[0, 2]) * z_new / K[0, 0]
            y_new = (y_ - K[1, 2]) * z_new / K[1, 1]
            point_cloud.append([x_new, y_new, z_new])
            per_point_rgb.append((rgb_image[y_, x_] / 255.0))
            per_point_sem_label.append(sem_seg_map[y_, x_])
            per_point_ins_label.append(ins_seg_map[y_, x_])

            per_point_idx.append([y_, x_])

    return np.array(point_cloud), np.array(per_point_rgb), np.array(per_point_sem_label), np.array(
        per_point_ins_label), np.array(per_point_idx)

def save_point_cloud_to_ply(points, colors, save_name='debug.ply', save_root='.'):
    '''
    Save point cloud to ply file
    '''
    PLY_HEAD = f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    file_sting = PLY_HEAD
    for i in range(len(points)):
        file_sting += f'{points[i][0]} {points[i][1]} {points[i][2]} {int(colors[i][0])} {int(colors[i][1])} {int(colors[i][2])}\n'
    from os.path import join as pjoin
    f = open(pjoin(save_root, save_name), 'w')
    f.write(file_sting)
    f.close()

def mask_change_reso(mask, tar_reso_x, tar_reso_y):
    mask_img = mask_heat_img = cv2.applyColorMap((mask*255).astype(np.uint8), cv2.COLORMAP_JET)
    img = Image.fromarray(mask_img, 'RGB')
    img = img.resize((tar_reso_x, tar_reso_y))#, Image.ANTIALIAS
    img = np.array(img)
    mask_new = 1 - img[:,:,0].clip(0,128)/128.0
    return mask_new

def load_data_single_file(data_root = "/home/birdswimming/HW/LLM-GAPartNet/vision/data/fea_data_all_relabel.npy"):
    data = np.load(data_root, allow_pickle=True).item()
    feas = data["feas"]
    obj_codes = data["obj_codes"]
    part_ids = data["part_ids"]
    cat_ids = data["cat_ids"]
    splits = data["splits"]
    # feas, obj_codes, cat_ids, part_ids, splits = load_new_data(data_root = FEA_ROOT, type = type_fea)
    feas = np.array(feas)
    splits = np.array(splits)
    cat_ids = np.array(cat_ids)

    train_mask = splits == "train"
    train_balanced_mask = train_mask.copy()
    train_balanced_mask[...] = False
    
    train_feas = torch.tensor(feas[train_mask])
    train_cat = cat_ids[train_mask]
    train_cat = np.array([int(ii) for ii in train_cat])
    cat_ids = np.array([int(ii) for ii in cat_ids])

    return train_feas, train_cat, cat_ids, splits, feas

def KNN_classifier(train_feas, train_cat, K):
    X = train_feas.cpu().numpy()
    y = train_cat
    neigh = KNeighborsClassifier(n_neighbors=K)
    neigh.fit(X, y)
    print("finish training!")
    return neigh

def query_part_anno(anno_root, name):
    '''
    input: 
        name: object name
        anno_root: annotation data root for GAPartNet
    output: 
        annotation dict
    function:
        1. load image
        2. load annotation
        3. return list of part info:
            part_info = {
                "obj_name": name, # object name
                "ins_id": part_id, # instance id
                "sem_id": sem_id, # semantic id
                "sem_label": sem_label, # semantic label
                "npcs_map": npcs_map, # for part
                "bbox": bbox, # for part
                "mask": mask, # for part
            }
    '''
    
    # load image
    seg_anno_path = f"{anno_root}/segmentation/{name}.npz"
    npcs_anno_path = f"{anno_root}/npcs/{name}.npz"
    bbox_aano_path = f"{anno_root}/bbox/{name}.pkl"
    
    # segmentation annotation
    if os.path.exists(seg_anno_path):
        seg_anno = np.load(seg_anno_path, mmap_mode='r')
        sem_seg_anno = seg_anno['semantic_segmentation'] # (800, 800)
        ins_seg_anno = seg_anno['instance_segmentation'] # (800, 800)
    else:
        import pdb; pdb.set_trace()
    
    # bbox annotation
    with open(bbox_aano_path, 'rb') as f:
        bboxes_anno = pickle.load(f)['bboxes_with_pose'] # dict
        bboxes_array = [np.array(bbox_anno['bbox_3d']) for bbox_anno in bboxes_anno] # (n, 4)
        
    # npcs annotation
    npcs_anno = np.load(npcs_anno_path, mmap_mode='r')['npcs_map'] # (800, 800, 3)
    
    total_parts = int(ins_seg_anno.max())
    parts_info = []
    for part_id in range(total_parts):
        mask = ins_seg_anno == (part_id + 1)
        sem_id = int(sem_seg_anno[mask].max())
        assert sem_seg_anno[mask].max() == sem_seg_anno[mask].min()
        sem_label = PART_ID2NAME[sem_id]
        npcs_map = npcs_anno[mask]
        bbox = bboxes_array[part_id]
        part_info = {
            "obj_name": name,
            "ins_id": part_id,
            "sem_id": sem_id,
            "sem_label": sem_label,
            "npcs_map": npcs_map,
            "bbox": bbox,
            "mask": mask,
        }
        parts_info.append(part_info)
        print(f"{name} {part_id} {sem_id} {sem_label}")

    return parts_info

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
        return None

    if pcs.shape[0] == npoint:
        return np.arange(pcs.shape[0])

    pcs_tensor = torch.from_numpy(np.expand_dims(pcs, 0)).float()
    fps_idx_tensor = farthest_point_sample(pcs_tensor, npoint)
    fps_idx = fps_idx_tensor.cpu().numpy()[0]
    return fps_idx

def get_point2image(pts, trans, K):
    pts_new = pts * trans[0] + trans[1:4]
    x_new = np.round(np.around(pts_new[:,0] * K[0][0] / (pts_new[:,2]+1e-6) + K[0][2]))
    y_new = np.round(np.around(pts_new[:,1] * K[1][1] / (pts_new[:,2]+1e-6) + K[1][2]))
    point2image = np.ceil(np.concatenate([y_new.reshape(-1,1), x_new.reshape(-1,1)], axis = 1)).astype(np.int32)
    if not (point2image>=0).all():
        import pdb; pdb.set_trace()
    return np.array(point2image)

# PART_NAME2ID = {
#     'others':             0,
#     'line_fixed_handle':  1,
#     'round_fixed_handle': 2,
#     'hinge_knob':      3,
#     'slider_button':      4, # slider_button
#     'hinge_door':         5, # hinge_door
#     'slider_drawer':         6, # slider_lid
#     'slider_lid':         7, # slider_lid
#     'hinge_lid':          8, # hinge_lid
#     'revolute_handle':    9,
# }

PART_ID2NAME_OLD = {
    0: 'others'             ,
    1: 'line_fixed_handle'  ,
    2: 'round_fixed_handle' ,
    3: 'revolute_handle'    ,
    4: 'slider_button'      , # slider_button
    5: 'hinge_door'         , # hinge_door
    6: 'slider_drawer'         , # slider_lid
    7: 'slider_lid'         , # slider_lid
    8: 'hinge_lid'          , # hinge_lid
    9: 'hinge_knob'         ,
}

TARGET_GAPARTS = [
    'line_fixed_handle', 'round_fixed_handle', 'slider_button', 'hinge_door', 'slider_drawer',
    'slider_lid', 'hinge_lid', 'hinge_knob', 'hinge_handle'
]
PART_ID2NAME = {
    0: 'others'             ,
    1: 'line_fixed_handle'  ,
    2: 'round_fixed_handle' ,
    3: 'slider_button'      ,
    4: 'hinge_door'         ,
    5: 'slider_drawer'      ,
    6: 'slider_lid'         ,
    7: 'hinge_lid'          ,
    8: 'hinge_knob'         ,
    9: 'revolute_handle'    ,
}

COLOR20 = np.array(
    [[230, 230, 230], [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
     [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
     [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
     [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190]])

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (0,0,0)
thickness              = 2
lineType               = 3

