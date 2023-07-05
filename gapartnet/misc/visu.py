import torch
import numpy as np
import yaml
from os.path import join as pjoin
import os
import argparse
import sys
sys.path.append(sys.path[0] + "/..")
import importlib
from structure.point_cloud import PointCloud
from dataset.gapartnet import apply_voxelization
from misc.pose_fitting import estimate_pose_from_npcs
import cv2
from typing import List
import glob
from misc.visu_util import OBJfile2points, map2image, save_point_cloud_to_ply, \
    WorldSpaceToBallSpace, FindMaxDis, draw_bbox_old, draw_bbox, COLOR20, \
    OTHER_COLOR, HEIGHT, WIDTH, EDGE, K, font, fontScale, fontColor,thickness, lineType 

def process_gapartnetfile(GAPARTNET_DATA_ROOT, name, split = "train"):
    data_path = f"{GAPARTNET_DATA_ROOT}/{split}/pth/{name}.pth"
    trans_path = f"{GAPARTNET_DATA_ROOT}/{split}/meta/{name}.txt"

    pc, rgb, semantic_label, instance_label, npcs_map = torch.load(data_path)
    
    trans = np.loadtxt(trans_path)
    xyz = pc * trans[0] + trans[1:4]

    # save_point_cloud_to_ply(xyz, rgb*255, data_path.split("/")[-1].split(".")[0]+"_preinput.ply")
    # save_point_cloud_to_ply(pc, rgb*255, data_path.split("/")[-1].split(".")[0]+"_input.ply")
    points_input = torch.cat((torch.tensor(pc),torch.tensor(rgb)), dim = 1)
    return points_input, trans, semantic_label, instance_label, npcs_map
              

def visualize_gapartnet(
    SAVE_ROOT, 
    GAPARTNET_DATA_ROOT,
    RAW_IMG_ROOT,
    save_option: List = [], 
    name: str = "pc",
    split: str = "",
    bboxes: np.ndarray = None, # type: ignore
    sem_preds: np.ndarray = None, # type: ignore
    ins_preds: np.ndarray = None, # type: ignore
    npcs_preds: np.ndarray = None, # type: ignore
    have_proposal = True, 
    save_detail = False,
):
    
    final_save_root = f"{SAVE_ROOT}/{split}"
    save_root = f"{SAVE_ROOT}/{split}/{name}"
    os.makedirs(final_save_root, exist_ok=True)
    if save_detail:
        os.makedirs(f"{save_root}", exist_ok=True)
    final_img = np.ones((3 * (HEIGHT + EDGE) + EDGE, 4 * (WIDTH + EDGE) + EDGE, 3), dtype=np.uint8) * 255
    
    points_input, trans, semantic_label, instance_label, npcs_map = process_gapartnetfile(GAPARTNET_DATA_ROOT, name, split)

    points_input = points_input.numpy()
    xyz_input = points_input[:,:3]
    rgb = points_input[:,3:6]
    xyz = xyz_input * trans[0] + trans[1:4]
    pc_img = map2image(xyz, rgb*255.0)
    pc_img = cv2.cvtColor(pc_img, cv2.COLOR_BGR2RGB)
    
    if "raw" in save_option:
        raw_img_path = f"{RAW_IMG_ROOT}/{name}.png"
        if os.path.exists(raw_img_path):
            raw_img = cv2.imread(raw_img_path)
            if save_detail:
                cv2.imwrite(f"{save_root}/raw.png", raw_img)
            X_START = EDGE
            Y_START = EDGE
            final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = raw_img
            text = "raw"
            cv2.putText(final_img, text, 
                (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
                font, fontScale, fontColor, thickness, lineType)
    if "pc" in save_option:
        if save_detail:
            cv2.imwrite(f"{save_root}/pc.png", pc_img)
        X_START = EDGE + (HEIGHT + EDGE)
        Y_START = EDGE
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = pc_img
        text = "pc"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "sem_pred" in save_option:
        sem_pred_img = map2image(xyz, COLOR20[sem_preds])
        sem_pred_img = cv2.cvtColor(sem_pred_img, cv2.COLOR_BGR2RGB)
        if save_detail:
            cv2.imwrite(f"{save_root}/sem_pred.png", sem_pred_img)
        X_START = EDGE + (WIDTH + EDGE)
        Y_START = EDGE + (HEIGHT + EDGE)
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = sem_pred_img
        text = "sem_pred"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "ins_pred" in save_option:
        # ins_pred_color = np.ones_like(xyz) * 230
        # if have_proposal:
        #     for ins_i in range(len(proposal_offsets) - 1):
        #         ins_pred_color[proposal_indices[proposal_offsets[ins_i]:proposal_offsets[ins_i + 1]]] = COLOR20[ins_i%19 + 1]
        # import pdb; pdb.set_trace()
        ins_pred_img = map2image(xyz, COLOR20[(ins_preds%20).astype(np.int_)])
        ins_pred_img = cv2.cvtColor(ins_pred_img, cv2.COLOR_BGR2RGB)
        if save_detail:
            cv2.imwrite(f"{save_root}/ins_pred.png", ins_pred_img)    
        X_START = EDGE + (WIDTH + EDGE) * 1
        Y_START = EDGE + (HEIGHT + EDGE) * 2
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = ins_pred_img
        text = "ins_pred"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "npcs_pred" in save_option:
        npcs_pred_img = map2image(xyz, npcs_preds*255.0)
        npcs_pred_img = cv2.cvtColor(npcs_pred_img, cv2.COLOR_BGR2RGB)
        if save_detail:
            cv2.imwrite(f"{save_root}/npcs_pred.png", npcs_pred_img)
        X_START = EDGE + (WIDTH + EDGE) * 1
        Y_START = EDGE + (HEIGHT + EDGE) * 3
        # import pdb
        # pdb.set_trace()
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = npcs_pred_img
        text = "npcs_pred"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "bbox_pred" in save_option:
        img_bbox_pred = pc_img.copy()
        draw_bbox(img_bbox_pred, bboxes, trans)
        if save_detail:
            cv2.imwrite(f"{save_root}/bbox_pred.png", img_bbox_pred)
        X_START = EDGE + (WIDTH + EDGE) * 2
        Y_START = EDGE + (HEIGHT + EDGE) * 2
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = img_bbox_pred
        text = "bbox_pred"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "bbox_pred_pure" in save_option:
        img_empty = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
        draw_bbox(img_empty, bboxes, trans)
        if save_detail:
            cv2.imwrite(f"{save_root}/bbox_pure.png", img_empty)
        X_START = EDGE + (WIDTH + EDGE) * 2
        Y_START = EDGE + (HEIGHT + EDGE) * 3
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = img_empty
        text = "bbox_pred_pure"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "sem_gt" in save_option:
        sem_gt = semantic_label
        sem_gt_img = map2image(xyz, COLOR20[sem_gt])
        sem_gt_img = cv2.cvtColor(sem_gt_img, cv2.COLOR_BGR2RGB)
        if save_detail:
            cv2.imwrite(f"{save_root}/sem_gt.png", sem_gt_img)      
        X_START = EDGE + (WIDTH + EDGE) * 0
        Y_START = EDGE + (HEIGHT + EDGE) * 1
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = sem_gt_img
        text = "sem_gt"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "ins_gt" in save_option:
        ins_gt = instance_label
        ins_color = COLOR20[ins_gt%19 + 1]
        ins_color[np.where(ins_gt == -100)] = 230
        ins_gt_img = map2image(xyz, ins_color)
        
        ins_gt_img = cv2.cvtColor(ins_gt_img, cv2.COLOR_BGR2RGB)
        if save_detail:
            cv2.imwrite(f"{save_root}/ins_gt.png", ins_gt_img)      
        X_START = EDGE + (WIDTH + EDGE) * 0
        Y_START = EDGE + (HEIGHT + EDGE) * 2
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = ins_gt_img
        text = "ins_gt"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "npcs_gt" in save_option:
        npcs_gt = npcs_map + 0.5
        npcs_gt_img = map2image(xyz, npcs_gt*255.0)
        npcs_gt_img = cv2.cvtColor(npcs_gt_img, cv2.COLOR_BGR2RGB)
        if save_detail:
            cv2.imwrite(f"{save_root}/npcs_gt.png", npcs_gt_img)
        X_START = EDGE + (WIDTH + EDGE) * 0
        Y_START = EDGE + (HEIGHT + EDGE) * 3
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = npcs_gt_img
        text = "npcs_gt"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "bbox_gt" in save_option:
        bboxes_gt = [[]]
        ins_gt = instance_label
        npcs_gt = npcs_map
        # import pdb
        # pdb.set_trace()
        num_ins = ins_gt.max()+1
        if num_ins >= 1:
            for ins_i in range(num_ins):
                mask_i = ins_gt == ins_i
                xyz_input_i = xyz_input[mask_i]
                npcs_i = npcs_gt[mask_i]
                if xyz_input_i.shape[0]<=5:
                    continue

                bbox_xyz, scale, rotation, translation, out_transform, best_inlier_idx = \
                    estimate_pose_from_npcs(xyz_input_i, npcs_i)
                if scale[0] == None:
                    continue
                bboxes_gt[0].append(bbox_xyz.tolist())
        img_bbox_gt = pc_img.copy()
        draw_bbox(img_bbox_gt, bboxes_gt[0], trans)
        if save_detail:
            cv2.imwrite(f"{save_root}/bbox_gt.png", img_bbox_gt)
        X_START = EDGE + (WIDTH + EDGE) * 2
        Y_START = EDGE + (HEIGHT + EDGE) * 1
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = img_bbox_gt
        text = "bbox_gt"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "bbox_gt_pure" in save_option:
        bboxes_gt = [[]]
        ins_gt = instance_label
        npcs_gt = npcs_map
        # import pdb
        # pdb.set_trace()
        num_ins = ins_gt.max()+1
        if num_ins >= 1:
            for ins_i in range(num_ins):
                mask_i = ins_gt == ins_i
                xyz_input_i = xyz_input[mask_i]
                npcs_i = npcs_gt[mask_i]
                if xyz_input_i.shape[0]<=5:
                    continue

                bbox_xyz, scale, rotation, translation, out_transform, best_inlier_idx = \
                    estimate_pose_from_npcs(xyz_input_i, npcs_i)
                if scale[0] == None:
                    continue

                bboxes_gt[0].append(bbox_xyz.tolist())
        img_bbox_gt_pure = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
        draw_bbox(img_bbox_gt_pure, bboxes_gt[0], trans)
        if save_detail:
            cv2.imwrite(f"{save_root}/bbox_gt_pure.png", img_bbox_gt_pure)
        X_START = EDGE + (WIDTH + EDGE) * 2
        Y_START = EDGE + (HEIGHT + EDGE) * 0
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = img_bbox_gt_pure
        text = "bbox_gt_pure"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    cv2.imwrite(f"{final_save_root}/{name}.png", final_img)
