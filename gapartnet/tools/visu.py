import torch
import numpy as np
import yaml
from os.path import join as pjoin
import os
import argparse
import sys
sys.path.append(sys.path[0] + "/..")
import importlib
from gapartnet.structures.point_cloud import PointCloud
from gapartnet.datasets.gapartnet_new import apply_voxelization
from gapartnet.utils.pose_fitting import estimate_pose_from_npcs
import cv2
from typing import List
import glob
from visu_utils import OBJfile2points, map2image, save_point_cloud_to_ply, \
    WorldSpaceToBallSpace, FindMaxDis, draw_bbox_old, draw_bbox, COLOR20, \
    OTHER_COLOR, HEIGHT, WIDTH, EDGE, K, font, fontScale, fontColor,thickness, lineType 

GAPARTNET_DATA_ROOT = "data/GAPartNet_All"
RAW_IMG_ROOT = "data/image_kuafu" # just for visualization, not necessary
SAVE_ROOT = "output/GAPartNet_result"

# OPTION
FEW_SHOT = True # if True, only visualize the FEW_NUM samples, otherwise visualize all
FEW_NUM = 10 # only valid when FEW_SHOT is True
save_option = ["raw", "pc", "sem_pred", "ins_pred", "npcs_pred", "bbox_pred", "pure_bbox", 
               "sem_gt", "ins_gt", "npcs_gt", "bbox_gt", "bbox_gt_pure"] # save options
SAVE_LOCAL = False 
splits = ["train", "val", "test_intra", "test_inter", ] #
dir_name = "visu" 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_path", type=str, default = "gapartnet.models.gapartnet.InsSeg")
    parser.add_argument("--ckpt", type=str, default = "ckpt/new.ckpt")
    parser.add_argument("--in_channels", type=int, default = 6)
    parser.add_argument("--device", type=str, default = "cuda:0")
    args = parser.parse_args()


    perception_cfg = {}
    perception_cfg["class_path"] = args.class_path
    perception_cfg["ckpt"] = args.ckpt
    perception_cfg["device"] = args.device
    perception_cfg["in_channels"] = args.in_channels
    # perception_cfg["channels"] = [16, 64, 112] # [16, 32, 48, 64, 80, 96, 112]

    return args, perception_cfg

class MODEL:
    def __init__(self, cfg):
        self.model_cfg = cfg
        self.perception_model = self._load_perception_model(self.model_cfg)
        self.device = cfg["device"]


    def _load_perception_model(self, perception_model_cfg):
        class_path = perception_model_cfg["class_path"]
        ckpt_pth = perception_model_cfg["ckpt"]
        device = perception_model_cfg["device"]

        module_name = ".".join(class_path.split(".")[:-1])
        class_name = class_path.split(".")[-1]

        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        net = cls.load_from_checkpoint(ckpt_pth)

        net.cluster_proposals_start_at = 0
        net.score_net_start_at = 0
        net.npcs_net_start_at = 0
        net.freeze()
        net.eval()
        net.to(device)

        return net

    def _inference_perception_model(self,  points_list: List[torch.Tensor]):
        device = self.perception_model.device

        pcs = []
        for points in points_list:
            pc = PointCloud(
                scene_id=["eval"],
                points=points,
                obj_cat = 0
            )
            pc = apply_voxelization(
                pc,  voxel_size=(1. / 100, 1. / 100, 1. / 100)
            )
            pc = pc.to(device=device)
            pcs.append(pc)

        with torch.no_grad():
            scene_ids, segmentations, proposals = self.perception_model(pcs)

        sem_preds = segmentations.sem_preds
        if proposals is not None:
            pt_xyz = proposals.pt_xyz
            batch_indices = proposals.batch_indices
            proposal_offsets = proposals.proposal_offsets
            num_points_per_proposal = proposals.num_points_per_proposal
            num_proposals = num_points_per_proposal.shape[0]
            npcs_preds = proposals.npcs_preds
            score_preds= proposals.score_preds

            indices = torch.arange(sem_preds.shape[0], dtype=torch.int64, device=device)
            proposal_indices = indices[proposals.valid_mask][proposals.sorted_indices]
            

        npcs_maps = pcs[0].points[:,:3].clone()
        npcs_maps[:] = 230./255.
        if proposals is not None:
            npcs_maps[proposal_indices] = npcs_preds
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
                bbox_xyz, scale, rotation, translation, out_transform, best_inlier_idx = \
                    estimate_pose_from_npcs(xyz_i.cpu().numpy(), npcs_i.cpu().numpy())
                # import pdb
                # pdb.set_trace()
                if scale[0] == None:
                    continue
                bboxes[batch_idx].append(bbox_xyz.tolist())
        try:
            return bboxes, sem_preds, npcs_maps, proposal_indices, proposal_offsets
        except:
            return bboxes, sem_preds, npcs_maps, None, None

    def inference(self, points):
        bboxes, sem_preds, npcs_maps, proposal_indices, proposal_offsets = self._inference_perception_model([points])
        return bboxes, sem_preds, npcs_maps, proposal_indices, proposal_offsets

    def inference_real(self, file_path, label = "", save_root = "/scratch/genghaoran/GAPartNet/GAPartNet_inference/asset/"):
        trans_gapartnet = np.array([ 1.26171422e+00, -6.60613179e-04,  4.20249701e-02,  4.23497820e+00])
        data_path = file_path
        if ".obj" in file_path:
            points = OBJfile2points(data_path)
            points[:, 2] = -points[:,2]
            points[:, 1] = -points[:,1]
            save_point_cloud_to_ply(points[:,:3], points[:,3:6], data_path.split("/")[-1].split(".")[0] + label+"_preinput.ply")
            xyz, max_radius, center = WorldSpaceToBallSpace(points[:,:3])
            trans = np.array([max_radius, center[0], center[1], center[2]])
        else:
            import pdb
            pdb.set_trace()   
        points_input = torch.cat(
            (torch.tensor(xyz, dtype=torch.float32  ,device = self.perception_model.device),
            torch.tensor(points[:,-3:], dtype=torch.float32  ,device = self.perception_model.device)),
            dim = 1)
        save_point_cloud_to_ply(points_input[:,:3], points_input[:,3:6]*255, data_path.split("/")[-1].split(".")[0] + label+"_input.ply")
        bboxes, sem_preds, npcs_maps, proposal_indices, proposal_offsets = self.perception_model._inference_perception_model([points_input])

        print("-------bbox" ,len(bboxes[0]),"-------")
        point_img = points_input.cpu().numpy()
        point_img[:,:3] = point_img[:,:3] * trans_gapartnet[0] + trans_gapartnet[1:4]
        img = map2image(point_img[:,:3], point_img[:,3:6]*255.0) 
        
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_raw = im_rgb.copy()
        # cv2.imwrite(save_root + label + data_path.split("/")[-1].split(".")[0]+".png", im_rgb)
        # trans = np.array([xyz_scale, xyz_mean[0], xyz_mean[1], xyz_mean[2]])
        
        cv2.imwrite(save_root + label+data_path.split("/")[-1].split(".")[0]+"_raw.png", im_rgb)
        draw_bbox(im_rgb, bboxes[0], trans_gapartnet)
        cv2.imwrite(save_root + label+data_path.split("/")[-1].split(".")[0]+"_bbox.png", im_rgb)
        # for i,bbox in enumerate(bboxes[0]):
        #     bbox_now = [bbox,]
        #     img_now = img_raw.copy()
        #     draw_bbox(img_now, bbox_now, trans_gapartnet)
        #     cv2.imwrite(save_root + label+data_path.split("/")[-1].split(".")[0]+f"_bbox_{i}.png", img_now)
        #     npcs_maps_now = npcs_maps.clone()
        #     npcs_maps_now[:] = 230./255.
        #     # import pdb
        #     # pdb.set_trace()
        #     npcs_maps_now[proposal_indices[proposal_offsets[i]:proposal_offsets[i+1]]]=npcs_maps[proposal_indices[proposal_offsets[i]:proposal_offsets[i+1]]]
        #     img_npcs_now = map2image(point_img[:,:3], npcs_maps_now.cpu().numpy()*255) 
        #     im_rgb_npcs_now = cv2.cvtColor(img_npcs_now, cv2.COLOR_BGR2RGB)
        #     cv2.imwrite(save_root + label+data_path.split("/")[-1].split(".")[0]+f"_npcs{i}.png", im_rgb_npcs_now)
        
        rgb_sem = COLOR20[sem_preds.cpu().numpy()]
        img_sem = map2image(point_img[:,:3], rgb_sem) 
        im_rgb_sem = cv2.cvtColor(img_sem, cv2.COLOR_BGR2RGB)
        # draw_bbox(im_rgb_sem, bboxes[0], trans_gapartnet)
        cv2.imwrite(save_root + label+data_path.split("/")[-1].split(".")[0]+"_sem.png", im_rgb_sem)

        img_npcs = map2image(point_img[:,:3], npcs_maps.cpu().numpy()*255) 
        im_rgb_npcs = cv2.cvtColor(img_npcs, cv2.COLOR_BGR2RGB)
        draw_bbox(im_rgb_npcs, bboxes[0], trans_gapartnet)
        cv2.imwrite(save_root + label+data_path.split("/")[-1].split(".")[0]+"_npcs.png", im_rgb_npcs)

    def inference_gapartnet(self, name, split = "train", other_string = ""):
        data_path = f"{GAPARTNET_DATA_ROOT}/{split}/pth/{name}.pth"
        trans_path = f"{GAPARTNET_DATA_ROOT}/{split}/meta/{name}.txt"
        pc, rgb, semantic_label, instance_label, npcs_map = torch.load(data_path)
        
        trans = np.loadtxt(trans_path)
        xyz = pc * trans[0] + trans[1:4]

        # save_point_cloud_to_ply(xyz, rgb*255, data_path.split("/")[-1].split(".")[0]+"_preinput.ply")
        # save_point_cloud_to_ply(pc, rgb*255, data_path.split("/")[-1].split(".")[0]+"_input.ply")
        
        points_input = torch.cat((torch.tensor(pc, device = self.perception_model.device),torch.tensor(rgb, device = self.perception_model.device)), dim = 1)

        bboxes, sem_preds, npcs_maps, proposal_indices, proposal_offsets = self.perception_model._inference_perception_model([points_input])

        # img = map2image(xyz, rgb*255.0) 
        # im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(save_root+data_path.split("/")[-1].split(".")[0]+".png", im_rgb)
        # draw_bbox(im_rgb, bboxes[0], trans)
        # cv2.imwrite(save_root+data_path.split("/")[-1].split(".")[0]+"_bbox.png", im_rgb)
        img = map2image(xyz, rgb*255.0) 
        
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_raw = im_rgb.copy()
        # cv2.imwrite(save_root + label + data_path.split("/")[-1].split(".")[0]+".png", im_rgb)
        # trans = np.array([xyz_scale, xyz_mean[0], xyz_mean[1], xyz_mean[2]])
        
        cv2.imwrite(save_root+others+"raw.png", im_rgb)
        # import pdb
        # pdb.set_trace()
        draw_bbox(im_rgb, bboxes[0], trans)
        cv2.imwrite(save_root+others+"bbox.png", im_rgb)
        # for i,bbox in enumerate(bboxes[0]):
        #     bbox_now = [bbox,]
        #     img_now = img_raw.copy()
        #     draw_bbox(img_now, bbox_now, trans)
        #     cv2.imwrite(save_root + label+data_path.split("/")[-1].split(".")[0]+f"_bbox_{i}.png", img_now)
        #     npcs_maps_now = npcs_maps.clone()
        #     npcs_maps_now[:] = 230./255.
        #     # import pdb
        #     # pdb.set_trace()
        #     npcs_maps_now[proposal_indices[proposal_offsets[i]:proposal_offsets[i+1]]]=npcs_maps[proposal_indices[proposal_offsets[i]:proposal_offsets[i+1]]]
        #     img_npcs_now = map2image(point_img[:,:3], npcs_maps_now.cpu().numpy()*255) 
        #     im_rgb_npcs_now = cv2.cvtColor(img_npcs_now, cv2.COLOR_BGR2RGB)
        #     cv2.imwrite(save_root + label+data_path.split("/")[-1].split(".")[0]+f"_npcs{i}.png", im_rgb_npcs_now)
        
        rgb_sem = COLOR20[sem_preds.cpu().numpy()]
        img_sem = map2image(xyz[:,:3], rgb_sem) 
        im_rgb_sem = cv2.cvtColor(img_sem, cv2.COLOR_BGR2RGB)
        # draw_bbox(im_rgb_sem, bboxes[0], trans)
        cv2.imwrite(save_root+others+"sem.png", im_rgb_sem)

        img_npcs = map2image(xyz[:,:3], npcs_maps.cpu().numpy()*255) 
        im_rgb_npcs = cv2.cvtColor(img_npcs, cv2.COLOR_BGR2RGB)
        draw_bbox(im_rgb_npcs, bboxes[0], trans)
        cv2.imwrite(save_root+others+"npcs.png", im_rgb_npcs)

    def process_objfile(self, file_path, label = "", save_root = "/scratch/genghaoran/GAPartNet/GAPartNet_inference/asset/"):
        data_path = file_path
        if ".obj" in file_path:
            points = OBJfile2points(data_path)
            points[:, 2] = -points[:,2]
            points[:, 1] = -points[:,1]
            save_point_cloud_to_ply(points[:,:3], points[:,3:6], data_path.split("/")[-1].split(".")[0] + label+"_preinput.ply")
            xyz, max_radius, center = WorldSpaceToBallSpace(points[:,:3])
            trans = np.array([max_radius, center[0], center[1], center[2]])
        else:
            import pdb
            pdb.set_trace()   
        points_input = torch.cat(
            (torch.tensor(xyz, dtype=torch.float32  ,device = self.perception_model.device),
            torch.tensor(points[:,-3:], dtype=torch.float32  ,device = self.perception_model.device)),
            dim = 1)
        trans_gapartnet = np.array([ 1.26171422e+00, -6.60613179e-04,  4.20249701e-02,  4.23497820e+00])
        return points_input, trans_gapartnet
    
    def process_gapartnetfile(self, name, split = "train"):
        data_path = f"{GAPARTNET_DATA_ROOT}/{split}/pth/{name}.pth"
        trans_path = f"{GAPARTNET_DATA_ROOT}/{split}/meta/{name}.txt"

        pc, rgb, semantic_label, instance_label, npcs_map = torch.load(data_path)
        
        trans = np.loadtxt(trans_path)
        xyz = pc * trans[0] + trans[1:4]

        # save_point_cloud_to_ply(xyz, rgb*255, data_path.split("/")[-1].split(".")[0]+"_preinput.ply")
        # save_point_cloud_to_ply(pc, rgb*255, data_path.split("/")[-1].split(".")[0]+"_input.ply")
        
        points_input = torch.cat((torch.tensor(pc, device = self.perception_model.device),torch.tensor(rgb, device = self.perception_model.device)), dim = 1)
        return points_input, trans, semantic_label, instance_label, npcs_map
        
def draw_result(save_option, save_root, name, points_input, trans, bboxes, sem_preds, npcs_maps, proposal_indices, proposal_offsets, gts=None, have_proposal = True, save_local = False):
    
    final_save_root = f"{save_root}/"
    save_root = f"{save_root}/{name}/"
    if save_local:
        os.makedirs(save_root, exist_ok=True)
    final_img = np.ones((3 * (HEIGHT + EDGE) + EDGE, 4 * (WIDTH + EDGE) + EDGE, 3), dtype=np.uint8) * 255
    xyz_input = points_input[:,:3]
    rgb = points_input[:,3:6]
    xyz = xyz_input * trans[0] + trans[1:4]
    pc_img = map2image(xyz, rgb*255.0)
    pc_img = cv2.cvtColor(pc_img, cv2.COLOR_BGR2RGB)
    if "raw" in save_option:
        raw_img_path = f"{RAW_IMG_ROOT}/{name}.png"
        if os.path.exists(raw_img_path):
            raw_img = cv2.imread(raw_img_path)
            if save_local:
                cv2.imwrite(f"{save_root}/raw.png", raw_img)
            X_START = EDGE
            Y_START = EDGE
            final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = raw_img
            text = "raw"
            cv2.putText(final_img, text, 
                (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
                font, fontScale, fontColor, thickness, lineType)
    if "pc" in save_option:
        if save_local:
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
        if save_local:
            cv2.imwrite(f"{save_root}/sem_pred.png", sem_pred_img)
        X_START = EDGE + (WIDTH + EDGE)
        Y_START = EDGE + (HEIGHT + EDGE)
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = sem_pred_img
        text = "sem_pred"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "ins_pred" in save_option:
        ins_pred_color = np.ones_like(xyz) * 230
        if have_proposal:
            for ins_i in range(len(proposal_offsets) - 1):
                ins_pred_color[proposal_indices[proposal_offsets[ins_i]:proposal_offsets[ins_i + 1]]] = COLOR20[ins_i%19 + 1]
        
        ins_pred_img = map2image(xyz, ins_pred_color)
        ins_pred_img = cv2.cvtColor(ins_pred_img, cv2.COLOR_BGR2RGB)
        if save_local:
            cv2.imwrite(f"{save_root}/ins_pred.png", ins_pred_img)    
        X_START = EDGE + (WIDTH + EDGE) * 1
        Y_START = EDGE + (HEIGHT + EDGE) * 2
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = ins_pred_img
        text = "ins_pred"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "npcs_pred" in save_option:
        npcs_pred_img = map2image(xyz, npcs_maps*255.0)
        npcs_pred_img = cv2.cvtColor(npcs_pred_img, cv2.COLOR_BGR2RGB)
        if save_local:
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
        draw_bbox(img_bbox_pred, bboxes[0], trans)
        if save_local:
            cv2.imwrite(f"{save_root}/bbox_pred.png", img_bbox_pred)
        X_START = EDGE + (WIDTH + EDGE) * 2
        Y_START = EDGE + (HEIGHT + EDGE) * 2
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = img_bbox_pred
        text = "bbox_pred"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "pure_bbox" in save_option:
        img_empty = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
        draw_bbox(img_empty, bboxes[0], trans)
        if save_local:
            cv2.imwrite(f"{save_root}/bbox_pure.png", img_empty)
        X_START = EDGE + (WIDTH + EDGE) * 2
        Y_START = EDGE + (HEIGHT + EDGE) * 3
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = img_empty
        text = "bbox_pred_pure"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "sem_gt" in save_option:
        sem_gt = gts[0]
        sem_gt_img = map2image(xyz, COLOR20[sem_gt])
        sem_gt_img = cv2.cvtColor(sem_gt_img, cv2.COLOR_BGR2RGB)
        if save_local:
            cv2.imwrite(f"{save_root}/sem_gt.png", sem_gt_img)      
        X_START = EDGE + (WIDTH + EDGE) * 0
        Y_START = EDGE + (HEIGHT + EDGE) * 1
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = sem_gt_img
        text = "sem_gt"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "ins_gt" in save_option:
        ins_gt = gts[1]
        ins_color = COLOR20[ins_gt%19 + 1]
        ins_color[np.where(ins_gt == -100)] = 230
        ins_gt_img = map2image(xyz, ins_color)
        
        ins_gt_img = cv2.cvtColor(ins_gt_img, cv2.COLOR_BGR2RGB)
        if save_local:
            cv2.imwrite(f"{save_root}/ins_gt.png", ins_gt_img)      
        X_START = EDGE + (WIDTH + EDGE) * 0
        Y_START = EDGE + (HEIGHT + EDGE) * 2
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = ins_gt_img
        text = "ins_gt"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "npcs_gt" in save_option:
        npcs_gt = gts[2] + 0.5
        npcs_gt_img = map2image(xyz, npcs_gt*255.0)
        npcs_gt_img = cv2.cvtColor(npcs_gt_img, cv2.COLOR_BGR2RGB)
        if save_local:
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
        ins_gt = gts[1]
        npcs_gt = gts[2]
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
        if save_local:
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
        ins_gt = gts[1]
        npcs_gt = gts[2]
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
        if save_local:
            cv2.imwrite(f"{save_root}/bbox_gt_pure.png", img_bbox_gt_pure)
        X_START = EDGE + (WIDTH + EDGE) * 2
        Y_START = EDGE + (HEIGHT + EDGE) * 0
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = img_bbox_gt_pure
        text = "bbox_gt_pure"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    cv2.imwrite(f"{final_save_root}/{name}.png", final_img)

def main():
    args, perception_cfg = get_args()

    # initialize the perception model    
    model = MODEL(perception_cfg)
    print("finish load model")

    FAIL = []
    for split in splits:
        paths = glob.glob(GAPARTNET_DATA_ROOT + "/" + split + "/pth/*")
        
        if FEW_SHOT:
            import random
            r_nums = random.sample(list(range(0, len(paths))), FEW_NUM)
            used_paths = []
            for r_num in r_nums:
                used_paths.append(paths[r_num])
            paths = used_paths
            
        for i, path in enumerate(paths):
            name = path.split(".")[0].split("/")[-1]
            print(split, " ", i, " ", name)

            save_root = f"{SAVE_ROOT}/{dir_name}/{split}"
            os.makedirs(save_root,exist_ok = True)
            final_save_root = f"{save_root}/"
            if os.path.exists(f"{final_save_root}/{name}.png"):
                continue

            points_input, trans, semantic_label, instance_label, npcs_map = model.process_gapartnetfile(name, split)
            bboxes, sem_preds, npcs_maps, proposal_indices, proposal_offsets = model.inference(points_input)

            # visualize results in the image

            # try:
            if proposal_indices == None:
                draw_result(save_option, save_root, name, 
                            points_input.cpu().numpy(), trans, bboxes, sem_preds.cpu().numpy(), npcs_maps.cpu().numpy(), 
                            proposal_indices, proposal_offsets, gts = [semantic_label, instance_label, npcs_map], 
                            have_proposal = False, save_local=SAVE_LOCAL) 
            else:            
                draw_result(save_option, save_root, name, points_input.cpu().numpy(), trans, bboxes, sem_preds.cpu().numpy(), 
                            npcs_maps.cpu().numpy(), proposal_indices.cpu().numpy(), proposal_offsets.cpu().numpy(), 
                            gts = [semantic_label, instance_label, npcs_map], have_proposal = True, save_local=SAVE_LOCAL)

    # return model

    
if __name__ == "__main__":
    model = main()