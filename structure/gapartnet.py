from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Union
from sklearn.decomposition import PCA
import numpy as np
import torch
import math
# import open3d

from os.path import join as pjoin
import logging

# from pc_flow.pycpd import RigidRegistration
# from pc_flow.pose_fitting import estimate_pose_from_npcs
from scipy.spatial.transform import Rotation as R
import random

Log = logging.getLogger(__name__)
import sys
sys.path.append("/data/haoran/LLM-GAPartNet/vision/gapartnet")
from structure.utils import query_part_anno, KNN_classifier, load_data_single_file, mask_change_reso, \
    PART_ID2NAME, PART_ID2NAME_OLD, COLOR20, font, fontScale, fontColor,thickness, lineType, save_point_cloud_to_ply,\
    map2image, get_point_cloud, FPS, _inference_perception_model, _load_perception_model, \
    WorldSpaceToBallSpace, draw_bbox, read_pcs_from_ply, _inference_perception_model_with_masks,\
    create_transformation_matrix, transform_point_cloud, get_point2image, _estimate_pose_with_masks
from structure.utils import TARGET_GAPARTS,draw_bbox_from_world
    
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import sys
sys.path.append(sys.path[0] + "/../")
# from GroundedSAM.grounded_sam_demo import inference_gounded_sam, load_dino_sam_model
# try:
#     from SAM.segment_anything import SamAutomaticMaskGenerator, sam_model_registry
# except:
#     Log.critical("import SAM failed! TODO: fix this")

import glob
# from dinov2.feature import DINOV2
from PIL import Image
import json
import warnings
warnings.filterwarnings("ignore")

@dataclass
class ObjIns:
    # basic
    name: str = None
    cate: str = None
    image: np.ndarray = None
    image_reso: List = None
    obj_mask: np.ndarray = None
    
    # camera info
    K: np.ndarray = None
    cam_p: np.ndarray = None
    cam_q: np.ndarray = None
    cam_e_mat: np.ndarray = None
    world2camera_rotation: np.ndarray = None
    camera2world_translation: np.ndarray = None
    
    # parts
    parts_sem_ids: List = None
    parts_ins_ids: List = None
    parts_bboxes: List = None
    parts_link_names: List = None
    
    # depth and
    depth: np.ndarray = None
    pcs_all_xyz: np.ndarray = None
    pcs_all_rgb: np.ndarray = None
    pcs_all_pixel: np.ndarray = None
    pcs_xyz: np.ndarray = None
    pcs_rgb: np.ndarray = None
    pcs_pixel: np.ndarray = None
    pcs_gt_sem: np.ndarray = None
    pcs_gt_ins: np.ndarray = None
    pcs_gt_npcs: np.ndarray = None
    pcs_xyz_ball: np.ndarray = None
    trans: np.ndarray = None
    
    pcs_pred_sem: np.ndarray = None
    pcs_pred_ins: np.ndarray = None
    pcs_pred_ins_masks: np.ndarray = None
    pcs_pred_ins_labels: np.ndarray = None
    pcs_pred_npcs: np.ndarray = None
    pcs_pred_bbox: List = None
    
    given_masks: np.ndarray = None
    given_masks_GAPart_ids: np.ndarray = None
    masks_pred_bbox: List = None
    
    # GT instance segmentation
    gt_ins_ids: np.ndarray = None
    gt_sem_ids: np.ndarray = None
    gt_masks: np.ndarray = None
    gt_bboxes: np.ndarray = None
    gt_npcs_list: List = None
    img_sem_map: np.ndarray = None
    img_ins_map: np.ndarray = None
    img_npcs_map: np.ndarray = None
    img_part_valid_mask: np.ndarray = None
    
    # DINO features
    image_dino_fea: np.ndarray = None
    
    # Grounded SAM results
    prompts: str = None
    masks_raw_reso: np.ndarray = None
    masks_low_reso: np.ndarray = None
    iou_predictions: List = None
    transformed_boxes: List = None
    point_masks: np.ndarray = None
    
    # SAM results
    sam_pred_masks: np.array = None
    sam_masks_low_reso: np.array = None
    sam_fea_max: np.array = None
    sam_GAPart_id_pred: np.ndarray = None
    sam_GAPart_label_pred: List = None

    # Grounded SAM + DINO + GAPart
    fea_max: np.ndarray = None
    GAPart_lable_pred: List = None
    GAPart_id_pred: np.ndarray = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }

    def to_tensor(self) -> "ObjIns":
        return MyImage(**{
            k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in self.to_dict().items()
        }) # type: ignore

    def to(self, device: torch.device) -> "ObjIns":
        return MyImage(**{
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in self.to_dict().items()
        }) # type: ignore

    def mask_fea_process(self): 
        if self.masks_raw_reso is None:
            print("No GroundedSAM results available")
            return
        self.masks_low_reso = np.array([mask_change_reso(mask[0], 50, 50) for mask in self.masks_raw_reso])
        # self.masks_low_reso = mask_change_reso(self.masks_raw_reso[:,0,...], 50, 50)
        self.fea_max = np.array([(self.image_dino_fea*self.masks_low_reso[i].reshape(50,50,1)).max(0).max(0) for i in range(len(self.masks_low_reso))])
    
    def sam_mask_fea_process(self): 
        if self.sam_pred_masks is None:
            print("No GroundedSAM results available")
            return
        self.sam_masks_low_reso = np.array([mask_change_reso(mask, 50, 50) for mask in self.sam_pred_masks])
        self.sam_fea_max = np.array([(self.image_dino_fea*self.sam_masks_low_reso[i].reshape(50,50,-1)).max(0).max(0) for i in range(len(self.sam_masks_low_reso))])

    def inference_gounded_sam(self, prompt, dino_model = None, sam_model = None):
        results = inference_gounded_sam(
            prompt,
            img = self.image,
            name = self.name,
            output_dir="",
            dino_model=dino_model,
            sam_model=sam_model,
        )
        if results[0] is None:
            print("No GroundedSAM results output")
            self.masks_raw_reso = None
            self.iou_predictions = None
            self.transformed_boxes = None
            return
        sam_pred_data = results[0]
        self.masks_raw_reso = sam_pred_data["masks"]
        self.iou_predictions = sam_pred_data["iou_predictions"]
        self.transformed_boxes = sam_pred_data["transformed_boxes"]

    def get_GAPart_grounding_result(self, classifier, use_inference = True):
        if use_inference:
            if self.fea_max is None:
                self.mask_fea_process()
                if self.fea_max is None:
                    print("No GroundedSAM results available")
                    return
            y_pred = classifier.predict(self.fea_max)
            self.GAPart_id_pred = y_pred
            self.GAPart_label_pred = [PART_ID2NAME[y_pred_i] for y_pred_i in y_pred]
        else:
            print("No GAPart Grounding results available")
    
    def get_sam_grounding_result(self, classifier, use_inference = True):
        if use_inference:
            if self.sam_fea_max is None:
                self.sam_mask_fea_process()
                if self.sam_fea_max is None:
                    print("No GroundedSAM results available")
                    return
            y_pred = classifier.predict(self.sam_fea_max)
            self.sam_GAPart_id_pred = y_pred
            self.sam_GAPart_label_pred = [PART_ID2NAME[y_pred_i] for y_pred_i in y_pred]
        else:
            print("No GAPart Grounding results available")
            
    def visualization(
            self,
            output_root = "output",
            options = ["img", "img_obj", "img_gt_ins", "img_gt_sem", "img_gt_npcs", "pred_ins", "pc", "input_pc"],
            render_text = True,
        ):
        D = 40 # delta
        # L = 800 # length for an image
        H,W = self.image_reso
        all_img_x_num = 3
        all_img_y_num = math.ceil(len(options)/3.)
        all_img = np.ones((D+(all_img_x_num)*(H+D), D+(all_img_y_num)*(W+D),3))*255
        img_num_i = -1
        xyz = self.pcs_xyz 
        rgb = self.pcs_rgb

        if "img" in options:
            img_num_i += 1
            img_base_x = D + int(img_num_i%all_img_x_num)*(H+D)
            img_base_y = D + int(img_num_i/all_img_x_num)*(W+D)
            all_img[img_base_x:img_base_x+H, img_base_y:img_base_y+W,:] = self.image
            cv2.putText(all_img, f"ori_img", (D + int(W/2) - 4*D, D + H + int(D/2)),font, fontScale, fontColor, thickness, lineType)
        if "img_obj" in options:
            image_obj = self.image.copy()
            img_num_i += 1
            img_base_x = D + int(img_num_i%all_img_x_num)*(H+D)
            img_base_y = D + int(img_num_i/all_img_x_num)*(W+D)
            if self.obj_mask is not None:
                image_obj[self.obj_mask] = [250,0,0]
            all_img[img_base_x:img_base_x+H, img_base_y:img_base_y+W,:] = image_obj
            cv2.putText(all_img, f"gt_ins", (D + int(W/2) - 4*D, D + H + int(D/2)),font, fontScale, fontColor, thickness, lineType)
        if "img_gt_sem" in options:
            img_gt_sem = self.image.copy()
            for part_id, ins_id in enumerate(self.parts_ins_ids):
                part_mask = self.img_ins_map == ins_id
                sem_id = self.parts_sem_ids[part_id]
                img_gt_sem[part_mask] = COLOR20[(sem_id)%19+1]
                if render_text:
                    first_pixel_x, first_pixel_y = (np.where(part_mask == True)[0][0], np.where(part_mask == True)[1][0])
                    text = TARGET_GAPARTS[sem_id]
                    cv2.putText(img_gt_sem, text, (first_pixel_y, first_pixel_x),font, fontScale, fontColor, thickness, lineType)
            
            img_num_i += 1
            img_base_x = D + int(img_num_i%all_img_x_num)*(H+D)
            img_base_y = D + int(img_num_i/all_img_x_num)*(W+D)
            all_img[img_base_x:img_base_x+H, img_base_y:img_base_y+W,:] = img_gt_sem
            cv2.putText(all_img, f"gt_sem", (img_base_y + int(W/2) - 4*D, img_base_x + H + int(D/2)),font, fontScale, fontColor, thickness, lineType)
        if "img_gt_ins" in options:
            img_gt_ins = self.image.copy()
            for part_id, ins_id in enumerate(self.parts_ins_ids):
                part_mask = self.img_ins_map == ins_id
                sem_id = self.parts_sem_ids[part_id]
                img_gt_ins[part_mask] = COLOR20[(part_id)%19+1]
                if render_text:
                    first_pixel_x, first_pixel_y = (np.where(part_mask == True)[0][0], np.where(part_mask == True)[1][0])
                    text = TARGET_GAPARTS[sem_id]
                    cv2.putText(img_gt_ins, text, (first_pixel_y, first_pixel_x),font, fontScale, fontColor, thickness, lineType)
            
            img_num_i += 1
            img_base_x = D + int(img_num_i%all_img_x_num)*(H+D)
            img_base_y = D + int(img_num_i/all_img_x_num)*(W+D)
            all_img[img_base_x:img_base_x+H, img_base_y:img_base_y+W,:] = img_gt_ins
            cv2.putText(all_img, f"gt_ins", (img_base_y + int(W/2) - 4*D, img_base_x + H + int(D/2)),font, fontScale, fontColor, thickness, lineType)
        if "img_gt_bbox" in options:
            img_gt_bbox = self.image.copy()
            # for part_id, bbox in enumerate(self.parts_bboxes):

            _, point2images = draw_bbox_from_world(img_gt_bbox, self.parts_bboxes, self.K, self.camera2world_translation, self.world2camera_rotation)
            if render_text:
                for bbox_id, point2image in enumerate(point2images):
                    for i in range(8):
                        first_pixel_y, first_pixel_x = point2image[i][0], point2image[i][1]
                        text = str(i)
                        cv2.putText(img_gt_bbox, text, (first_pixel_y, first_pixel_x),font, fontScale, fontColor, thickness, lineType)
            
            img_num_i += 1
            img_base_x = D + int(img_num_i%all_img_x_num)*(H+D)
            img_base_y = D + int(img_num_i/all_img_x_num)*(W+D)
            all_img[img_base_x:img_base_x+H, img_base_y:img_base_y+W,:] = img_gt_bbox
            cv2.putText(all_img, f"pc_pred_bbox", (img_base_y + int(W/2) - 4*D, img_base_x + H + int(D/2)),font, fontScale, fontColor, thickness, lineType)
        if "img_gt_npcs" in options:
            img_gt_npcs = self.image.copy()
            img_num_i += 1
            img_base_x = D + int(img_num_i%all_img_x_num)*(H+D)
            img_base_y = D + int(img_num_i/all_img_x_num)*(W+D)
            img_gt_npcs[self.img_part_valid_mask] = ((self.img_npcs_map[self.img_part_valid_mask]+0.5)*255).astype(np.uint8)
            all_img[img_base_x:img_base_x+H, img_base_y:img_base_y+W,:] = img_gt_npcs
            if render_text:
                for mask_id, gt_mask in enumerate(self.gt_masks):
                    first_pixel_x, first_pixel_y = (np.where(gt_mask == True)[0][0], np.where(gt_mask == True)[1][0])
                    text = PART_ID2NAME_OLD[self.gt_sem_ids[mask_id]]
                cv2.putText(all_img, text, (img_base_y + first_pixel_y, img_base_x + first_pixel_x),font, fontScale, fontColor, thickness, lineType)
            cv2.putText(all_img, f"gt_npcs", (img_base_y + int(W/2) - 4*D, img_base_x + H + int(D/2)),font, fontScale, fontColor, thickness, lineType)
        if "pc" in options:
            pc_img = map2image(xyz, rgb*255.0, self.K, H, W)
            pc_img = pc_img[...,::-1]
            img_num_i += 1
            img_base_x = D + int(img_num_i%all_img_x_num)*(H+D)
            img_base_y = D + int(img_num_i/all_img_x_num)*(W+D)
            all_img[img_base_x:img_base_x+H, img_base_y:img_base_y+W,:] = pc_img
            cv2.putText(all_img, f"pc", (img_base_y + int(W/2) - 4*D, img_base_x + H + int(D/2)),font, fontScale, fontColor, thickness, lineType)         
        if "GroundSAM_pred_ins" in options:
            img_pred_ins = self.image.copy()
            img_num_i += 1
            img_base_x = D + int(img_num_i%all_img_x_num)*(H+D)
            img_base_y = D + int(img_num_i/all_img_x_num)*(W+D)
            for mask_id, pred_mask in enumerate(self.masks_raw_reso[:,0,:,:]):
                img_pred_ins[pred_mask] = COLOR20[(mask_id+1)%20]
            all_img[img_base_x:img_base_x+H, img_base_y:img_base_y+W,:] = img_pred_ins
            if render_text:
                for mask_id, pred_mask in enumerate(self.masks_raw_reso[:,0,:,:]):
                    first_pixel_x, first_pixel_y = (np.where(pred_mask == True)[0][0], np.where(pred_mask == True)[1][0])
                    text = PART_ID2NAME[self.GAPart_id_pred[mask_id]]
                    cv2.putText(all_img, text, (img_base_y + first_pixel_y, img_base_x + first_pixel_x),font, fontScale, fontColor, thickness, lineType)
            cv2.putText(all_img, f"groundsam_pred_ins_{self.prompts}", (img_base_y + int(W/2) - 4*D, img_base_x + H + int(D/2)),font, fontScale, fontColor, thickness, lineType)
        if "SAM_pred_ins" in options:
            img_pred_ins = self.image.copy()
            img_num_i += 1
            img_base_x = D + int(img_num_i%all_img_x_num)*(H+D)
            img_base_y = D + int(img_num_i/all_img_x_num)*(W+D)
            for mask_id, pred_mask in enumerate(self.sam_pred_masks):
                img_pred_ins[pred_mask] = COLOR20[(mask_id+1)%20]
            all_img[img_base_x:img_base_x+H, img_base_y:img_base_y+W,:] = img_pred_ins
            if render_text:
                for mask_id, pred_mask in enumerate(self.sam_pred_masks):
                    first_pixel_x, first_pixel_y = (np.where(pred_mask == True)[0][0], np.where(pred_mask == True)[1][0])
                    text = PART_ID2NAME[self.sam_GAPart_id_pred[mask_id]]
                    cv2.putText(all_img, text, (img_base_y + first_pixel_y, img_base_x + first_pixel_x),font, fontScale, fontColor, thickness, lineType)
            cv2.putText(all_img, f"sam_pred_ins_{self.prompts}", (img_base_y + int(W/2) - 4*D, img_base_x + H + int(D/2)),font, fontScale, fontColor, thickness, lineType)      
        if "pc_pred_sem" in options:
            img_num_i += 1
            img_base_x = D + int(img_num_i%all_img_x_num)*(H+D)
            img_base_y = D + int(img_num_i/all_img_x_num)*(W+D)
            sem_pred_img = map2image(xyz, COLOR20[self.pcs_pred_sem%20], self.K, H, W)
            all_img[img_base_x:img_base_x+H, img_base_y:img_base_y+W,:] = sem_pred_img
            if render_text and self.pcs_pred_ins_masks is not None:
                for mask_id, pt_mask in enumerate(self.pcs_pred_ins_masks):
                    first_pixel_x, first_pixel_y = (self.pcs_pixel[np.where(pt_mask == True)[0][0]][0], self.pcs_pixel[np.where(pt_mask == True)[0][0]][1])
                    text = PART_ID2NAME[self.pcs_pred_ins_labels[mask_id].item()]
                    cv2.putText(all_img, text, (img_base_y + first_pixel_y, img_base_x + first_pixel_x),font, fontScale, fontColor, thickness, lineType)

            cv2.putText(all_img, f"pc_pred_sem", (img_base_y + int(W/2) - 4*D, img_base_x + H + int(D/2)),font, fontScale, fontColor, thickness, lineType)
        if "pc_pred_ins" in options:
            img_num_i += 1
            img_base_x = D + int(img_num_i%all_img_x_num)*(H+D)
            img_base_y = D + int(img_num_i/all_img_x_num)*(W+D)
            ins_pred_img = map2image(xyz, COLOR20[self.pcs_pred_ins%20], self.K, H, W)
            all_img[img_base_x:img_base_x+H, img_base_y:img_base_y+W,:] = ins_pred_img
            
            num_ins = self.pcs_pred_ins_masks.shape[0]
            for ins_i in range(num_ins):
                pred_mask = self.pcs_pred_ins_masks[ins_i]
                ins_pred_img_tmp = map2image(xyz[pred_mask], COLOR20[self.pcs_pred_ins[pred_mask]%20], self.K, H, W)
                cv2.imwrite(f"/data/haoran/LLM-GAPartNet/vision/output/part_masks/{ins_i}.png", ins_pred_img_tmp)
            
            if self.pcs_pred_ins_masks is not None and render_text:
                num_ins = self.pcs_pred_ins_masks.shape[0]
                for ins_i in range(num_ins):
                    pred_mask = self.pcs_pred_ins_masks[ins_i]
                    # import pdb; pdb.set_trace()
                    # ins_pred_img_tmp = map2image(xyz[pred_mask], COLOR20[self.pcs_pred_ins%20], self.K, H, W)
                    # cv2.imwrite(f"/data/haoran/LLM-GAPartNet/vision/output/part_masks/{ins_i}.png", ins_pred_img_tmp)
                    pc_id = np.where(pred_mask == True)[0][0]
                    first_pixel_x, first_pixel_y = self.pcs_pixel[pc_id]
                    text = PART_ID2NAME[self.pcs_pred_ins_labels[ins_i].item()]
                    cv2.putText(all_img, text, (img_base_y + first_pixel_y, img_base_x + first_pixel_x),font, fontScale, fontColor, thickness, lineType)

            cv2.putText(all_img, f"pc_pred_ins", (img_base_y + int(W/2) - 4*D, img_base_x + H + int(D/2)),font, fontScale, fontColor, thickness, lineType)
        if "pc_pred_npcs" in options:
            img_num_i += 1
            img_base_x = D + int(img_num_i%all_img_x_num)*(H+D)
            img_base_y = D + int(img_num_i/all_img_x_num)*(W+D)
            npcs_pred_img = map2image(xyz, (self.pcs_pred_npcs*255).astype(np.uint8), self.K, H, W)
            all_img[img_base_x:img_base_x+H, img_base_y:img_base_y+W,:] = npcs_pred_img
            cv2.putText(all_img, f"pc_pred_npcs", (img_base_y + int(W/2) - 4*D, img_base_x + H + int(D/2)),font, fontScale, fontColor, thickness, lineType)
        if "pc_pred_bbox" in options:
            img_num_i += 1
            img_base_x = D + int(img_num_i%all_img_x_num)*(H+D)
            img_base_y = D + int(img_num_i/all_img_x_num)*(W+D)
            pc_img = map2image(xyz, rgb*255.0, self.K, H, W)
            img_bbox_pred = pc_img.copy()
            draw_bbox(img_bbox_pred, self.pcs_pred_bbox[0], self.trans, self.K)
            all_img[img_base_x:img_base_x+H, img_base_y:img_base_y+W,:] = img_bbox_pred
            cv2.putText(all_img, f"pc_pred_bbox", (img_base_y + int(W/2) - 4*D, img_base_x + H + int(D/2)),font, fontScale, fontColor, thickness, lineType)
        if "masks_pred_bbox" in options:
            img_num_i += 1
            img_base_x = D + int(img_num_i%all_img_x_num)*(H+D)
            img_base_y = D + int(img_num_i/all_img_x_num)*(W+D)
            pc_img = map2image(xyz, rgb*255.0, self.K, H, W)
            img_bbox_pred = pc_img.copy()
            draw_bbox(img_bbox_pred, self.masks_pred_bbox[0], self.trans, self.K)
            for i, bbox in enumerate(self.masks_pred_bbox[0]):
                img_bbox_pred_new = pc_img.copy()
                draw_bbox(img_bbox_pred_new, [bbox], self.trans, self.K)
                cv2.imwrite(f"{output_root}/bbox_{i}_{PART_ID2NAME[self.given_masks_GAPart_ids[i].item()]}.png", img_bbox_pred_new)
                
            all_img[img_base_x:img_base_x+H, img_base_y:img_base_y+W,:] = img_bbox_pred
            cv2.putText(all_img, f"pc_pred_bbox", (img_base_y + int(W/2) - 4*D, img_base_x + H + int(D/2)),font, fontScale, fontColor, thickness, lineType)
        if "input_pc" in options:
            img_num_i += 1
            img_base_x = D + int(img_num_i%all_img_x_num)*(H+D)
            img_base_y = D + int(img_num_i/all_img_x_num)*(W+D)
            xyz_input, rgb_input = self.pcs_xyz_ball, self.pcs_rgb
            # import pdb; pdb.set_trace()
            # save_point_cloud_to_ply(xyz_input, rgb_input*255, f"{self.name}.ply", output_root)
            trans_gapartnet = np.array([ 1.26171422e+00, -6.60613179e-04,  4.20249701e-02,  4.23497820e+00])
            # trans_gapartnet = np.array([ 1.12502e+00,8.619621e-03,1.721e-01,4.357e+00])
            xyz_input = xyz_input * trans_gapartnet[0] + trans_gapartnet[1:4]
            K_GAPartNet = np.array([
                [1268.637939453125, 0, 400, 0], 
                [0, 1268.637939453125, 400, 0],
                [0, 0, 1, 0], 
                [0, 0, 0, 1]
            ], dtype=np.float32)
            
            input_pc_img = map2image(xyz_input, rgb_input*255.0, K_GAPartNet, 800, 800)
            input_pc_img = cv2.resize(input_pc_img, (W,H))
            # rgb to bgr
            input_pc_img = input_pc_img[...,::-1]
            all_img[img_base_x:img_base_x+H, img_base_y:img_base_y+W,:] = input_pc_img
            cv2.putText(all_img, f"input_pc", (img_base_y + int(W/2) - 4*D, img_base_x + H + int(D/2)),font, fontScale, fontColor, thickness, lineType)

        os.makedirs(output_root, exist_ok=True)
        Image.fromarray(all_img.astype(np.uint8)).save(f"{output_root}/{self.name}.png")

        return all_img
    
    def inference_sam(self, sam_mask_generator):
        import time
        s = time.time()
        # import pdb; pdb.set_trace()
        # sam_mask_generator.set_image(self.image)
        # masks, iou_predictions, low_res_masks = sam_mask_generator.predict_torch(point_coords = None,point_labels = None,boxes = None, multimask_output = True,)
        
        masks = sam_mask_generator.generate(self.image)
        self.sam_pred_masks = [mask['segmentation'] for mask in masks]
        e = time.time()
        print(f"Time for SAM: {e-s}; # Masks {len(masks)}")

    def get_dino_fea(self, dinov2_model, dino_root = None, use_inference = True):
        if os.path.exists(dino_root + "/" + self.name + ".npy"):
            self.image_dino_fea = np.load(dino_root + "/" + self.name + ".npy").reshape(50,50,-1)
        elif use_inference:
            self.image_dino_fea = dinov2_model.get_fea(self.image)[0].reshape(50,50,-1)
        else:
            print("No DINO feature available")
            
    def get_dino_sam_result(self, sam_model, dino_model, prompt = None, result_root = None, use_inference = True):
        if os.path.exists(result_root + "/" + prompt + "/" + self.name + "/sam_data.npy"):
            sam_pred_data = np.load(result_root + "/" + prompt + "/" + self.name + "/sam_data.npy", allow_pickle=True).item()
            self.masks_raw_reso = sam_pred_data["masks"]
            self.masks_low_reso = sam_pred_data["low_res_masks"]
            self.iou_predictions = sam_pred_data["iou_predictions"]
            self.transformed_boxes = sam_pred_data["transformed_boxes"]
        elif use_inference:
            self.inference_gounded_sam(prompt, dino_model = dino_model, sam_model = sam_model)
        else:
            print("No GroundedSAM results available!")
        return self.masks_raw_reso, self.masks_low_reso, self.iou_predictions, self.transformed_boxes
    
    def get_depth_info(self, depth_root):
        depth_path_1 = f"{depth_root}/{self.name}.npz"
        depth_path_2 = f"{depth_root}/{self.name}.exr"
        depth_path_3 = f"{depth_root}/{self.name}/depth.npz"
        if os.path.exists(depth_path_1):
            depth_img = np.load(depth_path_1)['depth_map']
            self.depth = depth_img
            self.obj_mask = self.depth!=0
        elif os.path.exists(depth_path_2):
            depth_img = cv2.imread(depth_path_2,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            depth_img = cv2.resize(depth_img, (self.image.shape[1], self.image.shape[0]))
            self.depth = depth_img[:,:,2]
            self.depth[np.where(self.depth>1)] = 0.
            self.depth[np.where(self.depth<-1)] = 0.
            self.depth[np.isnan(self.depth)] = 0.
            if self.obj_mask is not None:
                self.depth[np.where(self.obj_mask==False)] = 0.
        elif os.path.exists(depth_path_3):
            depth_img = np.load(depth_path_3)['depth_map']
            self.depth = depth_img
            self.obj_mask = self.depth!=0
        else:
            print("No depth available!")

    def get_meta_info(self, meta_input, mode):
        # mode: 0: GAPartNet, 1: Manip Data, 2: Real-world Data
        if mode == 0:
            meta_root = meta_input
            meta_path = f"{meta_root}/{self.name}.json"
            f = open(meta_path)
            meta = json.load(f)
            if "camera_intrinsic" in meta:
                self.K = np.array(meta['camera_intrinsic']).reshape(3, 3)
        elif mode == 1:
            self.K = np.array(
                [[813.1547241210938,0.0,256.0],
                [0.0,813.1547241210938,256.0],
                [0.0,0.0,1.0]])
            # meta_path = f"{meta_input}/{self.name}/episode.json"
            f = open(meta_input)
            meta = json.load(f)
            self.cam_p = np.array(meta["camera"]["p"])
            self.cam_q = np.array(meta["camera"]["q"])
        elif mode == 2:
            meta_path = meta_input
            f = open(meta_path)
            meta = json.load(f)
            self.K = np.array(meta['K']).reshape(3, 3).T
        else:
            print("wrong mode")

    def get_GAPart_gt(self, anno_root):
        if os.path.exists(f"{anno_root}/segmentation/{self.name}.npz"):  
            img_anno = query_part_anno(anno_root, self.name)
            self.gt_ins_ids = np.array([img_anno[i]["ins_id"] for i in range(len(img_anno))])
            self.gt_sem_ids = np.array([img_anno[i]["sem_id"] for i in range(len(img_anno))])
            self.gt_npcs_list = [img_anno[i]["npcs_map"] for i in range(len(img_anno))]
            self.gt_masks = np.array([img_anno[i]["mask"] for i in range(len(img_anno))])
            self.gt_bboxes = np.array([img_anno[i]["bbox"] for i in range(len(img_anno))])
            
            img_sem_map = np.ones(self.image.shape[:2])*(-1)
            img_ins_map = np.ones(self.image.shape[:2])*(-1)
            img_npcs_map = np.ones((self.image.shape[0], self.image.shape[1], 3))*(-1)
            
            for mask_id, mask in enumerate(self.gt_masks):
                img_sem_map[mask] = self.gt_sem_ids[mask_id]
                img_ins_map[mask] = self.gt_ins_ids[mask_id]
                img_npcs_map[mask] = self.gt_npcs_list[mask_id]
            self.img_sem_map = img_sem_map
            self.img_ins_map = img_ins_map
            self.img_npcs_map = img_npcs_map
            self.img_part_valid_mask = img_sem_map!=-1
        else:
            print("No GAPart ground truth available!")

    def get_pc(self, use_device_pcs = False, device_pcs_root = None, mode = 0):
        if use_device_pcs:
            points, self.pcs_all_rgb = read_pcs_from_ply(f"{device_pcs_root}/{self.name}.ply")
            points[:, 2] = -points[:,2]
            points[:, 1] = -points[:,1]
            self.pcs_all_xyz = points
            return
            
        point_cloud = []
        per_point_rgb = []

        per_point_idx = []
        rgb_image = self.image
        depth_map = self.depth
        K = self.K
        # import pdb; pdb.set_trace()
        # open3d.geometry.create_point_cloud_from_depth_image(depth, intrinsic, extrinsic=(with default value), depth_scale=1, depth_trunc=1000000.0, stride=1)
        for y_ in range(rgb_image.shape[0]):
            for x_ in range(rgb_image.shape[1]):
                # if self.img_sem_map is not None and self.img_sem_map[y_, x_] == -1:
                #     continue
                z_new = float(depth_map[y_, x_])
                x_new = (x_ - K[0, 2]) * z_new / K[0, 0]
                y_new = (y_ - K[1, 2]) * z_new / K[1, 1]
                point_cloud.append([x_new, y_new, z_new])
                per_point_rgb.append((rgb_image[y_, x_] / 255.0))
                per_point_idx.append([y_, x_])
        per_point_idx = np.array(per_point_idx)
        valid_point_mask = (depth_map!=0).reshape(-1)
        
        # if self.img_sem_map is not None:
        #     valid_point_mask = valid_point_mask & (self.img_sem_map.reshape(-1)!=-1)
        pcs = np.array(point_cloud)
        if mode == 1:
            pass
        elif mode == 2: # pay attention!
            pcs[:, 2] = -pcs[:,2]
            pcs[:, 1] = -pcs[:,1]
            # self.cam_e_mat = create_transformation_matrix(self.cam_p, self.cam_q)
            # pcs = transform_point_cloud(pcs.T, self.cam_e_mat)
            # valid_point_mask = ((depth_map!=0).reshape(-1)) & (np.array(pcs)[:,1] > -2.5)
            # save_point_cloud_to_ply(pcs[valid_point_mask],np.array(per_point_rgb)[valid_point_mask]*255,"test_.ply",".")
        
        self.pcs_all_xyz = pcs[valid_point_mask]
        self.pcs_all_rgb = np.array(per_point_rgb)[valid_point_mask][...,::-1]
        self.pcs_all_pixel = np.array(per_point_idx)[valid_point_mask]

    def get_downsampled_pc(self, sampled_num = 20000):
        if self.pcs_all_xyz.shape[0] < sampled_num:
            self.pcs_xyz = self.pcs_all_xyz
            self.pcs_rgb = self.pcs_all_rgb
            self.pcs_pixel = self.pcs_all_pixel
            import pdb; pdb.set_trace()
            
        else:
            if self.pcs_all_xyz.shape[0] > 4*sampled_num:
                ids = np.array(random.sample(range(self.pcs_all_xyz.shape[0]), int(4*sampled_num)))
                points_tmp_xyz = self.pcs_all_xyz[ids]
                points_tmp_rgb = self.pcs_all_rgb[ids]
                if self.pcs_all_pixel is not None:
                    points_tmp_pixel = self.pcs_all_pixel[ids]
            else:
                points_tmp_xyz = self.pcs_all_xyz
                points_tmp_rgb = self.pcs_all_rgb
                if self.pcs_all_pixel is not None:
                    points_tmp_pixel = self.pcs_all_pixel

            sampled_points_ids = FPS(points_tmp_xyz, sampled_num)
            self.pcs_xyz = points_tmp_xyz[sampled_points_ids]
            self.pcs_rgb = points_tmp_rgb[sampled_points_ids]
            if self.pcs_all_pixel is None:
                self.pcs_pixel = None
            else:
                self.pcs_pixel = points_tmp_pixel[sampled_points_ids]
        
        xyz, max_radius, center = WorldSpaceToBallSpace(self.pcs_xyz)
        trans = np.array([max_radius, center[0], center[1], center[2]])
        self.pcs_xyz_ball = xyz
        self.trans = trans
        if self.img_sem_map is not None:
            self.pcs_gt_sem = self.img_sem_map[self.pcs_pixel[:,0], self.pcs_pixel[:,1]]
        if self.img_ins_map is not None:
            self.pcs_gt_ins = self.img_ins_map[self.pcs_pixel[:,0], self.pcs_pixel[:,1]]
        if self.img_npcs_map is not None:
            self.pcs_gt_npcs = self.img_npcs_map[self.pcs_gt_ins.astype(np.int32)]
        if self.given_masks is not None:
            self.given_masks_pc = self.given_masks[:,self.pcs_pixel[:,0], self.pcs_pixel[:,1]]

    def inference_GAPartNet(self, gapartnet_model, use_sam_masks = False):
        if use_sam_masks:
            others = {}
            if self.sam_pred_masks is not None and self.sam_GAPart_id_pred is not None:
                
                pc_masks = np.array([mask[self.pcs_pixel[:,0],self.pcs_pixel[:,1]] for mask in self.sam_pred_masks])
                pc_masks_valid = pc_masks.sum(1) > 5 # Haoran: 5 is a magic number
                
                pc_masks = pc_masks[pc_masks_valid]
                # mask_labels = mask_labels[pc_masks_valid]
                mask_ids = self.sam_GAPart_id_pred[pc_masks_valid]
                mask_labels = np.array(self.sam_GAPart_label_pred)[pc_masks_valid]
                others["sam_masks_pc"] = pc_masks
                others["sam_GAPart_ids_pc"] = mask_ids
                others["sam_GAPart_labels_pc"] = mask_labels
            else:
                import pdb; pdb.set_trace()
            bboxes, sem_preds, npcs_maps, proposal_indices, proposal_offsets, proposal_sem_pred = \
                _inference_perception_model(gapartnet_model, [np.concatenate((self.pcs_xyz_ball, self.pcs_rgb), axis = 1)], self.name, others, use_sam_masks)
        else:
            bboxes, sem_preds, npcs_maps, proposal_indices, proposal_offsets, proposal_sem_pred = \
                _inference_perception_model(gapartnet_model, [np.concatenate((self.pcs_xyz_ball, self.pcs_rgb), axis = 1)], self.name)
        
        sem_preds = sem_preds.cpu().numpy()
        npcs_maps = npcs_maps.cpu().numpy()
        ins_map = np.zeros(self.pcs_xyz.shape[0]).astype(np.int32)
        if proposal_indices is not None and proposal_offsets is not None:
            proposal_indices = proposal_indices.cpu().numpy() # type:ignore
            proposal_offsets = proposal_offsets.cpu().numpy()
        
        if proposal_indices is not None and proposal_offsets is not None:
            for ins_i in range(len(proposal_offsets) - 1):
                ins_map[proposal_indices[proposal_offsets[ins_i]:proposal_offsets[ins_i + 1]]] = ins_i + 1
            ins_map = ins_map.astype(np.int32)
            ins_masks = np.zeros((len(proposal_offsets) - 1,self.pcs_xyz.shape[0])).astype(bool)
            for ins_i in range(0, len(proposal_offsets) - 1):
                ins_masks[ins_i, proposal_indices[proposal_offsets[ins_i]:proposal_offsets[ins_i + 1]]] = True
        else:
            ins_masks = None
        self.pcs_pred_sem = sem_preds
        self.pcs_pred_ins = ins_map
        self.pcs_pred_npcs = npcs_maps
        self.pcs_pred_bbox = bboxes
        self.pcs_pred_ins_masks = ins_masks
        self.pcs_pred_ins_labels = proposal_sem_pred

    def estimate_pose_GAPartNet(self, gapartnet_model):
        others = {}
        if self.sam_pred_masks is not None and self.sam_GAPart_id_pred is not None:
            
            pc_masks = np.array([mask[self.pcs_pixel[:,0],self.pcs_pixel[:,1]] for mask in self.sam_pred_masks])
            pc_masks_valid = pc_masks.sum(1) > 5 # Haoran: 5 is a magic number
            
            pc_masks = pc_masks[pc_masks_valid]
            # mask_labels = mask_labels[pc_masks_valid]
            mask_ids = self.sam_GAPart_id_pred[pc_masks_valid]
            mask_labels = np.array(self.sam_GAPart_label_pred)[pc_masks_valid]
            others["sam_masks_pc"] = pc_masks
            others["sam_GAPart_ids_pc"] = mask_ids
            others["sam_GAPart_labels_pc"] = mask_labels
        elif self.given_masks_pc is not None and self.given_masks_GAPart_ids is not None:            
            pc_masks = np.array([mask for mask in self.given_masks_pc])
            pc_masks_valid = pc_masks.sum(1) > 5 # Haoran: 5 is a magic number
            
            pc_masks = pc_masks[pc_masks_valid]
            # mask_labels = mask_labels[pc_masks_valid]
            mask_ids = self.given_masks_GAPart_ids[pc_masks_valid]
            mask_labels = np.array(self.given_masks_GAPart_ids)[pc_masks_valid]
            others["sam_masks_pc"] = pc_masks
            others["sam_GAPart_ids_pc"] = mask_ids
            others["sam_GAPart_labels_pc"] = mask_labels
        else:
            import pdb; pdb.set_trace()
        bboxes, npcs_maps, proposal_offsets = \
            _estimate_pose_with_masks(gapartnet_model, [np.concatenate((self.pcs_xyz_ball, self.pcs_rgb), axis = 1)], self.name, others)
        self.masks_pred_bbox = bboxes
        return bboxes

    def inference_fusion_GAPartNet(self, gapartnet_model):
        labels = self.GAPart_id_pred
        if self.masks_raw_reso is None or self.GAPart_id_pred is None:
            self.inference_GAPartNet(gapartnet_model)
            return
        self.pixel_mask_to_point_mask()
        thre = 0.95
        thre_mask = self.point_masks.max(1) < thre
        self.point_masks = self.point_masks[thre_mask]
        labels = labels[thre_mask]
        bboxes, sem_preds, npcs_maps, proposal_indices, proposal_offsets, proposal_sem_labels = \
            _inference_perception_model_with_masks(gapartnet_model, [np.concatenate((self.pcs_xyz_ball, self.pcs_rgb), axis = 1)], self.point_masks, labels)
        sem_preds = sem_preds.cpu().numpy()
        npcs_maps = npcs_maps.cpu().numpy()
        ins_map = np.zeros(self.pcs_xyz.shape[0])
        if proposal_indices is not None and proposal_offsets is not None:
            proposal_indices = proposal_indices.cpu().numpy() # type:ignore
            proposal_offsets = proposal_offsets.cpu().numpy()
            proposal_sem_labels = proposal_sem_labels.cpu().numpy()
            ins_masks = np.zeros((len(proposal_offsets) - 1,self.pcs_xyz.shape[0])).astype(bool)
            for ins_ip in range(0, len(proposal_offsets) - 1):
                ins_i = len(proposal_offsets) - 2 - ins_ip
                ins_map[proposal_indices[proposal_offsets[ins_i]:proposal_offsets[ins_i + 1]]] = ins_i + 1
            for ins_i in range(0, len(proposal_offsets) - 1):
                ins_masks[ins_i, proposal_indices[proposal_offsets[ins_i]:proposal_offsets[ins_i + 1]]] = True
        else:
            ins_masks = None
        ins_map = ins_map.astype(np.int32)
        self.pcs_pred_sem = sem_preds
        self.pcs_pred_ins = ins_map
        self.pcs_pred_ins_masks = ins_masks
        self.pcs_pred_ins_labels = proposal_sem_labels
        self.pcs_pred_npcs = npcs_maps
        self.pcs_pred_bbox = bboxes
        
    def seg_obj(self, obj_name, dino_model, sam_model):
        results = inference_gounded_sam(obj_name,img = self.image,name = self.name,output_dir="",dino_model=dino_model,sam_model=sam_model,)
        sam_pred_data = results[0]
        if sam_pred_data == None:
            self.obj_mask = np.ones((self.image.shape[0], self.image.shape[1])).astype(bool)
        else:
            obj_idx = np.argmax(sam_pred_data["masks"].sum(-1).sum(-1).sum(-1))
            self.obj_mask = sam_pred_data["masks"][obj_idx][0]

    def pixel_mask_to_point_mask(self):
        pixel_mask = self.masks_raw_reso[:,0,:,:]
        self.point_masks = pixel_mask[:,self.pcs_pixel[:,0], self.pcs_pixel[:,1]]
        
    def save_pre_inference_pc(self, output_root):
        if self.pcs_xyz_ball is not None:
            save_point_cloud_to_ply(self.pcs_xyz_ball, self.pcs_rgb, f"{self.name}.ply", output_root)
        else:
            import pdb; pdb.set_trace()    

    def from_pc_to_img(self):
        H,W = self.image_reso
        self.image = map2image(self.pcs_all_xyz, self.pcs_all_rgb*255.0, self.K, H, W)
        return self.image

    def get_pc_pixel_ids(self):
        self.pcs_pixel = get_point2image(self.pcs_xyz_ball, self.trans, self.K)
        # H,W = self.image_reso
        # image_ = map2image(self.pcs_xyz, self.pcs_rgb*255.0, self.K, H, W)
        # import pdb; pdb.set_trace()
        return self.pcs_pixel
        

def load_models(USE_GROUNDING, USE_DINOV2_INFERENCE, USE_DINO_SAM_INFERENCE, 
                USE_PERCEPTION, USE_SAM, base_path="./", GAPARTNET_CKPT_PATH = "",
                USE_2D_FOR_PERCEPTION = False):
    # get KNN classifier
    if USE_GROUNDING:
        TRAIN_FEA_ROOT = "ckpts/fea_data_all_relabel.npy"
        train_feas, train_cat, cat_ids, splits, feas = load_data_single_file(data_root = TRAIN_FEA_ROOT)
        classifier = KNN_classifier(train_feas, train_cat, 5)
        del train_feas, train_cat, cat_ids, splits, feas
    else:
        classifier = None
        
    # get dinov2 model
    if USE_DINOV2_INFERENCE:
        dinov2_model = DINOV2()
    else:
        dinov2_model = None

    # get dino_sam model
    if USE_DINO_SAM_INFERENCE:
        dino_model, sam_model = load_dino_sam_model(
                                    pjoin(base_path, "GroundedSAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py"),
                                    pjoin(base_path, "ckpts/groundingdino_swinb_cogcoor.pth"),
                                    pjoin(base_path, "ckpts/sam_vit_h_4b8939.pth")
                                )
    else:
        dino_model, sam_model = None, None
        
    if USE_PERCEPTION:
        perception_model = _load_perception_model(
            ckpt_path=GAPARTNET_CKPT_PATH, USE_2D_FOR_PERCEPTION = USE_2D_FOR_PERCEPTION
        )
    else:
        perception_model = None
        
    if USE_SAM:
        sam = sam_model_registry["vit_h"](checkpoint="ckpts/sam_vit_h_4b8939.pth").to(device="cuda")
        sam_mask_generator = SamAutomaticMaskGenerator(sam)
    else:
        sam_mask_generator = None
        
    print("finish loading models")
     
    return classifier, dinov2_model, dino_model, sam_model, perception_model, sam_mask_generator

def estimate_joint_angle(
        img_obj_1: ObjIns,
        img_obj_2: ObjIns,
        visu: bool = True,
        visu_root: str = "",
        options: List[str] = ["img1", "img2","pcs1", "pcs2", "joint", "joint_ransac"],
        sample_number: int = 500,
        joint_type: str = "revolute"
    ):
    '''
    input: two frame of point cloud (only part, please segment first)
    output:
        axis: joint axis direction
        axis_t: joint axis translation
        angle: angle between the two point cloud
    '''
    
    pc1 = img_obj_1.pcs_xyz
    pc2 = img_obj_2.pcs_xyz
    
    M = pc2.mean(axis=0)
    S = ((pc2 - M).max(axis = 0) - (pc2 - M).min(axis = 0)).max()
    X = (pc1 - M)/S
    Y = (pc2 - M)/S
   
    X = np.array(random.sample(list(X), sample_number))
    Y = np.array(random.sample(list(Y), sample_number))
    
    

    bbox_trans, scale, rotation, translation, out_transform, best_inlier_idx = estimate_pose_from_npcs(X, Y)
    r_ = R.from_matrix(rotation)
    axis_angle = r_.as_rotvec()
    axis = axis_angle / np.linalg.norm(axis_angle)
    angle = np.linalg.norm(axis_angle)
    r = R.from_rotvec((np.pi/2-angle/2) *  axis.squeeze())
    axis_t_norm = (np.dot(translation, r.as_matrix())/2) / np.sin(angle/2)
    axis_t = axis_t_norm.squeeze()*S + M.squeeze()
    axis_ransac = axis
    axis_t_ransac = axis_t
    angle_ransac = angle
    

    reg = RigidRegistration(**{'X': X, 'Y': Y})
    
    
    reg.register()

    r_ = R.from_matrix(reg.R)
    axis_angle = r_.as_rotvec()
    axis = axis_angle / np.linalg.norm(axis_angle)
    angle = np.linalg.norm(axis_angle)
    r = R.from_rotvec((np.pi/2-angle/2) *  axis.squeeze())
    axis_t_norm = (np.dot(reg.t, r.as_matrix())/2) / np.sin(angle/2)
    axis_t = axis_t_norm.squeeze()*S + M.squeeze()


    if visu:
        D = 40 # delta
        # L = 800 # length for an image
        H,W = img_obj_1.image.shape[:2]
        all_img_x_num = 2
        all_img_y_num = math.ceil(len(options)/(all_img_x_num + 0.0))
        all_img = np.ones((D+(all_img_x_num)*(H+D), D+(all_img_y_num)*(W+D),3))*255
        img_num_i = -1
        
        
        if "img1" in options:
            img_num_i += 1
            img_base_x = D + int(img_num_i%all_img_x_num)*(H+D)
            img_base_y = D + int(img_num_i/all_img_x_num)*(W+D)
            all_img[img_base_x:img_base_x+H, img_base_y:img_base_y+W,:] = img_obj_1.image
            cv2.putText(all_img, f"img1", (D + int(W/2) - 4*D, D + H + int(D/2)),font, fontScale, fontColor, thickness, lineType)
        if "img2" in options:
            img_num_i += 1
            img_base_x = D + int(img_num_i%all_img_x_num)*(H+D)
            img_base_y = D + int(img_num_i/all_img_x_num)*(W+D)
            all_img[img_base_x:img_base_x+H, img_base_y:img_base_y+W,:] = img_obj_2.image
            cv2.putText(all_img, f"img2", (D + int(W/2) - 4*D, D + H + int(D/2)),font, fontScale, fontColor, thickness, lineType)
        if "pcs1" in options:
            xyz = img_obj_1.pcs_xyz
            rgb = img_obj_1.pcs_rgb
            pc_img = map2image(xyz, rgb*255.0, img_obj_1.K, H, W)
            img_num_i += 1
            img_base_x = D + int(img_num_i%all_img_x_num)*(H+D)
            img_base_y = D + int(img_num_i/all_img_x_num)*(W+D)
            all_img[img_base_x:img_base_x+H, img_base_y:img_base_y+W,:] = pc_img
            cv2.putText(all_img, f"pc1", (img_base_y + int(W/2) - 4*D, img_base_x + H + int(D/2)),font, fontScale, fontColor, thickness, lineType)
        if "pcs2" in options:
            xyz = img_obj_2.pcs_xyz
            rgb = img_obj_2.pcs_rgb
            pc_img = map2image(xyz, rgb*255.0, img_obj_2.K, H, W)
            img_num_i += 1
            img_base_x = D + int(img_num_i%all_img_x_num)*(H+D)
            img_base_y = D + int(img_num_i/all_img_x_num)*(W+D)
            all_img[img_base_x:img_base_x+H, img_base_y:img_base_y+W,:] = pc_img
            cv2.putText(all_img, f"pc2", (img_base_y + int(W/2) - 4*D, img_base_x + H + int(D/2)),font, fontScale, fontColor, thickness, lineType)
        if "joint" in options:
            img_joint = img_obj_1.image.copy()
            K = img_obj_1.K
            joint_1_x = axis_t[0] - axis[0]
            joint_1_y = axis_t[1] - axis[1]
            joint_1_z = axis_t[2] - axis[2]
            joint_1_x_img = (np.around(joint_1_x * K[0][0] / joint_1_z + K[0][2])).astype(dtype=int)
            joint_1_y_img = (np.around(joint_1_y * K[1][1] / joint_1_z + K[1][2])).astype(dtype=int)
            joint_2_x = axis_t[0] + axis[0]
            joint_2_y = axis_t[1] + axis[1]
            joint_2_z = axis_t[2] + axis[2]
            joint_2_x_img = (np.around(joint_2_x * K[0][0] / joint_2_z + K[0][2])).astype(dtype=int)
            joint_2_y_img = (np.around(joint_2_y * K[1][1] / joint_2_z + K[1][2])).astype(dtype=int)
            color = [255,0,255]
            cv2.line(img_joint, [joint_1_x_img, joint_1_y_img], [joint_2_x_img, joint_2_y_img], color=(int(color[0]),int(color[1]),int(color[2])),thickness=2)
            img_num_i += 1
            img_base_x = D + int(img_num_i%all_img_x_num)*(H+D)
            img_base_y = D + int(img_num_i/all_img_x_num)*(W+D)
            all_img[img_base_x:img_base_x+H, img_base_y:img_base_y+W,:] = img_joint
            cv2.putText(all_img, f"joint", (img_base_y + int(W/2) - 4*D, img_base_x + H + int(D/2)),font, fontScale, fontColor, thickness, lineType)
        if "joint_ransac" in options:
            axis = axis_ransac
            axis_t = axis_t_ransac
            angle = angle_ransac
            img_joint = img_obj_1.image.copy()
            K = img_obj_1.K
            joint_1_x = axis_t[0] - axis[0]
            joint_1_y = axis_t[1] - axis[1]
            joint_1_z = axis_t[2] - axis[2]
            joint_1_x_img = (np.around(joint_1_x * K[0][0] / joint_1_z + K[0][2])).astype(dtype=int)
            joint_1_y_img = (np.around(joint_1_y * K[1][1] / joint_1_z + K[1][2])).astype(dtype=int)
            joint_2_x = axis_t[0] + axis[0]
            joint_2_y = axis_t[1] + axis[1]
            joint_2_z = axis_t[2] + axis[2]
            joint_2_x_img = (np.around(joint_2_x * K[0][0] / joint_2_z + K[0][2])).astype(dtype=int)
            joint_2_y_img = (np.around(joint_2_y * K[1][1] / joint_2_z + K[1][2])).astype(dtype=int)
            color = [255,0,255]
            cv2.line(img_joint, [joint_1_x_img, joint_1_y_img], [joint_2_x_img, joint_2_y_img], color=(int(color[0]),int(color[1]),int(color[2])),thickness=2)
            img_num_i += 1
            img_base_x = D + int(img_num_i%all_img_x_num)*(H+D)
            img_base_y = D + int(img_num_i/all_img_x_num)*(W+D)
            all_img[img_base_x:img_base_x+H, img_base_y:img_base_y+W,:] = img_joint
            cv2.putText(all_img, f"joint_ransac", (img_base_y + int(W/2) - 4*D, img_base_x + H + int(D/2)),font, fontScale, fontColor, thickness, lineType)

        os.makedirs(visu_root, exist_ok=True)
        cv2.imwrite(f"{visu_root}/{img_obj_1.name}_joint_angle.png", all_img)

    return axis, axis_t, angle