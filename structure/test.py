import numpy as np
import torch

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
PART_NAME2ID = {
    'others':             0,
    'line_fixed_handle':  1,
    'round_fixed_handle': 2,
    'slider_button':      3,
    'hinge_door':         4,
    'slider_drawer':      5,
    'slider_lid':         6,
    'hinge_lid':          7,
    'hinge_knob':         8,
    'revolute_handle':    9,
}

data_root = "/home/birdswimming/HW/LLM-GAPartNet/vision/data/fea_data_all.npy"
target_root = "/home/birdswimming/HW/LLM-GAPartNet/vision/data/fea_data_all_relabel.npy"
data = np.load(data_root, allow_pickle=True).item()
feas = data["feas"]
obj_codes = data["obj_codes"]
part_ids = data["part_ids"]
cat_ids = data["cat_names"]
splits = data["splits"]
# feas, obj_codes, cat_ids, part_ids, splits = load_new_data(data_root = FEA_ROOT, type = type_fea)
data["cat_names"] = np.array([PART_ID2NAME_OLD[int(cat_id)] for cat_id in  cat_ids])
data["cat_ids"] = np.array([PART_NAME2ID[name] for name in data["cat_names"]])

import pdb; pdb.set_trace()

np.save(target_root, data, allow_pickle=True)