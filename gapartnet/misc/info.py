import math
from typing import Tuple

import torch

OBJECT_NAME2ID = {
    # seen category
    "Box": 0,
    "Remote": 1,
    "Microwave": 2,
    "Camera": 3,
    "Dishwasher": 4,
    "WashingMachine": 5,
    "CoffeeMachine": 6,
    "Toaster": 7,
    "StorageFurniture": 8,
    "AKBBucket": 9, # akb48
    "AKBBox": 10, # akb48
    "AKBDrawer": 11, # akb48
    "AKBTrashCan": 12, # akb48
    "Bucket": 13, # new
    "Keyboard": 14, # new
    "Printer": 15, # new
    "Toilet": 16, # new
    # unseen category
    "KitchenPot": 17,
    "Safe": 18,
    "Oven": 19,
    "Phone": 20,
    "Refrigerator": 21,
    "Table": 22,
    "TrashCan": 23,
    "Door": 24,
    "Laptop": 25,
    "Suitcase": 26, # new
}

TARGET_PARTS = [
    'others',
    'line_fixed_handle',
    'round_fixed_handle',
    'slider_button',
    'hinge_door',
    'slider_drawer',
    'slider_lid',
    'hinge_lid',
    'hinge_knob',
    'revolute_handle'
]

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


TARGET_PARTS = [
    'others',
    'line_fixed_handle',
    'round_fixed_handle',
    'slider_button',
    'hinge_door',
    'slider_drawer',
    'slider_lid',
    'hinge_lid',
    'hinge_knob',
    'revolute_handle',
]

TARGET_IDX = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
]
PI = math.pi
SYMMETRY_MATRIX = [
    # type 0
    [
        [
            [ 1.0,     0,     0],
            [ 0,     1.0,     0],
            [ 0,       0,   1.0],
        ],
        [
            [ 1.0,     0,     0],
            [ 0,     1.0,     0],
            [ 0,       0,   1.0],
        ],
    ],

    # type 1
    [
        [
            [ 1.0,     0,     0],
            [ 0,     1.0,     0],
            [ 0,       0,   1.0],
        ],
        [
            [-1.0,     0,     0],
            [ 0,    -1.0,     0],
            [ 0,       0,   1.0],
        ],
    ],

    # type 2
    [
        [
            [ 1.0,     0,     0],
            [ 0,     1.0,     0],
            [ 0,       0,   1.0],
        ],
        [
            [-1.0,     0,     0],
            [ 0,     1.0,     0],
            [ 0,       0,  -1.0],
        ],
    ],

    # type 3
    [
        [
            [ 1.0,     0,     0],
            [ 0,     1.0,     0],
            [ 0,       0,   1.0],
        ],
        [
            [ math.cos(PI/6), math.sin(PI/6),          0],
            [-math.sin(PI/6), math.cos(PI/6),          0],
            [              0,              0,        1.0]
        ],
        [
            [ math.cos(PI*2/6), math.sin(PI*2/6),          0],
            [-math.sin(PI*2/6), math.cos(PI*2/6),          0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*3/6), math.sin(PI*3/6),          0],
            [-math.sin(PI*3/6), math.cos(PI*3/6),          0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*4/6), math.sin(PI*4/6),          0],
            [-math.sin(PI*4/6), math.cos(PI*4/6),          0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*5/6), math.sin(PI*5/6),          0],
            [-math.sin(PI*5/6), math.cos(PI*5/6),          0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*6/6), math.sin(PI*6/6),         0],
            [-math.sin(PI*6/6), math.cos(PI*6/6),         0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*7/6), math.sin(PI*7/6),         0],
            [-math.sin(PI*7/6), math.cos(PI*7/6),         0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*8/6), math.sin(PI*8/6),         0],
            [-math.sin(PI*8/6), math.cos(PI*8/6),         0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*9/6), math.sin(PI*9/6),         0],
            [-math.sin(PI*9/6), math.cos(PI*9/6),         0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*10/6), math.sin(PI*10/6),         0],
            [-math.sin(PI*10/6), math.cos(PI*10/6),         0],
            [                 0,                 0,        1.0]
        ],
        [
            [ math.cos(PI*11/6), math.sin(PI*11/6),         0],
            [-math.sin(PI*11/6), math.cos(PI*11/6),         0],
            [                 0,                 0,        1.0]
        ],
    ],

    # type 4
    [
        [
            [ 1.0,     0,     0],
            [ 0,     1.0,     0],
            [ 0,       0,   1.0],
        ],
        [
            [ math.cos(PI/6), math.sin(PI/6),          0],
            [-math.sin(PI/6), math.cos(PI/6),          0],
            [              0,              0,        1.0]
        ],
        [
            [ math.cos(PI*2/6), math.sin(PI*2/6),          0],
            [-math.sin(PI*2/6), math.cos(PI*2/6),          0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*3/6), math.sin(PI*3/6),          0],
            [-math.sin(PI*3/6), math.cos(PI*3/6),          0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*4/6), math.sin(PI*4/6),          0],
            [-math.sin(PI*4/6), math.cos(PI*4/6),          0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*5/6), math.sin(PI*5/6),        0],
            [-math.sin(PI*5/6), math.cos(PI*5/6),        0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*6/6), math.sin(PI*6/6),        0],
            [-math.sin(PI*6/6), math.cos(PI*6/6),        0],
            [                0,               0,        1.0]
        ],
        [
            [ math.cos(PI*7/6), math.sin(PI*7/6),        0],
            [-math.sin(PI*7/6), math.cos(PI*7/6),        0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*8/6), math.sin(PI*8/6),        0],
            [-math.sin(PI*8/6), math.cos(PI*8/6),        0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*9/6), math.sin(PI*9/6),        0],
            [-math.sin(PI*9/6), math.cos(PI*9/6),        0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*10/6), math.sin(PI*10/6),        0],
            [-math.sin(PI*10/6), math.cos(PI*10/6),        0],
            [                 0,                 0,        1.0]
        ],
        [
            [ math.cos(PI*11/6), math.sin(PI*11/6),        0],
            [-math.sin(PI*11/6), math.cos(PI*11/6),        0],
            [                 0,                 0,        1.0]
        ],
        ######################  inverse  ######################
        [
            [ math.sin(PI/6), math.cos(PI/6),        0],
            [ math.cos(PI/6), -math.sin(PI/6),        0],
            [              0,              0,       -1.0]
        ],
        [
            [ math.sin(PI*2/6), math.cos(PI*2/6),        0],
            [ math.cos(PI*2/6), -math.sin(PI*2/6),        0],
            [                0,                0,       -1.0]
        ],
        [
            [ math.sin(PI*3/6), math.cos(PI*3/6),        0],
            [ math.cos(PI*3/6), -math.sin(PI*3/6),        0],
            [                0,                0,       -1.0]
        ],
        [
            [ math.sin(PI*4/6), math.cos(PI*4/6),        0],
            [ math.cos(PI*4/6), -math.sin(PI*4/6),        0],
            [                0,                0,       -1.0]
        ],
        [
            [ math.sin(PI*5/6), math.cos(PI*5/6),        0],
            [ math.cos(PI*5/6), -math.sin(PI*5/6),        0],
            [                0,                0,       -1.0]
        ],
        [
            [ math.sin(PI*6/6), math.cos(PI*6/6),        0],
            [ math.cos(PI*6/6), -math.sin(PI*6/6),        0],
            [                0,                0,       -1.0]
        ],
        [
            [ math.sin(PI*7/6), math.cos(PI*7/6),        0],
            [ math.cos(PI*7/6), -math.sin(PI*7/6),        0],
            [                0,                0,       -1.0]
        ],
        [
            [ math.sin(PI*8/6), math.cos(PI*8/6),        0],
            [ math.cos(PI*8/6), -math.sin(PI*8/6),        0],
            [                0,                0,       -1.0]
        ],
        [
            [ math.sin(PI*9/6), math.cos(PI*9/6),        0],
            [ math.cos(PI*9/6), -math.sin(PI*9/6),        0],
            [                0,                0,       -1.0]
        ],
        [
            [ math.sin(PI*10/6), math.cos(PI*10/6),        0],
            [ math.cos(PI*10/6), -math.sin(PI*10/6),        0],
            [                 0,                 0,       -1.0]
        ],
        [
            [ math.sin(PI*11/6), math.cos(PI*11/6),        0],
            [ math.cos(PI*11/6), -math.sin(PI*11/6),        0],
            [                 0,                 0,       -1.0]
        ],
        [
            [ math.sin(PI*12/6), math.cos(PI*12/6),        0],
            [ math.cos(PI*12/6), -math.sin(PI*12/6),        0],
            [                 0,                 0,       -1.0]
        ],
    ],
]


def get_symmetry_matrix() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # type 0 / 1 / 2
    sm_1 = torch.as_tensor(SYMMETRY_MATRIX[:3], dtype=torch.float32)
    # type 3
    sm_2 = torch.as_tensor(SYMMETRY_MATRIX[3:4], dtype=torch.float32)
    # type 4
    sm_3 = torch.as_tensor(SYMMETRY_MATRIX[4:5], dtype=torch.float32)

    return sm_1, sm_2, sm_3


# import numpy as np
# OTHER_COLOR = [230, 230, 230]

# COLOR20 = np.array(
#     [[0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
#      [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
#      [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
#      [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190]])

# COLOR40 = np.array(
#         [[175,58,119], [81,175,144], [184,70,74], [40,116,79], [184,134,219], [130,137,46], [110,89,164], [92,135,74], [220,140,190], [94,103,39],
#         [144,154,219], [160,86,40], [67,107,165], [194,170,104], [162,95,150], [143,110,44], [146,72,105], [225,142,106], [162,83,86], [227,124,143],[88,170,108], [174,105,226], [78,194,83], [198,62,165], [133,188,52], [97,101,219], [190,177,52], [139,65,168], [75,202,137], [225,66,129],
#         [68,135,42], [226,116,210], [146,186,98], [68,105,201], [219,148,53], [85,142,235], [212,85,42], [78,176,223], [221,63,77], [68,195,195]
#         ])

# SEMANTIC_COLOR = np.array(
#     [OTHER_COLOR, # others
#      [202, 145, 99],
#      [203, 202, 102],
#      [140, 203, 103],
#      [109, 189, 205],
#      [112,157,206],
#      [128,129,212],
#      [175,124,211],
#      [208,118,167]
#      [203,140,103],

#     ])

# # SEMANTIC_COLOR = np.array(
# #     [OTHER_COLOR, # others
# #      [140,  103,203,],
# #      [140,  103,203,],
# #      [140,  103,203,],
# #      [140,  103,203,],
# #      [140,  103,203,],
# #      [140,  103,203,],
# #      [140,  103,203,],
# #      [140,  103,203,],

# #     ])
# SEMANTIC_IDX2NAME = {
#     0: 'others',
#     1: 'line_fixed_handle',
#     2: 'round_fixed_handle',
#     3: 'slider_button',
#     4: 'hinge_door',
#     5: 'slider_drawer',
#     6: 'slider_lid',
#     7: 'hinge_lid',
#     8: 'hinge_knob',
#     9: 'revolute_handle',
    
# }


# SYMMETRY_MATRIX_INDEX = [0,1,2,2,4,0,2,4,3,1]


# import math
# PI = math.pi

# SYMMETRY_MATRIX = {
#     1:[[
#         [-1.0,     0,     0],
#         [ 0,    -1.0,     0],
#         [ 0,       0,   1.0]
#     ]],
#     2:[
#         [
#             [ math.cos(PI/6),math.sin(PI/6),          0],
#             [-math.sin(PI/6),math.cos(PI/6),          0],
#             [              0,             0,        1.0]
#         ],
#         [
#             [ math.cos(PI*2/6),math.sin(PI*2/6),          0],
#             [-math.sin(PI*2/6),math.cos(PI*2/6),          0],
#             [                0,               0,        1.0]
#         ],
#         [  
#             [ math.cos(PI*3/6),math.sin(PI*3/6),          0],
#             [-math.sin(PI*3/6),math.cos(PI*3/6),          0],
#             [                0,               0,        1.0]
#         ],
#         [
#             [ math.cos(PI*4/6),math.sin(PI*4/6),          0],
#             [-math.sin(PI*4/6),math.cos(PI*4/6),          0],
#             [                0,               0,        1.0]
#         ],
#         [
#             [ math.cos(PI*5/6),math.sin(PI*5/6),          0],
#             [-math.sin(PI*5/6),math.cos(PI*5/6),          0],
#             [                0,               0,        1.0]
#         ],
#         [
#             [ math.cos(PI*6/6),math.sin(PI*6/6),         0],
#             [-math.sin(PI*6/6),math.cos(PI*6/6),         0],
#             [                0,               0,        1.0]
#         ],
#         [
#             [ math.cos(PI*7/6),math.sin(PI*7/6),         0],
#             [-math.sin(PI*7/6),math.cos(PI*7/6),         0],
#             [                0,               0,        1.0]
#         ],
#         [
#             [ math.cos(PI*8/6),math.sin(PI*8/6),         0],
#             [-math.sin(PI*8/6),math.cos(PI*8/6),         0],
#             [                0,               0,        1.0]
#         ],
#         [
#             [ math.cos(PI*9/6),math.sin(PI*9/6),         0],
#             [-math.sin(PI*9/6),math.cos(PI*9/6),         0],
#             [                0,               0,        1.0]
#         ],
#         [
#             [ math.cos(PI*10/6),math.sin(PI*10/6),         0],
#             [-math.sin(PI*10/6),math.cos(PI*10/6),         0],
#             [                 0,                0,        1.0]
#         ],
#         [
#             [ math.cos(PI*11/6),math.sin(PI*11/6),         0],
#             [-math.sin(PI*11/6),math.cos(PI*11/6),         0],
#             [                 0,                0,        1.0]
#         ]
#     ],
    
#     3:[
#         [
#             [ math.cos(PI/6),math.sin(PI/6),          0],
#             [-math.sin(PI/6),math.cos(PI/6),          0],
#             [              0,             0,        1.0]
#         ],
#         [
#             [ math.cos(PI*2/6),math.sin(PI*2/6),          0],
#             [-math.sin(PI*2/6),math.cos(PI*2/6),          0],
#             [                0,               0,        1.0]
#         ],
#         [
#             [ math.cos(PI*3/6),math.sin(PI*3/6),          0],
#             [-math.sin(PI*3/6),math.cos(PI*3/6),          0],
#             [                0,               0,        1.0]
#         ],
#         [
#             [ math.cos(PI*4/6),math.sin(PI*4/6),          0],
#             [-math.sin(PI*4/6),math.cos(PI*4/6),          0],
#             [                0,               0,        1.0]
#         ],
#         [
#             [ math.cos(PI*5/6),math.sin(PI*5/6),        0],
#             [-math.sin(PI*5/6),math.cos(PI*5/6),        0],
#             [                0,               0,        1.0]
#         ],
#         [
#             [ math.cos(PI*6/6),math.sin(PI*6/6),        0],
#             [-math.sin(PI*6/6),math.cos(PI*6/6),        0],
#             [                0,               0,        1.0]
#         ],
#         [
#             [ math.cos(PI*7/6),math.sin(PI*7/6),        0],
#             [-math.sin(PI*7/6),math.cos(PI*7/6),        0],
#             [                0,               0,        1.0]
#         ],
#         [
#             [ math.cos(PI*8/6),math.sin(PI*8/6),        0],
#             [-math.sin(PI*8/6),math.cos(PI*8/6),        0],
#             [                0,               0,        1.0]
#         ],
#         [
#             [ math.cos(PI*9/6),math.sin(PI*9/6),        0],
#             [-math.sin(PI*9/6),math.cos(PI*9/6),        0],
#             [                0,               0,        1.0]
#         ],
#         [
#             [ math.cos(PI*10/6),math.sin(PI*10/6),        0],
#             [-math.sin(PI*10/6),math.cos(PI*10/6),        0],
#             [                 0,                0,        1.0]
#         ],
#         [
#             [ math.cos(PI*11/6),math.sin(PI*11/6),        0],
#             [-math.sin(PI*11/6),math.cos(PI*11/6),        0],
#             [                 0,                0,        1.0]
#         ],

#         ######################  inverse  ###################### 

#         [
#             [ math.sin(PI/6),math.cos(PI/6),        0],
#             [ math.cos(PI/6),-math.sin(PI/6),        0],
#             [              0,             0,       -1.0]
#         ],
#         [
#             [ math.sin(PI*2/6),math.cos(PI*2/6),        0],
#             [ math.cos(PI*2/6),-math.sin(PI*2/6),        0],
#             [                0,               0,       -1.0]
#         ],
#         [
#             [ math.sin(PI*3/6),math.cos(PI*3/6),        0],
#             [ math.cos(PI*3/6),-math.sin(PI*3/6),        0],
#             [                0,               0,       -1.0]
#         ],
#         [
#             [ math.sin(PI*4/6),math.cos(PI*4/6),        0],
#             [ math.cos(PI*4/6),-math.sin(PI*4/6),        0],
#             [                0,               0,       -1.0]
#         ],
#         [
#             [ math.sin(PI*5/6),math.cos(PI*5/6),        0],
#             [ math.cos(PI*5/6),-math.sin(PI*5/6),        0],
#             [                0,               0,       -1.0]
#         ],
#         [
#             [ math.sin(PI*6/6),math.cos(PI*6/6),        0],
#             [ math.cos(PI*6/6),-math.sin(PI*6/6),        0],
#             [                0,               0,       -1.0]
#         ],
#         [
#             [ math.sin(PI*7/6),math.cos(PI*7/6),        0],
#             [ math.cos(PI*7/6),-math.sin(PI*7/6),        0],
#             [                0,               0,       -1.0]
#         ],
#         [
#             [ math.sin(PI*8/6),math.cos(PI*8/6),        0],
#             [ math.cos(PI*8/6),-math.sin(PI*8/6),        0],
#             [                0,               0,       -1.0]
#         ],
#         [
#             [ math.sin(PI*9/6),math.cos(PI*9/6),        0],
#             [ math.cos(PI*9/6),-math.sin(PI*9/6),        0],
#             [                0,               0,       -1.0]
#         ],
#         [
#             [ math.sin(PI*10/6),math.cos(PI*10/6),        0],
#             [ math.cos(PI*10/6),-math.sin(PI*10/6),        0],
#             [                 0,                0,       -1.0]
#         ],
#         [
#             [ math.sin(PI*11/6),math.cos(PI*11/6),        0],
#             [ math.cos(PI*11/6),-math.sin(PI*11/6),        0],
#             [                 0,                0,       -1.0]
#         ],
#         [
#             [ math.sin(PI*12/6),math.cos(PI*12/6),        0],
#             [ math.cos(PI*12/6),-math.sin(PI*12/6),        0],
#             [                 0,                0,       -1.0]
#         ]

#     ],
#     4:[[
#         [-1.0,     0,     0],
#         [ 0,     1.0,     0],
#         [ 0,       0,  -1.0]
#     ]],
    
# }
