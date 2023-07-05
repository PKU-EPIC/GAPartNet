import os
import sys
from os.path import join as pjoin
import json
import numpy as np


# TODO: Set the path to the dataset
DATASET_PATH = './dataset/partnet_all_annotated_new'

SAVE_PATH = './example_rendered'

VISU_SAVE_PATH = './visu'

ID_PATH = './meta/partnet_all_id_list.txt'

TARGET_GAPARTS = [
    'line_fixed_handle', 'round_fixed_handle', 'slider_button', 'hinge_door', 'slider_drawer',
    'slider_lid', 'hinge_lid', 'hinge_knob', 'hinge_handle'
]

OBJECT_CATEGORIES = [
    'Box', 'Camera', 'CoffeeMachine', 'Dishwasher', 'KitchenPot', 'Microwave', 'Oven', 'Phone', 'Refrigerator',
    'Remote', 'Safe', 'StorageFurniture', 'Table', 'Toaster', 'TrashCan', 'WashingMachine', 'Keyboard', 'Laptop', 'Door', 'Printer',
    'Suitcase', 'Bucket', 'Toilet'
]

CAMERA_POSITION_RANGE = {
    'Box': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.5
    }],
    'Camera': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.5
    }, {
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': -60.0,
        'phi_max': 60.0,
        'distance_min': 3.5,
        'distance_max': 4.5
    }],
    'CoffeeMachine': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Dishwasher': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'KitchenPot': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Microwave': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Oven': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Phone': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Refrigerator': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Remote': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Safe': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'StorageFurniture': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 4.1,
        'distance_max': 5.2
    }],
    'Table': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.8,
        'distance_max': 4.5
    }],
    'Toaster': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'TrashCan': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 4,
        'distance_max': 5.5
    }],
    'WashingMachine': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Keyboard': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3,
        'distance_max': 3.5
    }],
    'Laptop': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Door': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Printer': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Suitcase': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Bucket': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Toilet': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }]
}

BACKGROUND_RGB = np.array([216, 206, 189], dtype=np.uint8)
