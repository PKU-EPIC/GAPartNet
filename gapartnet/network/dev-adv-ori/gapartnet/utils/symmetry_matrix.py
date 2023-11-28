import math
from typing import Tuple

import torch

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
