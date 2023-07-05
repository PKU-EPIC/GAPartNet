from typing import Optional

import numpy as np


def hsv_to_rgb(
    h: np.ndarray, s: Optional[np.ndarray] = None, v: Optional[np.ndarray] = None
) -> np.ndarray:
    if s is None:
        s = np.ones_like(h)
    if v is None:
        v = np.ones_like(h)

    print("h", h.shape, s.shape, v.shape)

    i = (h * 6.0).astype(np.int32)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    res = np.zeros((h.shape[0], 3), dtype=np.float32)
    mask_0 = i == 0
    res[mask_0] = np.stack([v[mask_0], t[mask_0], p[mask_0]], axis=1)
    mask_1 = i == 1
    res[mask_1] = np.stack([q[mask_1], v[mask_1], p[mask_1]], axis=1)
    mask_2 = i == 2
    res[mask_2] = np.stack([p[mask_2], v[mask_2], t[mask_2]], axis=1)
    mask_3 = i == 3
    res[mask_3] = np.stack([p[mask_3], q[mask_3], v[mask_3]], axis=1)
    mask_4 = i == 4
    res[mask_4] = np.stack([t[mask_4], p[mask_4], v[mask_4]], axis=1)
    mask_5 = i == 5
    res[mask_5] = np.stack([v[mask_5], p[mask_5], q[mask_5]], axis=1)
    return res
