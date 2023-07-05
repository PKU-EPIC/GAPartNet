import numpy as np


def estimate_similarity_umeyama(source_hom: np.ndarray, target_hom: np.ndarray):
    num_points = source_hom.shape[1]

    source_centroid = np.mean(source_hom[:3, :], axis=1)
    target_centroid = np.mean(target_hom[:3, :], axis=1)

    centered_source = source_hom[:3, :] - np.tile(source_centroid, (num_points, 1)).transpose()
    centered_target = target_hom[:3, :] - np.tile(target_centroid, (num_points, 1)).transpose()

    cov = np.matmul(centered_target, np.transpose(centered_source)) / num_points

    if np.isnan(cov).any():
        raise RuntimeError("There are NANs in the input.")

    U, D, Vh = np.linalg.svd(cov, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]

    var_P = np.var(source_hom[:3, :], axis=1).sum()
    scale_factor = 1 / var_P * np.sum(D)
    scale = np.array([scale_factor, scale_factor, scale_factor])
    scale_matrix = np.diag(scale)

    rotation = np.matmul(U, Vh).T

    translation = target_hom[:3, :].mean(axis=1) - source_hom[:3, :].mean(axis=1).dot(
        scale_factor * rotation
    )

    out_transform = np.identity(4)
    out_transform[:3, :3] = scale_matrix @ rotation
    out_transform[:3, 3] = translation

    return scale, rotation, translation, out_transform


def evaluate_model(
    out_transform: np.ndarray, source_hom: np.ndarray, target_hom: np.ndarray, pass_thrsh: float
):
    diff = target_hom - np.matmul(out_transform, source_hom)
    residual_vec = np.linalg.norm(diff[:3, :], axis=0)
    residual = np.linalg.norm(residual_vec)
    inlier_idx = np.where(residual_vec < pass_thrsh)
    num_inliers = np.count_nonzero(inlier_idx)
    inlier_ratio = num_inliers / source_hom.shape[1]
    return residual, inlier_ratio, inlier_idx[0]


def get_RANSAC_inliers(
    source_hom: np.ndarray, target_hom: np.ndarray,
    max_iters: int, pass_thrsh: float, stop_thrsh: float,
):
    best_residual = 1e10
    best_inlier_ratio = 0
    best_inlier_idx = np.arange(source_hom.shape[1])

    for i in range(max_iters):
        # Pick 5 random (but corresponding) points from source and target
        rand_idx = np.random.randint(source_hom.shape[1], size=5)
        _, _, _, out_transform = estimate_similarity_umeyama(
            source_hom[:, rand_idx], target_hom[:, rand_idx]
        )

        residual, inlier_ratio, inlier_idx = evaluate_model(
            out_transform, source_hom, target_hom, pass_thrsh
        )
        if residual < best_residual:
            best_residual = residual
            best_inlier_ratio = inlier_ratio
            best_inlier_idx = inlier_idx

        if best_residual < stop_thrsh:
            break

    return best_inlier_ratio, best_inlier_idx


def estimate_similarity_transform(
    source: np.ndarray, target: np.ndarray,
    stop_thrsh: float = 0.5,
    max_iters: int = 100,
):
    if source.shape[0] == 1:
        source = np.repeat(source, 2, axis=0)
        target = np.repeat(target, 2, axis=0)

    source_hom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
    target_hom = np.transpose(np.hstack([target, np.ones([target.shape[0], 1])]))

    # Auto-parameter selection based on source-target heuristics
    source_norm = np.mean(np.linalg.norm(source, axis=1))
    target_norm = np.mean(np.linalg.norm(target, axis=1))

    ratio_st = (source_norm / target_norm)
    ratio_ts = (target_norm / source_norm)
    pass_thrsh = ratio_st if ratio_st > ratio_ts else ratio_ts

    best_inlier_ratio, best_inlier_idx = \
        get_RANSAC_inliers(
            source_hom, target_hom, max_iters=max_iters,
            pass_thrsh=pass_thrsh, stop_thrsh=stop_thrsh,
        )
    source_inliers_hom = source_hom[:, best_inlier_idx]
    target_inliers_hom = target_hom[:, best_inlier_idx]

    if best_inlier_ratio < 0.01:
        return np.asarray([None, None, None]), None, None, None, None

    scale, rotation, translation, out_transform = estimate_similarity_umeyama(
        source_inliers_hom, target_inliers_hom
    )

    return scale, rotation, translation, out_transform, best_inlier_idx


def estimate_pose_from_npcs(xyz, npcs):
    scale, rotation, translation, out_transform, best_inlier_idx = \
        estimate_similarity_transform(npcs, xyz)

    if scale[0] == None:
        return None, np.asarray([None,None,None]), None, None, None, best_inlier_idx
    try:
        rotation_inv = np.linalg.pinv(rotation)
    except:
        import pdb
        pdb.set_trace()
    trans_seg = np.dot((xyz - translation), rotation_inv) / scale[0]
    npcs_max = abs(trans_seg[best_inlier_idx]).max(0)

    bbox_raw = np.asarray([
        [-npcs_max[0], -npcs_max[1], -npcs_max[2]],
        [npcs_max[0], -npcs_max[1], -npcs_max[2]],
        [-npcs_max[0], npcs_max[1], -npcs_max[2]],
        [-npcs_max[0], -npcs_max[1], npcs_max[2]],
        [npcs_max[0], npcs_max[1], -npcs_max[2]],
        [npcs_max[0], -npcs_max[1], npcs_max[2]],
        [-npcs_max[0], npcs_max[1], npcs_max[2]],
        [npcs_max[0], npcs_max[1], npcs_max[2]],
    ])
    bbox_trans = np.dot((bbox_raw * scale[0]), rotation) + translation

    return bbox_trans, scale, rotation, translation, out_transform, best_inlier_idx
