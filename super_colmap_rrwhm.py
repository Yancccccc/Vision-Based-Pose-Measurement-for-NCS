try:
    from SuperPointDetectors import get_super_points_from_scenes_return
    from matchers import mutual_nn_matcher
    from matchers_rrwhm import rrwhm_mfps_spherical_matcher
    from database import COLMAPDatabase
except Exception:
    # allow running as a standalone script
    from SuperPointDetectors import get_super_points_from_scenes_return
    from matchers import mutual_nn_matcher
    from matchers_rrwhm import rrwhm_mfps_spherical_matcher
    from database import COLMAPDatabase


import cv2
import os, time
import numpy as np
import argparse
camModelDict = {'SIMPLE_PINHOLE': 0,
                'PINHOLE': 1,
                'SIMPLE_RADIAL': 2,
                'RADIAL': 3,
                'OPENCV': 4,
                'FULL_OPENCV': 5,
                'SIMPLE_RADIAL_FISHEYE': 6,
                'RADIAL_FISHEYE': 7,
                'OPENCV_FISHEYE': 8,
                'FOV': 9,
                'THIN_PRISM_FISHEYE': 10}

def get_init_cameraparams(width, height, modelId):
    # Ground-truth intrinsics from transforms_train.json (no distortion)
    fx = 1111.1113654242622
    fy = 1111.1113654242622
    cx = 399.5
    cy = 399.5

    if modelId == 0:  # SIMPLE_PINHOLE
        f = (fx + fy) / 2.0
        return np.array([f, cx, cy], dtype=np.float64)
    elif modelId == 1:  # PINHOLE
        return np.array([fx, fy, cx, cy], dtype=np.float64)
    elif modelId == 2 or modelId == 6:  # SIMPLE_RADIAL / SIMPLE_RADIAL_FISHEYE
        f = (fx + fy) / 2.0
        return np.array([f, cx, cy, 0.0], dtype=np.float64)
    elif modelId == 3 or modelId == 7:  # RADIAL / RADIAL_FISHEYE
        f = (fx + fy) / 2.0
        return np.array([f, cx, cy, 0.0, 0.0], dtype=np.float64)
    elif modelId == 4 or modelId == 8:  # OPENCV / OPENCV_FISHEYE
        return np.array([fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    elif modelId == 5:  # FULL_OPENCV
        return np.array([fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    elif modelId == 9:  # FOV (omega not distortion) - don't use if unnecessary
        omega = 0.0
        return np.array([fx, fy, cx, cy, omega], dtype=np.float64)

    # fallback
    return np.array([fx, fy, cx, cy], dtype=np.float64)


def init_cameras_database(db, images_path, cameratype, single_camera):
    print("init cameras database ......................................")
    images_name = []
    width = None
    height = None
    for name in sorted(os.listdir(images_path)):
        if 'jpg' in name or 'png' in name:
            images_name.append(name)
            if width is None:
                img = cv2.imread(os.path.join(images_path, name))
                height, width = img.shape[:2]
    cameraModel = camModelDict[cameratype]
    params = get_init_cameraparams(width, height, cameraModel)
    if single_camera:
        db.add_camera(cameraModel, width, height, params, camera_id=0)
    for i, name in enumerate(images_name):
        if single_camera:
            db.add_image(name, 0, image_id=i)
            continue
        db.add_camera(cameraModel, width, height, params, camera_id=i)
        db.add_image(name, i, image_id=i)
    return images_name

def import_feature(db, images_path, images_name):
    print("feature extraction by super points ...........................")
    sps = get_super_points_from_scenes_return(images_path)
    db.execute("DELETE FROM keypoints;")
    db.execute("DELETE FROM descriptors;")
    db.execute("DELETE FROM matches;")
    for i, name in enumerate(images_name):
        keypoints = sps[name]['keypoints']
        n_keypoints = keypoints.shape[0]
        keypoints = keypoints[:, :2]
        keypoints = np.concatenate([keypoints.astype(np.float32),
            np.ones((n_keypoints, 1)).astype(np.float32), np.zeros((n_keypoints, 1)).astype(np.float32)], axis=1)
        db.add_keypoints(i, keypoints)

    return sps

def import_feature_from_sps(db, sps, images_name):
    print("feature extraction by super points ...........................")
    db.execute("DELETE FROM keypoints;")
    db.execute("DELETE FROM descriptors;")
    db.execute("DELETE FROM matches;")
    for i, name in enumerate(images_name):
        keypoints = sps[name]['keypoints']
        n_keypoints = keypoints.shape[0]
        keypoints = keypoints[:, :2]
        keypoints = np.concatenate([keypoints.astype(np.float32),
            np.ones((n_keypoints, 1)).astype(np.float32), np.zeros((n_keypoints, 1)).astype(np.float32)], axis=1)
        db.add_keypoints(i, keypoints)


def _to_xy(kp):
    """convert keypoints to (N,2) float32"""
    if kp is None:
        return None
    kp = np.asarray(kp)
    if kp.dtype == object:
        try:
            return np.array([[p.pt[0], p.pt[1]] for p in kp], dtype=np.float32)
        except Exception:
            pass
    if kp.ndim == 2 and kp.shape[0] == 2 and kp.shape[1] != 2:
        kp = kp.T
    if kp.ndim == 2 and kp.shape[1] >= 2:
        return kp[:, :2].astype(np.float32, copy=False)
    raise ValueError(f"Unsupported keypoints shape: {kp.shape}, dtype={kp.dtype}")

def filter_matches_fundamental(kp1, kp2, matches,
                               ransac_thresh=1.0, conf=0.999,
                               min_inliers=60, min_inlier_ratio=0.20,
                               max_samples=5000):
    """Two-view F-RANSAC filtering (weak geometry gate)."""
    if matches is None:
        return np.zeros((0, 2), dtype=np.uint32)
    matches = np.asarray(matches)
    if matches.ndim != 2 or matches.shape[1] != 2 or len(matches) < 8:
        return np.zeros((0, 2), dtype=np.uint32)

    kp1_xy = _to_xy(kp1)
    kp2_xy = _to_xy(kp2)
    n1, n2 = kp1_xy.shape[0], kp2_xy.shape[0]

    i = matches[:, 0]
    j = matches[:, 1]
    valid = (i >= 0) & (i < n1) & (j >= 0) & (j < n2)
    matches = matches[valid]
    if len(matches) < 8:
        return np.zeros((0, 2), dtype=np.uint32)

    matches = np.unique(matches, axis=0)
    if len(matches) < 8:
        return np.zeros((0, 2), dtype=np.uint32)

    if len(matches) > max_samples:
        idx = np.random.choice(len(matches), max_samples, replace=False)
        matches = matches[idx]

    pts1 = kp1_xy[matches[:, 0]]
    pts2 = kp2_xy[matches[:, 1]]

    good = np.isfinite(pts1).all(axis=1) & np.isfinite(pts2).all(axis=1)
    pts1, pts2, matches = pts1[good], pts2[good], matches[good]
    if len(matches) < 8:
        return np.zeros((0, 2), dtype=np.uint32)

    F, mask = cv2.findFundamentalMat(
        pts1, pts2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=ransac_thresh,
        confidence=conf
    )
    if F is None or mask is None:
        return np.zeros((0, 2), dtype=np.uint32)

    inl = mask.ravel().astype(bool)
    inl_matches = matches[inl]
    if len(inl_matches) < min_inliers:
        return np.zeros((0, 2), dtype=np.uint32)
    if len(inl_matches) / max(len(matches), 1) < min_inlier_ratio:
        return np.zeros((0, 2), dtype=np.uint32)

    return inl_matches.astype(np.uint32)

def match_features(db, sps, images_name, match_list_path,
                   matcher_name="mutual_nn",
                   rrwhm_cfg=None,
                   use_geom_filter=False):
    """
    Sequential matching with configurable matcher.

    matcher_name:
      - mutual_nn: baseline mutual NN on L2 descriptors
      - rrwhm_mfps: RRWHM (pure python) with MFPS spherical-triplet hyperedges
    """
    print("match features by sequential match............................")
    step_range = [1, 2, 3, 5, 8, 13, 16, 24, 32]
    num_images = len(images_name)
    match_list = open(match_list_path, 'w')

    if rrwhm_cfg is None:
        rrwhm_cfg = {}

    total_matches = 0
    for step in step_range:
        for i in range(0, num_images - step):
            img1 = images_name[i]
            img2 = images_name[i + step]
            match_list.write(f"{img1} {img2}\n")

            D1 = sps[img1].get('descriptors', None)
            D2 = sps[img2].get('descriptors', None)
            if D1 is None or D2 is None:
                continue

            # SuperPoint may store (D,N)
            D1 = D1 * 1.0
            D2 = D2 * 1.0

            if matcher_name == "mutual_nn":
                matches = mutual_nn_matcher(D1, D2).astype(np.uint32)
                scores = None
            elif matcher_name == "rrwhm_mfps":
                kp1 = sps[img1].get('keypoints', None)
                kp2 = sps[img2].get('keypoints', None)
                if kp1 is None or kp2 is None:
                    continue
                matches, scores = rrwhm_mfps_spherical_matcher(D1, D2, kp1, kp2, config=rrwhm_cfg)
                matches = matches.astype(np.uint32)
            else:
                raise ValueError(f"Unknown matcher_name: {matcher_name}")

            if use_geom_filter and matches is not None and len(matches) > 0:
                kp1 = sps[img1].get('keypoints', None)
                kp2 = sps[img2].get('keypoints', None)
                if kp1 is not None and kp2 is not None:
                    # step-adaptive thresholds (weak gate)
                    if step <= 5:
                        ransac_thresh, min_inliers, min_ratio = 1.5, 60, 0.20
                    elif step <= 13:
                        ransac_thresh, min_inliers, min_ratio = 1.0, 80, 0.25
                    else:
                        ransac_thresh, min_inliers, min_ratio = 0.8, 120, 0.30
                    matches = filter_matches_fundamental(
                        kp1, kp2, matches,
                        ransac_thresh=ransac_thresh,
                        min_inliers=min_inliers,
                        min_inlier_ratio=min_ratio
                    )

            if matches is not None and len(matches) > 0:
                db.add_matches(i, i + step, matches)
                total_matches += len(matches)

            if (i % 10) == 0:
                nm = 0 if matches is None else len(matches)
                print(f"  step={step} pair {i}/{num_images - step - 1}: {nm} matches")

    match_list.close()
    print(f"total matches: {total_matches}")
    return total_matches


def operate(cmd):
    print(cmd)
    start = time.perf_counter()
    os.system(cmd)
    end = time.perf_counter()
    duration = end - start
    print("[%s] cost %f s" % (cmd, duration))

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mapper(projpath, images_path):
    database_path = os.path.join(projpath, "database.db")
    colmap_sparse_path = os.path.join(projpath, "sparse")
    makedir(colmap_sparse_path)

    # mapper = "colmap mapper --database_path %s --image_path %s --output_path %s" % (
    #     database_path, images_path, colmap_sparse_path
    # )
    mapper = (
        "colmap mapper "
        "--database_path {db} --image_path {img} --output_path {out} "
        "--Mapper.ba_refine_focal_length 0 "
        "--Mapper.ba_refine_principal_point 0 "
        "--Mapper.ba_refine_extra_params 0 "
    ).format(db=database_path, img=images_path, out=colmap_sparse_path)

    operate(mapper)

def geometric_verification(database_path, match_list_path):
    print("Running geometric verification..................................")
    cmd = "colmap matches_importer --database_path %s --match_list_path %s --match_type pairs" % (
        database_path, match_list_path
    )
    operate(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='super points colmap')
    parser.add_argument("--projpath", required=True, type=str)
    parser.add_argument("--cameraModel", type=str, required=False, default="SIMPLE_RADIAL")
    parser.add_argument("--images_path", required=False, type=str, default="rgb")
    parser.add_argument("--single_camera", action='store_true')

    # matcher
    parser.add_argument("--matcher", type=str, default="mutual_nn",
                        choices=["mutual_nn", "rrwhm_mfps"],
                        help="feature matcher to use")
    parser.add_argument("--geom_filter", action='store_true',
                        help="apply weak two-view F-RANSAC gate after matching")

    # RRWHM + MFPS spherical-triplet config
    parser.add_argument("--m1", type=int, default=300, help="MFPS samples in image1")
    parser.add_argument("--m2", type=int, default=300, help="MFPS samples in image2")
    parser.add_argument("--topk", type=int, default=50, help="candidate topk per node (vMF unary)")
    parser.add_argument("--kappa", type=float, default=20.0, help="vMF concentration for unary")
    parser.add_argument("--max_candidates", type=int, default=800, help="max association nodes")
    parser.add_argument("--knn2d", type=int, default=10, help="2D kNN for local triplets")
    parser.add_argument("--area_min", type=float, default=2.0, help="min 2D triangle area for degeneracy filter")
    parser.add_argument("--max_triplets", type=int, default=200000, help="max high-order hyperedges (triplets)")
    parser.add_argument("--max_pairs", type=int, default=300000, help="max 2nd-order hyperedges (pairs)")
    parser.add_argument("--sigma_theta2", type=float, default=0.25, help="H2: kernel width for geodesic-distance consistency")
    parser.add_argument("--sigma_scale2", type=float, default=0.35, help="H2: kernel width for 2D scale-ratio consistency")
    parser.add_argument("--sigma_ang2", type=float, default=0.35, help="H2: kernel width for 2D angle consistency")
    parser.add_argument("--sigma_e", type=float, default=0.20, help="kernel width for spherical edges")
    parser.add_argument("--sigma_a", type=float, default=0.20, help="kernel width for spherical angles")
    parser.add_argument("--sigma_A", type=float, default=0.20, help="kernel width for spherical area")
    parser.add_argument("--rrwhm_iter", type=int, default=300, help="RRWHM iterations")
    parser.add_argument("--rrwhm_c", type=float, default=0.2, help="RRWHM mixing c")
    parser.add_argument("--rrwhm_beta", type=float, default=30.0, help="RRWHM inflate beta")
    # Strategy-B (prefilter -> refine) knobs
    parser.add_argument("--prefilter", action="store_true", default=True, help="Stage-1 fast prefilter before RRWHM")
    parser.add_argument("--pre_top", type=int, default=400, help="Keep top-K candidates after prefilter")
    parser.add_argument("--min_refine", type=int, default=80, help="If candidates < min_refine, skip RRWHM and return fast matches")
    parser.add_argument("--prefilter_ratio", type=float, default=0.9, help="Lowe ratio for mutual_nn_ratio prefilter")
    parser.add_argument("--lambda1", type=float, default=0.05, help="RRWHM weight for 1st-order unary term")
    parser.add_argument("--lambda2", type=float, default=0.20, help="RRWHM weight for 2nd-order term (indH2)")
    parser.add_argument("--lambda3", type=float, default=0.75, help="RRWHM weight for 3rd-order term (indH3)")
    parser.add_argument("--topM", type=int, default=1000, help="keep top-M matches after RRWHM")
    parser.add_argument("--min_keep_score", type=float, default=0.0, help="min score keep threshold")
    parser.add_argument("--seed", type=int, default=0, help="random seed for sampling")

    args = parser.parse_args()
    database_path = os.path.join(args.projpath, "database.db")
    match_list_path = os.path.join(args.projpath, "image_pairs_to_match.txt")
    if os.path.exists(database_path):
        cmd = "rm -rf %s" % database_path
        operate(cmd)
    images_path = os.path.join(args.projpath, args.images_path)
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    images_name = init_cameras_database(db, images_path, args.cameraModel, args.single_camera)
    sps = import_feature(db, images_path, images_name)
    rrwhm_cfg = {
        "m1": args.m1, "m2": args.m2, "seed": args.seed,
        "topk": args.topk, "kappa": args.kappa,
        "max_candidates": args.max_candidates,
        "knn2d": args.knn2d, "area_min": args.area_min,
        "max_triplets": args.max_triplets,
        "max_pairs": args.max_pairs,
        "sigma_theta2": args.sigma_theta2,
        "sigma_scale2": args.sigma_scale2,
        "sigma_ang2": args.sigma_ang2,
        "sigma_e": args.sigma_e, "sigma_a": args.sigma_a, "sigma_A": args.sigma_A,
        "max_iter": args.rrwhm_iter, "c": args.rrwhm_c, "beta": args.rrwhm_beta,
        "prefilter": args.prefilter, "pre_top": args.pre_top, "min_refine": args.min_refine,
        "prefilter_ratio": args.prefilter_ratio,
        "lambda1": args.lambda1, "lambda2": args.lambda2, "lambda3": args.lambda3,
        "topM": args.topM, "min_keep_score": args.min_keep_score,
    }
    match_features(db, sps, images_name, match_list_path,
                   matcher_name=args.matcher,
                   rrwhm_cfg=rrwhm_cfg,
                   use_geom_filter=args.geom_filter)
    db.commit()
    db.close()

    geometric_verification(database_path, match_list_path)
    mapper(args.projpath, images_path)
