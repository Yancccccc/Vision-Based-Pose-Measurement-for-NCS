# matchers.py
import numpy as np
import torch
import cv2


def l2_normalize_desc(D: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize descriptors row-wise."""
    if D is None or D.size == 0:
        return D
    n = np.linalg.norm(D, axis=1, keepdims=True)
    return D / (n + eps)


# -----------------------------
# Basic matchers (GPU torch)
# -----------------------------
def mutual_nn_matcher(descriptors1, descriptors2, device="cuda"):
    """
    Mutual nearest neighbors matcher for L2 normalized descriptors.
    Returns matches: (M,2) [idx1, idx2]
    """
    des1 = torch.from_numpy(descriptors1).to(device)
    des2 = torch.from_numpy(descriptors2).to(device)
    sim = des1 @ des2.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = ids1 == nn21[nn12]
    matches = torch.stack([ids1[mask], nn12[mask]]).t()
    return matches.data.cpu().numpy()


def ratio_matcher(descriptors1, descriptors2, ratio=0.8, device="cuda"):
    """
    Symmetric Lowe's ratio test matcher for L2 normalized descriptors.
    Returns matches: (M,2)
    """
    des1 = torch.from_numpy(descriptors1).to(device)
    des2 = torch.from_numpy(descriptors2).to(device)
    sim = des1 @ des2.t()

    # top2 1->2
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nns_dist = torch.sqrt(torch.clamp(2 - 2 * nns_sim, min=1e-8))
    ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    nn12 = nns[:, 0]

    # top2 2->1
    nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
    nns_dist = torch.sqrt(torch.clamp(2 - 2 * nns_sim, min=1e-8))
    ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    nn21 = nns[:, 0]

    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ratios12 <= ratio) & (ratios21[nn12] <= ratio)
    matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)
    return matches.data.cpu().numpy()


def mutual_nn_ratio_matcher(descriptors1, descriptors2, ratio=0.8, device="cuda"):
    """
    Mutual NN + symmetric Lowe ratio. Returns matches: (M,2)
    """
    des1 = torch.from_numpy(descriptors1).to(device)
    des2 = torch.from_numpy(descriptors2).to(device)
    sim = des1 @ des2.t()

    # top2 1->2
    nns_sim12, nns12 = torch.topk(sim, 2, dim=1)
    nns_dist12 = torch.sqrt(torch.clamp(2 - 2 * nns_sim12, min=1e-8))
    ratios12 = nns_dist12[:, 0] / (nns_dist12[:, 1] + 1e-8)
    nn12 = nns12[:, 0]

    # top2 2->1
    nns_sim21, nns21 = torch.topk(sim.t(), 2, dim=1)
    nns_dist21 = torch.sqrt(torch.clamp(2 - 2 * nns_sim21, min=1e-8))
    ratios21 = nns_dist21[:, 0] / (nns_dist21[:, 1] + 1e-8)
    nn21 = nns21[:, 0]

    ids1 = torch.arange(0, sim.shape[0], device=device)

    mask = (ids1 == nn21[nn12]) & (ratios12 <= ratio) & (ratios21[nn12] <= ratio)
    matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)
    return matches.data.cpu().numpy()


def mutual_nn_ratio_matcher_with_scores(descriptors1, descriptors2, ratio=0.8, device="cuda"):
    """
    Same as mutual_nn_ratio_matcher, but also returns a score per match.
    score = best cosine similarity (if desc L2-normalized).
    Returns:
      matches: (M,2) int
      scores:  (M,) float
    """
    des1 = torch.from_numpy(descriptors1).to(device)
    des2 = torch.from_numpy(descriptors2).to(device)
    sim = des1 @ des2.t()

    nns_sim12, nns12 = torch.topk(sim, 2, dim=1)
    nns_dist12 = torch.sqrt(torch.clamp(2 - 2 * nns_sim12, min=1e-8))
    ratios12 = nns_dist12[:, 0] / (nns_dist12[:, 1] + 1e-8)
    nn12 = nns12[:, 0]
    best_sim = nns_sim12[:, 0]

    nns_sim21, nns21 = torch.topk(sim.t(), 2, dim=1)
    nns_dist21 = torch.sqrt(torch.clamp(2 - 2 * nns_sim21, min=1e-8))
    ratios21 = nns_dist21[:, 0] / (nns_dist21[:, 1] + 1e-8)
    nn21 = nns21[:, 0]

    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12]) & (ratios12 <= ratio) & (ratios21[nn12] <= ratio)

    i_idx = ids1[mask]
    j_idx = nn12[mask]
    scores = best_sim[mask]

    matches = torch.stack([i_idx, j_idx], dim=1).detach().cpu().numpy().astype(np.int32)
    scores = scores.detach().cpu().numpy().astype(np.float32)
    return matches, scores


# -----------------------------
# Geometry: Sampson + RANSAC
# -----------------------------
def _to_homo(pts_xy: np.ndarray) -> np.ndarray:
    ones = np.ones((pts_xy.shape[0], 1), dtype=np.float64)
    return np.concatenate([pts_xy.astype(np.float64), ones], axis=1)


def sampson_distance(F: np.ndarray, pts1_xy: np.ndarray, pts2_xy: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Sampson distance for fundamental matrix.
    pts1_xy, pts2_xy: (N,2) float
    returns: (N,) float
    """
    x1 = _to_homo(pts1_xy)  # (N,3)
    x2 = _to_homo(pts2_xy)

    Fx1 = (F @ x1.T).T       # (N,3)
    Ftx2 = (F.T @ x2.T).T    # (N,3)
    x2tFx1 = np.sum(x2 * Fx1, axis=1)

    denom = Fx1[:, 0]**2 + Fx1[:, 1]**2 + Ftx2[:, 0]**2 + Ftx2[:, 1]**2
    return (x2tFx1**2) / (denom + eps)


def ransac_fundamental_filter(kp1_xy: np.ndarray,
                             kp2_xy: np.ndarray,
                             matches: np.ndarray,
                             ransac_thresh: float = 1.0,
                             confidence: float = 0.999,
                             max_iters: int = 10000):
    """
    Run cv2.findFundamentalMat RANSAC on matches and return:
      inlier_mask: (M,) bool
      sampson: (M,) float (computed using estimated F; large if F invalid)
    """
    M = int(matches.shape[0]) if matches is not None else 0
    if M < 8:
        return np.zeros((M,), dtype=bool), np.full((M,), np.inf, dtype=np.float32)

    pts1 = kp1_xy[matches[:, 0]].astype(np.float32)
    pts2 = kp2_xy[matches[:, 1]].astype(np.float32)

    F, mask = cv2.findFundamentalMat(
        pts1, pts2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=float(ransac_thresh),
        confidence=float(confidence),
        maxIters=int(max_iters)
    )
    if mask is None or F is None:
        return np.zeros((M,), dtype=bool), np.full((M,), np.inf, dtype=np.float32)

    inlier_mask = (mask.ravel().astype(np.uint8) > 0)
    try:
        sd = sampson_distance(F, pts1, pts2).astype(np.float32)
    except Exception:
        sd = np.full((M,), np.inf, dtype=np.float32)
    return inlier_mask, sd


def topk_by_scores(matches: np.ndarray, scores: np.ndarray, topk: int):
    if matches.shape[0] <= topk:
        return matches, scores
    idx = np.argpartition(-scores, topk - 1)[:topk]
    return matches[idx], scores[idx]