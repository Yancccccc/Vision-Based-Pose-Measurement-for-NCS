import torch


# Mutual nearest neighbors matcher for L2 normalized descriptors.
def mutual_nn_matcher(descriptors1, descriptors2, device="cuda"):
    des1 = torch.from_numpy(descriptors1).to(device)
    des2 = torch.from_numpy(descriptors2).to(device)
    sim = des1 @ des2.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = ids1 == nn21[nn12]
    matches = torch.stack([ids1[mask], nn12[mask]]).t()
    return matches.data.cpu().numpy()


# Symmetric Lowe's ratio test matcher for L2 normalized descriptors.
def ratio_matcher(descriptors1, descriptors2, ratio=0.8, device="cuda"):
    des1 = torch.from_numpy(descriptors1).to(device)
    des2 = torch.from_numpy(descriptors2).to(device)
    sim = des1 @ des2.t()

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn12 = nns[:, 0]

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn21 = nns[:, 0]
    
    # Symmetric ratio test.
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio)
    
    # Final matches.
    matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)

    return matches.data.cpu().numpy()


# Mutual NN + symmetric Lowe's ratio test matcher for L2 normalized descriptors.
def mutual_nn_ratio_matcher(descriptors1, descriptors2, ratio=0.8, device="cuda"):
    des1 = torch.from_numpy(descriptors1).to(device)
    des2 = torch.from_numpy(descriptors2).to(device)
    sim = des1 @ des2.t()

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN and match similarity.
    nn12 = nns[:, 0]

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn21 = nns[:, 0]
    
    # Mutual NN + symmetric ratio test.
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = torch.min(ids1 == nn21[nn12], torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio))
    
    # Final matches.
    matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)

    return matches.data.cpu().numpy()


# ===========================
# RRWHM (pure Python) + MFPS spherical-triplet hyperedges
# No C++/CUDA extension required.
# ===========================
import numpy as np
try:
    from sklearn.neighbors import NearestNeighbors
except Exception as _e:
    NearestNeighbors = None


def _ensure_desc_nd(desc: np.ndarray) -> np.ndarray:
    """
    SuperPoint descriptors may be (D,N) or (N,D).
    Return (N,D) float32.

    Robust handling for small-N cases where N < D (e.g., 256x128).
    Heuristic:
      - If both dims look like typical descriptor dims (64/128/256/512), treat the larger as D.
      - If only the first dim looks like a descriptor dim, assume (D,N) and transpose.
    """
    desc = np.asarray(desc)
    if desc.ndim != 2:
        raise ValueError(f"Descriptors must be 2D, got {desc.shape}")

    dset = (64, 128, 256, 512)

    a, b = desc.shape
    if a in dset and b in dset:
        # ambiguous; for SuperPoint in practice D is larger (often 256) while N can be smaller (90-200)
        if a > b:
            desc = desc.T
    elif a in dset and b not in dset:
        # likely (D,N)
        desc = desc.T
    # else: assume already (N,D)

    return desc.astype(np.float32, copy=False)


def _l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def _triangle_area_2d(p1, p2, p3) -> float:
    return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]))


def _sph_theta(u: np.ndarray, v: np.ndarray, eps: float = 1e-6) -> float:
    c = float(np.dot(u, v))
    c = np.clip(c, -1.0 + eps, 1.0 - eps)
    return float(np.arccos(c))


def _spherical_triangle_features(u1, u2, u3):
    # edges
    a = _sph_theta(u2, u3)
    b = _sph_theta(u1, u3)
    c = _sph_theta(u1, u2)
    edges = np.sort([a, b, c]).astype(np.float32)

    def safe_acos(x):
        return np.arccos(np.clip(x, -1.0 + 1e-7, 1.0 - 1e-7))

    ca, cb, cc = np.cos(a), np.cos(b), np.cos(c)
    sa, sb, sc = np.sin(a), np.sin(b), np.sin(c)

    alpha = safe_acos((ca - cb * cc) / (sb * sc + 1e-12))
    beta  = safe_acos((cb - ca * cc) / (sa * sc + 1e-12))
    gamma = safe_acos((cc - ca * cb) / (sa * sb + 1e-12))
    angles = np.sort([alpha, beta, gamma]).astype(np.float32)

    area = float(alpha + beta + gamma - np.pi)
    area = max(area, 0.0)
    return edges, angles, area


def _triplet_affinity_spherical(x1, x2, x3, y1, y2, y3,
                               sigma_e=0.20, sigma_a=0.20, sigma_A=0.20,
                                 cand_ij_external=None, pre_scores_external=None):
    eP, aP, AP = _spherical_triangle_features(x1, x2, x3)
    eQ, aQ, AQ = _spherical_triangle_features(y1, y2, y3)
    de = float(np.abs(eP - eQ).sum())
    da = float(np.abs(aP - aQ).sum())
    dA = float(abs(AP - AQ))
    return float(np.exp(-de / (sigma_e + 1e-12)) *
                 np.exp(-da / (sigma_a + 1e-12)) *
                 np.exp(-dA / (sigma_A + 1e-12)))


def _mfps_on_sphere(Xu: np.ndarray, m: int, seed: int = 0) -> np.ndarray:
    """
    Manifold farthest point sampling on unit sphere.
    Pick points to maximize minimum geodesic distance (equiv minimize max dot).
    """
    Xu = np.asarray(Xu, np.float32)
    N = Xu.shape[0]
    m = int(min(max(m, 0), N))
    if m <= 0:
        return np.zeros((0,), np.int32)

    rng = np.random.default_rng(seed)
    mu = Xu.mean(axis=0)
    mu = mu / (np.linalg.norm(mu) + 1e-8)
    dots = Xu @ mu
    first = int(np.argmin(dots))
    selected = [first]
    best_maxdot = (Xu @ Xu[first]).copy()

    for _ in range(1, m):
        nxt = int(np.argmin(best_maxdot))
        selected.append(nxt)
        best_maxdot = np.maximum(best_maxdot, Xu @ Xu[nxt])
    return np.array(selected, dtype=np.int32)


def _knn_2d(kp_xy: np.ndarray, k: int = 10):
    """
    2D kNN indices for each keypoint (exclude self).
    Uses sklearn if available; otherwise falls back to a pure NumPy O(N^2) method.
    """
    kp = np.asarray(kp_xy, np.float32)[:, :2]
    N = kp.shape[0]
    kk = int(min(max(k, 1), max(N - 1, 1)))

    if NearestNeighbors is not None:
        nn = NearestNeighbors(n_neighbors=min(kk + 1, N)).fit(kp)
        _, idx = nn.kneighbors(kp)
        return [row[1:].astype(np.int32) for row in idx]

    # fallback: O(N^2) distance
    d2 = ((kp[:, None, :] - kp[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    idx = np.argpartition(d2, kk, axis=1)[:, :kk]
    return [row.astype(np.int32) for row in idx]


def _build_candidates_vmf(desc1_u: np.ndarray, desc2_u: np.ndarray,
                         topk: int = 50, mutual: bool = True,
                         kappa: float = 20.0,
                         max_candidates: int = 800):
    """
    Candidate association nodes W via vMF kernel: exp(kappa * <x,y>).
    Returns cand_ij (M,2), cand_score (M,), unary (N1,N2)
    """
    sim = desc1_u @ desc2_u.T
    unary = np.exp(kappa * sim).astype(np.float32)

    N1, N2 = unary.shape
    K = min(topk, N2)
    topj = np.argpartition(unary, -K, axis=1)[:, -K:]

    cand = set()
    for i in range(N1):
        for j in topj[i]:
            cand.add((i, int(j)))

    if mutual:
        K2 = min(topk, N1)
        topi = np.argpartition(unary, -K2, axis=0)[-K2:, :]  # (K2, N2)
        mutual_set = set()
        for j in range(N2):
            iset = set(int(x) for x in topi[:, j])
            for (i, jj) in cand:
                if jj == j and i in iset:
                    mutual_set.add((i, j))
        cand = mutual_set

    cand = np.array(list(cand), dtype=np.int32)
    if cand.size == 0:
        return cand.reshape(0, 2), np.zeros((0,), np.float32), unary

    scores = unary[cand[:, 0], cand[:, 1]]
    order = np.argsort(scores)[::-1]
    cand = cand[order][:max_candidates]
    scores = scores[order][:max_candidates].astype(np.float32)
    return cand, scores, unary


def build_problem_mfps_spherical(kp1_xy, desc1, kp2_xy, desc2,
                                 m1=300, m2=300, seed=0,
                                 topk=50, mutual=True, kappa=20.0, max_candidates=800,
                                 knn2d=10, area_min=2.0,
                                 max_pairs=300000,
                                 max_triplets=200000,
                                 # H2 kernel widths
                                 sigma_theta2=0.25, sigma_scale2=0.35, sigma_ang2=0.35,
                                 # H3 kernel widths
                                 sigma_e=0.20, sigma_a=0.20, sigma_A=0.20,cand_ij_external=None, pre_scores_external=None):
    """
    Build RRWHM problem dict using:
      - manifold MFPS sampling on descriptor sphere
      - high-order spherical triangle invariants (edges+angles+area)
      - weak 2D degeneracy filter
    Flatten index MUST be t = i + j*N1 (Fortran order) to match RRWHM.cpp.
    """
    kp1 = np.asarray(kp1_xy, np.float32)[:, :2]
    kp2 = np.asarray(kp2_xy, np.float32)[:, :2]
    d1 = _ensure_desc_nd(desc1)
    d2 = _ensure_desc_nd(desc2)
    d1u = _l2_normalize_np(d1)
    d2u = _l2_normalize_np(d2)

    N1, N2 = d1u.shape[0], d2u.shape[0]

    s1 = _mfps_on_sphere(d1u, m1, seed=seed)
    s2 = _mfps_on_sphere(d2u, m2, seed=seed + 1)
    S1 = set(int(x) for x in s1)
    S2 = set(int(x) for x in s2)

    if cand_ij_external is not None:
        cand_ij = np.asarray(cand_ij_external, np.int32).reshape(-1, 2)
        if pre_scores_external is not None:
            cand_score = np.asarray(pre_scores_external, np.float32).reshape(-1)
            if cand_score.shape[0] != cand_ij.shape[0]:
                cand_score = np.ones((cand_ij.shape[0],), np.float32)
        else:
            # Re-compute unary only on provided candidates (vMF on hypersphere)
            dots = np.sum(d1u[cand_ij[:, 0]] * d2u[cand_ij[:, 1]], axis=1)
            cand_score = np.exp(kappa * dots).astype(np.float32)
    else:
        cand_ij, cand_score, _ = _build_candidates_vmf(d1u, d2u, topk=topk, mutual=mutual,
                                                       kappa=kappa, max_candidates=max_candidates)

    # Keep only MFPS subset candidates (innovation: MFPS-driven sampling)
    if cand_ij.shape[0] > 0:
        keep = np.array([(int(i) in S1 and int(j) in S2) for i, j in cand_ij], dtype=bool)
        cand_ij = cand_ij[keep]
        cand_score = cand_score[keep]

    if cand_ij.shape[0] == 0:
        return {
            "nP1": int(N1), "nP2": int(N2),
            "indH1": np.zeros((0, 1), np.int32), "valH1": np.zeros((0, 1), np.float64),
            "indH2": np.zeros((0, 2), np.int32), "valH2": np.zeros((0, 1), np.float64),
            "indH3": np.zeros((0, 3), np.int32), "valH3": np.zeros((0, 1), np.float64),
        }

    flat_t = (cand_ij[:, 0].astype(np.int64) + cand_ij[:, 1].astype(np.int64) * N1).astype(np.int64)
    ij_to_t = { (int(i), int(j)) : int(t) for (i, j), t in zip(cand_ij, flat_t) }

    neigh1 = _knn_2d(kp1, k=knn2d)
    neigh2 = _knn_2d(kp2, k=knn2d)

    # H2 (pairwise hyperedges): undirected sparse edges between association nodes
    pair_w = {}  # (t_min, t_max) -> weight

    indH3 = []
    valH3 = []

    for (i, j), t0 in zip(cand_ij, flat_t):
        if len(indH3) >= max_triplets:
            break
        i = int(i); j = int(j)

        Ni = [int(x) for x in neigh1[i] if int(x) in S1]
        Nj = [int(x) for x in neigh2[j] if int(x) in S2]
        if len(Ni) < 2 or len(Nj) < 2:
            continue

        neigh_pairs = []
        for ia in Ni:
            for ja in Nj:
                key = (ia, ja)
                if key in ij_to_t:
                    neigh_pairs.append((ia, ja, ij_to_t[key]))
        if len(neigh_pairs) < 2:
            continue

        P0 = kp1[i]; Q0 = kp2[j]
        x0 = d1u[i]; y0 = d2u[j]


        # -------- H2: pairwise compatibility edges between (i,j)=t0 and (ia,ja)=t1 --------
        # Measure consistency of descriptor-manifold geodesic distance + weak 2D scale/angle
        eps2 = 1e-8
        for ia, ja, t1 in neigh_pairs:
            if t1 == t0:
                continue
            vP = kp1[ia] - P0
            vQ = kp2[ja] - Q0
            dp = float(np.linalg.norm(vP))
            dq = float(np.linalg.norm(vQ))
            if dp < 1e-6 or dq < 1e-6:
                continue
            # 2D relative angle difference (rotation-invariant-ish)
            cosang = float(np.dot(vP, vQ) / (dp * dq + eps2))
            cosang = np.clip(cosang, -1.0 + 1e-7, 1.0 - 1e-7)
            ang = float(np.arccos(cosang))  # [0, pi]
            # 2D scale ratio difference (log)
            sc = float(abs(np.log((dp + eps2) / (dq + eps2))))
            # manifold geodesic-distance consistency
            thP = _sph_theta(x0, d1u[ia])
            thQ = _sph_theta(y0, d2u[ja])
            dth = float(abs(thP - thQ))

            w2 = float(np.exp(-dth / (sigma_theta2 + eps2)) *
                      np.exp(-sc / (sigma_scale2 + eps2)) *
                      np.exp(-ang / (sigma_ang2 + eps2)))

            if w2 <= 0.0:
                continue
            a2, b2 = (int(t0), int(t1)) if t0 < t1 else (int(t1), int(t0))
            prev = pair_w.get((a2, b2))
            if prev is None or w2 > prev:
                pair_w[(a2, b2)] = w2

        L = len(neigh_pairs)
        for a in range(L):
            ia, ja, t1 = neigh_pairs[a]
            for b in range(a + 1, L):
                ib, jb, t2 = neigh_pairs[b]
                if ia == ib or ja == jb:
                    continue

                # weak 2D degeneracy filter
                if _triangle_area_2d(P0, kp1[ia], kp1[ib]) < area_min:
                    continue
                if _triangle_area_2d(Q0, kp2[ja], kp2[jb]) < area_min:
                    continue

                w = _triplet_affinity_spherical(
                    x0, d1u[ia], d1u[ib],
                    y0, d2u[ja], d2u[jb],
                    sigma_e=sigma_e, sigma_a=sigma_a, sigma_A=sigma_A
                )
                indH3.append((int(t0), int(t1), int(t2)))
                valH3.append(w)

                if len(indH3) >= max_triplets:
                    break
            if len(indH3) >= max_triplets:
                break

    indH3 = np.asarray(indH3, np.int32)
    valH3 = np.asarray(valH3, np.float64).reshape(-1, 1)

    indH1 = np.asarray(flat_t, np.int32).reshape(-1, 1)
    valH1 = np.asarray(cand_score, np.float64).reshape(-1, 1)

    
    # finalize H2 edges
    if len(pair_w) > 0:
        items = list(pair_w.items())
        if len(items) > max_pairs:
            items.sort(key=lambda kv: kv[1], reverse=True)
            items = items[:max_pairs]
        indH2 = np.asarray([k for k, _ in items], np.int32)
        valH2 = np.asarray([v for _, v in items], np.float64).reshape(-1, 1)
    else:
        indH2 = np.zeros((0, 2), np.int32)
        valH2 = np.zeros((0, 1), np.float64)

    return {
        "nP1": int(N1),
        "nP2": int(N2),
        "indH1": np.asfortranarray(indH1),
        "valH1": np.asfortranarray(valH1),
        "indH2": np.asfortranarray(indH2),
        "valH2": np.asfortranarray(valH2),
        "indH3": np.asfortranarray(indH3),
        "valH3": np.asfortranarray(valH3),
    }


# ---------- RRWHM pure python core ----------

def _normalize_sum1(x, eps=1e-12):
    s = float(x.sum())
    if s <= eps:
        x[:] = 1.0 / max(1, x.size)
    else:
        x /= s
    return x

def _inflate(x, beta=30.0, eps=1e-12):
    mx = float(x.max())
    if mx <= eps:
        return np.ones_like(x)
    amp = beta / mx
    return np.exp(amp * x)

def _sinkhorn_bistochastic(vec, N1, N2, delta_min=1e-12, eps=1e-12, max_iter=10000):
    M = vec.reshape((N1, N2), order="F").copy()
    delta = delta_min + 1.0
    it = 0
    while delta > delta_min and it < max_iter:
        it += 1
        T = M.copy()
        rs = M.sum(axis=1, keepdims=True)
        M /= (rs + eps)
        cs = M.sum(axis=0, keepdims=True)
        M /= (cs + eps)
        delta = float(np.sum((M - T) ** 2))
    return M.reshape(-1, order="F")


def rrwhm_python(problem, max_iter=300, c=0.2, beta=30.0,
                 delta_min=1e-12, stop_delta=1e-6,
                 use_unary_init=True,
                 lambda1=0.05, lambda2=0.20, lambda3=0.75):
    """
    Pure Python RRWHM core supporting 1st/2nd/3rd order terms.

    Association node index (flatten) MUST be: t = i + j*N1  (Fortran order).
    problem dict keys:
      nP1, nP2
      indH1 (Nt1,1), valH1 (Nt1,1) optional  -> unary on association nodes
      indH2 (Nt2,2), valH2 (Nt2,1) optional  -> pairwise compatibility between association nodes
      indH3 (Nt3,3), valH3 (Nt3,1) optional  -> triplet compatibility
    """
    N1 = int(problem["nP1"])
    N2 = int(problem["nP2"])
    NN = N1 * N2

    # -------- unary (H1) vector over association nodes --------
    u1 = np.zeros(NN, np.float64)
    indH1 = problem.get("indH1", None)
    valH1 = problem.get("valH1", None)
    if indH1 is not None and valH1 is not None and len(indH1) > 0:
        indH1 = np.asarray(indH1, dtype=np.int64).reshape(-1)
        v1 = np.asarray(valH1, dtype=np.float64).reshape(-1)
        good = (indH1 >= 0) & (indH1 < NN)
        u1[indH1[good]] = np.maximum(v1[good], 0.0)
        _normalize_sum1(u1)

    # -------- H2 (pairwise) --------
    indH2 = problem.get("indH2", None)
    valH2 = problem.get("valH2", None)
    has_H2 = indH2 is not None and valH2 is not None and len(indH2) > 0
    if has_H2:
        indH2 = np.asarray(indH2, dtype=np.int64)
        w2_raw = np.asarray(valH2, dtype=np.float64).reshape(-1)
        p2 = indH2[:, 0].astype(np.int64, copy=False)
        q2 = indH2[:, 1].astype(np.int64, copy=False)
        # clip invalid
        good = (p2 >= 0) & (p2 < NN) & (q2 >= 0) & (q2 < NN) & (p2 != q2)
        p2, q2, w2_raw = p2[good], q2[good], w2_raw[good]
        if len(p2) == 0:
            has_H2 = False
        else:
            # normalize like RRWHM.cpp: divide by maximum weighted degree
            deg2 = np.zeros(NN, np.float64)
            deg2 += np.bincount(p2, weights=w2_raw, minlength=NN)
            deg2 += np.bincount(q2, weights=w2_raw, minlength=NN)
            Hmax2 = float(deg2.max()) if deg2.size else 1.0
            if Hmax2 <= 0:
                Hmax2 = 1.0
            w2 = w2_raw / Hmax2

    # -------- H3 (triplet) --------
    indH3 = problem.get("indH3", None)
    valH3 = problem.get("valH3", None)
    has_H3 = indH3 is not None and valH3 is not None and len(indH3) > 0
    if has_H3:
        indH3 = np.asarray(indH3, dtype=np.int64)
        w3_raw = np.asarray(valH3, dtype=np.float64).reshape(-1)
        a3 = indH3[:, 0].astype(np.int64, copy=False)
        b3 = indH3[:, 1].astype(np.int64, copy=False)
        d3 = indH3[:, 2].astype(np.int64, copy=False)
        good = (a3 >= 0) & (a3 < NN) & (b3 >= 0) & (b3 < NN) & (d3 >= 0) & (d3 < NN)
        good &= (a3 != b3) & (a3 != d3) & (b3 != d3)
        a3, b3, d3, w3_raw = a3[good], b3[good], d3[good], w3_raw[good]
        if len(a3) == 0:
            has_H3 = False
        else:
            deg3 = np.zeros(NN, np.float64)
            deg3 += np.bincount(a3, weights=w3_raw, minlength=NN)
            deg3 += np.bincount(b3, weights=w3_raw, minlength=NN)
            deg3 += np.bincount(d3, weights=w3_raw, minlength=NN)
            Hmax3 = float(deg3.max()) if deg3.size else 1.0
            if Hmax3 <= 0:
                Hmax3 = 1.0
            w3 = w3_raw / Hmax3

    # -------- init X --------
    X = np.ones(NN, np.float64) / (NN if NN > 0 else 1)
    if use_unary_init and u1.sum() > 0:
        X = u1.copy()
    _normalize_sum1(X)

    # -------- main loop --------
    delta = delta_min + 1.0
    it = 0
    eps = 1e-12

    # re-normalize lambdas (avoid all-zero)
    sL = float(lambda1 + lambda2 + lambda3)
    if sL <= eps:
        lambda3 = 1.0
        lambda1 = lambda2 = 0.0
        sL = 1.0
    lambda1 /= sL
    lambda2 /= sL
    lambda3 /= sL

    while it < max_iter and delta > stop_delta:
        it += 1

        Xnew = np.zeros(NN, np.float64)

        # 3rd order contraction
        if has_H3 and lambda3 > 0:
            t1 = w3 * X[b3] * X[d3]
            t2 = w3 * X[a3] * X[d3]
            t3 = w3 * X[a3] * X[b3]
            Xnew += np.bincount(a3, weights=t1, minlength=NN)
            Xnew += np.bincount(b3, weights=t2, minlength=NN)
            Xnew += np.bincount(d3, weights=t3, minlength=NN)

        # 2nd order contraction
        if has_H2 and lambda2 > 0:
            Xnew += np.bincount(p2, weights=w2 * X[q2], minlength=NN)
            Xnew += np.bincount(q2, weights=w2 * X[p2], minlength=NN)

        # 1st order unary
        if u1.sum() > 0 and lambda1 > 0:
            Xnew += (u1 * (NN * lambda1))  # scaled before normalization

        _normalize_sum1(Xnew)

        # Inflate + bistochastic normalize (Sinkhorn on N1xN2)
        Y = _inflate(Xnew, beta=beta)
        Y = _sinkhorn_bistochastic(Y, N1, N2, delta_min=delta_min)
        _normalize_sum1(Y)

        Xout = c * Xnew + (1.0 - c) * Y
        _normalize_sum1(Xout)

        delta = float(np.sum((Xout - X) ** 2))
        X = Xout

    return X.reshape((N1, N2), order="F")


def _greedy_one_to_one_from_X(X: np.ndarray, top: int = 5000):
    """
    Greedy discretization on X (N1,N2): pick highest scores with one-to-one constraint.
    """
    N1, N2 = X.shape
    flat = X.reshape(-1, order="F")
    idx = np.argsort(flat)[::-1][:min(top, flat.size)]
    ii, jj = np.unravel_index(idx, (N1, N2), order="F")
    used_i, used_j = set(), set()
    matches = []
    scores = []
    for i, j in zip(ii, jj):
        i = int(i); j = int(j)
        if i in used_i or j in used_j:
            continue
        used_i.add(i); used_j.add(j)
        matches.append([i, j])
        scores.append(float(X[i, j]))
    return np.asarray(matches, np.uint32), np.asarray(scores, np.float32)


def rrwhm_mfps_spherical_matcher(descriptors1, descriptors2, keypoints1, keypoints2,
                                config=None):
    """
    Drop-in matcher:
      input: descriptors1/2 (SuperPoint), keypoints1/2
      output: matches uint32 (N,2), and scores float32 (N,)
    """
    if config is None:
        config = {}
    kp1 = np.asarray(keypoints1, np.float32)[:, :2]
    kp2 = np.asarray(keypoints2, np.float32)[:, :2]


    # ===== Stage-1 prefilter (Strategy B): fast screening with MutualNN+Ratio =====
    prefilter = bool(config.get("prefilter", True))
    pre_top = int(config.get("pre_top", 400))
    min_refine = int(config.get("min_refine", 80))
    pre_ratio = float(config.get("prefilter_ratio", 0.9))
    device = str(config.get("device", "cuda"))

    # Ensure descriptors are (N,D) and L2-normalized (on hypersphere)
    d1 = _ensure_desc_nd(descriptors1)
    d2 = _ensure_desc_nd(descriptors2)
    d1u = _l2_normalize_np(d1)
    d2u = _l2_normalize_np(d2)

    cand_ij_ext = None
    cand_score_ext = None
    if prefilter:
        try:
            cand_ij_ext = mutual_nn_ratio_matcher(d1u.astype(np.float32), d2u.astype(np.float32),
                                                  ratio=pre_ratio, device=device)
        except Exception:
            # Fallback: mutual NN only (still fast)
            cand_ij_ext = mutual_nn_matcher(d1u.astype(np.float32), d2u.astype(np.float32), device=device)

        if cand_ij_ext is None or len(cand_ij_ext) == 0:
            return np.zeros((0, 2), np.uint32), np.zeros((0,), np.float32)

        # Score candidates by cosine similarity on the hypersphere (dot product)
        sc = np.sum(d1u[cand_ij_ext[:, 0]] * d2u[cand_ij_ext[:, 1]], axis=1).astype(np.float32)
        order = np.argsort(sc)[::-1]
        if pre_top > 0:
            order = order[:min(pre_top, len(order))]
        cand_ij_ext = cand_ij_ext[order]
        sc = sc[order]
        # unary (vMF) scores for RRWHM H1
        kappa = float(config.get("kappa", 20.0))
        cand_score_ext = np.exp(kappa * sc).astype(np.float32)

        # Early-exit: too few candidates -> skip RRWHM and return fast matches
        if len(cand_ij_ext) < min_refine:
            topM = int(config.get("topM", 1000))
            keep = min(topM, len(cand_ij_ext))
            return cand_ij_ext[:keep].astype(np.uint32), sc[:keep].astype(np.float32)

    problem = build_problem_mfps_spherical(
        kp1, descriptors1, kp2, descriptors2,
        m1=int(config.get("m1", 300)),
        m2=int(config.get("m2", 300)),
        seed=int(config.get("seed", 0)),
        topk=int(config.get("topk", 50)),
        mutual=bool(config.get("mutual", True)),
        kappa=float(config.get("kappa", 20.0)),
        max_candidates=int(config.get("max_candidates", 800)),
        knn2d=int(config.get("knn2d", 10)),
        area_min=float(config.get("area_min", 2.0)),
        max_pairs=int(config.get("max_pairs", 300000)),
        max_triplets=int(config.get("max_triplets", 200000)),
        sigma_theta2=float(config.get("sigma_theta2", 0.25)),
        sigma_scale2=float(config.get("sigma_scale2", 0.35)),
        sigma_ang2=float(config.get("sigma_ang2", 0.35)),
        sigma_e=float(config.get("sigma_e", 0.20)),
        sigma_a=float(config.get("sigma_a", 0.20)),
        sigma_A=float(config.get("sigma_A", 0.20)),
            cand_ij_external=cand_ij_ext,
        pre_scores_external=cand_score_ext,
    )

    X = rrwhm_python(
        problem,
        max_iter=int(config.get("max_iter", 300)),
        c=float(config.get("c", 0.2)),
        beta=float(config.get("beta", 30.0)),
        use_unary_init=bool(config.get("use_unary_init", True)),
        lambda1=float(config.get("lambda1", 0.05)),
        lambda2=float(config.get("lambda2", 0.20)),
        lambda3=float(config.get("lambda3", 0.75)),
    )

    matches, scores = _greedy_one_to_one_from_X(X, top=int(config.get("top", 5000)))

    # optional: keep top-M
    topM = int(config.get("topM", 1000))
    if topM > 0 and matches.shape[0] > topM:
        order = np.argsort(scores)[::-1][:topM]
        matches = matches[order]
        scores = scores[order]

    min_keep = float(config.get("min_keep_score", 0.0))
    if min_keep > 0:
        keep = scores >= min_keep
        matches = matches[keep]
        scores = scores[keep]

    return matches.astype(np.uint32), scores.astype(np.float32)
