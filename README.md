<h1 align="center">
Prior-Free Vision-Based Pose Measurement for Non-Cooperative Spacecraft via Manifold-Induced Higher-Order Correspondence Modeling
</h1>

<p align="center">
  Chuan Yan, Hongfeng Long, Yuebo Ma, Rujin Zhao, and Zhenming Peng
</p>


Abstract：Vision-based pose measurement for non-cooperative spacecraft (NCS) is essential for close-range operations, but repetitive and symmetric structures in spacecraft imagery often lead to ambiguous cross-view correspondences and unstable SfM-based pose recovery. To address this problem, this paper proposes a prior-free pose measurement framework based on manifold-induced higher-order correspondence modeling. The proposed method embeds $L_2$-normalized descriptors onto a unit hypersphere and jointly models first-order descriptor similarity, second-order local geometric consistency, and third-order structural consistency in the induced association domain. The resulting higher-order affinity model is optimized by reweighted random walks hypergraph matching (RRWHM) and discretized into one-to-one correspondences for SfM-based pose recovery. Experiments on synthetic NCS sequences, the real SS1A sequence, and public Middlebury sequences show that the proposed method generally achieves lower pose errors, better trajectory continuity, and more stable local pose recovery than the compared matching strategies.Frame-wise error trends further suggest that the proposed method helps reduce pose drift and improves local pose consistency in SfM-based pose measurement.
## Motivation
<img width="452" height="396" alt="fig00" src="https://github.com/user-attachments/assets/d841ea60-1e61-4dec-a892-53037575e457" />



## Run the demo
python super_colmap_rrwhm.py   --projpath DESD   --images_path rgb   --cameraModel PINHOLE   --single_camera   --matcher rrwhm_mfps   --m1 120 --m2 120   --topk 120 --kappa 15   --max_candidates 1200   --max_pairs 60000   --max_triplets 30000   --rrwhm_iter 80 --rrwhm_c 0.2 --rrwhm_beta 12   --lambda1 0.08 --lambda2 0.35 --lambda3 0.57   --prefilter --pre_top 800 --min_refine 40 --prefilter_ratio 0.95   --topM 700



## Qualitative Results
<img width="1518" height="1491" alt="fig822" src="https://github.com/user-attachments/assets/b9348460-bc14-4b42-a003-3ff273d45076" />

