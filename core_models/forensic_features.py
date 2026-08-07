"""
Hand-crafted forensic feature extraction for the classical-ML tie-breaker
(core_models/forensic_ml.py).

Not trying to replace the deep ensemble - complementary signal for the
specific blind spot training/diagnose_insight.py found: InsightFace fakes
in OpenRL/DeepFakeFace are face-swaps applied directly onto the SAME real
IMDB-WIKI photos used as the real baseline (identical filenames confirmed
in both zips), so most of the image genuinely is unmanipulated camera
output. Whole-image classifiers correctly read that as mostly-authentic;
the actual tell is localized to the face region and its blending boundary.

Extracts the same feature set (color moments, LBP texture histogram, noise
residual stats, edge density) from three regions per image:
  - whole image
  - tight face crop
  - a wider "context" crop around the face (captures the blend boundary
    and surrounding area a tight crop would exclude)
plus a whole-image JPEG blockiness ratio and a has_face flag.

If no face is detected, face/context features are zero-filled and
has_face=0 - the classifier learns to weight those accordingly rather than
receiving fabricated values.
"""

import numpy as np
import cv2

from utils.gradcam import detect_and_align_face

REGION_FEATURE_NAMES = [
    "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
    "noise_mean", "noise_std", "edge_density",
    "lbp_0", "lbp_1", "lbp_2", "lbp_3", "lbp_4", "lbp_5", "lbp_6", "lbp_7",
]
N_REGION_FEATURES = len(REGION_FEATURE_NAMES)  # 17

FEATURE_NAMES = (
    [f"whole_{n}" for n in REGION_FEATURE_NAMES]
    + [f"face_{n}" for n in REGION_FEATURE_NAMES]
    + [f"context_{n}" for n in REGION_FEATURE_NAMES]
    + ["jpeg_blockiness", "has_face"]
)
N_FEATURES = len(FEATURE_NAMES)  # 53


def _lbp_histogram(gray, n_bins=8):
    """Vectorized 8-neighbor, radius-1 LBP, binned into a normalized histogram."""
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return [0.0] * n_bins

    center = gray[1:-1, 1:-1].astype(np.int16)
    neighbors = [
        gray[0:-2, 0:-2], gray[0:-2, 1:-1], gray[0:-2, 2:],
        gray[1:-1, 2:], gray[2:, 2:], gray[2:, 1:-1],
        gray[2:, 0:-2], gray[1:-1, 0:-2],
    ]
    code = np.zeros_like(center, dtype=np.uint8)
    for i, n in enumerate(neighbors):
        code |= ((n.astype(np.int16) >= center).astype(np.uint8) << i)

    hist, _ = np.histogram(code, bins=n_bins, range=(0, 256))
    total = hist.sum()
    return (hist / total).tolist() if total > 0 else [0.0] * n_bins


def _region_features(pil_crop):
    """17 features for one image region: color moments, noise, edges, LBP."""
    arr = np.array(pil_crop.convert("RGB"), dtype=np.float64)
    if arr.size == 0 or arr.shape[0] < 3 or arr.shape[1] < 3:
        return [0.0] * N_REGION_FEATURES

    gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    pixels = arr.reshape(-1, 3)
    means = pixels.mean(axis=0)
    stds = pixels.std(axis=0)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_mean = float(np.mean(np.abs(laplacian)))
    noise_std = float(np.std(laplacian))

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_density = float(np.mean(np.sqrt(sobel_x**2 + sobel_y**2)))

    lbp_hist = _lbp_histogram(gray, n_bins=8)

    return list(means) + list(stds) + [noise_mean, noise_std, edge_density] + lbp_hist


def _jpeg_blockiness(gray):
    """Ratio of discontinuity at the 8x8 JPEG grid vs elsewhere - >1 suggests
    block-boundary artifacts (recompression, splicing)."""
    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.0
    gray = gray.astype(np.float64)
    col_idx = np.arange(8, w - 1, 8)
    if len(col_idx) == 0:
        return 0.0
    boundary_diff = np.mean(np.abs(gray[:, col_idx] - gray[:, col_idx - 1]))
    all_diff = np.mean(np.abs(np.diff(gray, axis=1)))
    return float(boundary_diff / (all_diff + 1e-6))


def extract_forensic_features(pil_img):
    """
    Returns a (N_FEATURES,) float32 vector: whole-image + face + context
    region features, plus JPEG blockiness and has_face.
    """
    img = pil_img.convert("RGB")
    whole_feats = _region_features(img)

    gray_whole = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    blockiness = _jpeg_blockiness(gray_whole)

    try:
        face_crop, bbox = detect_and_align_face(img, expand_ratio=0.1)
    except Exception:
        face_crop, bbox = None, None

    has_face = 1.0 if face_crop is not None else 0.0
    face_feats = _region_features(face_crop) if face_crop is not None else [0.0] * N_REGION_FEATURES

    context_feats = [0.0] * N_REGION_FEATURES
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        w, h = img.size
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1), (y2 - y1)
        scale = 1.6  # 60% wider than the face bbox - captures the blend boundary
        nx1 = max(0, int(cx - bw * scale / 2))
        ny1 = max(0, int(cy - bh * scale / 2))
        nx2 = min(w, int(cx + bw * scale / 2))
        ny2 = min(h, int(cy + bh * scale / 2))
        if nx2 > nx1 and ny2 > ny1:
            context_crop = img.crop((nx1, ny1, nx2, ny2))
            context_feats = _region_features(context_crop)

    return np.array(
        whole_feats + face_feats + context_feats + [blockiness, has_face],
        dtype=np.float32,
    )
