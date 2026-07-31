"""
ood.py — Out-of-Distribution detection helpers shared across all ME* apps.

Two-layer defence:
  Layer 1 – Confidence threshold   : rejects images where the model is uncertain
  Layer 2 – Feature-space distance : rejects images whose MobileNetV2 features
                                     are far from the training distribution

Usage
-----
    from ood import load_class_stats, build_feature_extractor, is_ood

    stats          = load_class_stats(model_dir / "class_stats.npz")
    feat_extractor = build_feature_extractor(model)
    ood, reason    = is_ood(feat_extractor, stats, pil_image, raw_prob)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from PIL import Image
import keras

# ── Tuneable thresholds ────────────────────────────────────────────────────────
# Layer 1 — minimum classifier confidence to accept a prediction.
#   A value of 0.70 means the model must be at least 70% confident.
#   Images near 0.5 (uncertain) are rejected.
CONFIDENCE_THRESHOLD: float = 0.70

# Layer 2 — maximum normalised Mahalanobis-like distance to nearest class centre.
#   We compute: min over classes of  mean( |feat - class_mean| / class_std )
#   Values > Z_THRESHOLD are considered out-of-distribution.
#   Calibrated broadly across multiple models; in-dist images score ~0.4–1.5, OOD ~2.5+.
#   Threshold 2.0 gives a safe margin for diverse subject matter (food, skin, animals).
Z_THRESHOLD: float = 2.0

IMAGE_SIZE = (224, 224)


# ── Class statistics helpers ───────────────────────────────────────────────────

def load_class_stats(npz_path: Path) -> dict | None:
    """Load per-class mean/std feature vectors saved by the notebook.

    Supports two npz layouts:
      • New layout (notebook Cell 28/29): keys are 'means', 'stds', 'class_names'
        means.shape == (n_classes, feat_dim),  stds.shape == (n_classes, feat_dim)
      • Legacy layout: keys are '{class}_mean', '{class}_std'

    Returns None if the file does not exist yet (graceful fallback to
    Layer 1 only until the notebook has been run).
    """
    if not Path(npz_path).exists():
        return None
    data = np.load(npz_path, allow_pickle=True)

    # ── New layout ────────────────────────────────────────────────────────────
    if "means" in data.files and "stds" in data.files and "class_names" in data.files:
        class_names = data["class_names"].tolist()
        means       = data["means"]   # (n_classes, feat_dim)
        stds        = data["stds"]    # (n_classes, feat_dim)
        return {
            name: {"mean": means[i], "std": stds[i]}
            for i, name in enumerate(class_names)
        }

    # ── Legacy layout ─────────────────────────────────────────────────────────
    stats = {}
    keys = set(k.rsplit("_", 1)[0] for k in data.files)
    for cls in keys:
        stats[cls] = {
            "mean": data[f"{cls}_mean"],
            "std":  data[f"{cls}_std"],
        }
    return stats


def build_feature_extractor(model: keras.Model) -> keras.Model | None:
    """Build a sub-model that outputs the GlobalAveragePooling2D features.

    Returns None if the layer cannot be found (safe fallback).
    """
    try:
        gap_layer = next(
            l for l in model.layers
            if "global_average_pooling" in l.name.lower()
        )
        return keras.Model(inputs=model.input, outputs=gap_layer.output)
    except StopIteration:
        return None


# ── Core OOD check ─────────────────────────────────────────────────────────────

def _preprocess(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)


def _feature_distance(
    feat_extractor: keras.Model,
    stats: dict,
    arr: np.ndarray,
) -> float:
    """Return the minimum normalised z-distance to any class centroid."""
    feat = feat_extractor.predict(arr, verbose=0)[0]   # (1280,)
    distances = []
    for cls_stats in stats.values():
        mean = cls_stats["mean"]
        std  = np.clip(cls_stats["std"], 1e-6, None)   # avoid div-by-zero
        z    = np.abs(feat - mean) / std
        distances.append(float(np.mean(z)))
    return min(distances)


def is_ood(
    feat_extractor: keras.Model | None,
    stats: dict | None,
    pil_image: Image.Image,
    raw_prob: float,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    z_threshold: float = Z_THRESHOLD,
) -> tuple[bool, str]:
    """Run both OOD layers and return (is_rejected, reason_message).

    Parameters
    ----------
    feat_extractor      Sub-model outputting GAP features (None = skip Layer 2).
    stats               Per-class mean/std dict (None = skip Layer 2).
    pil_image           The uploaded PIL image.
    raw_prob            Sigmoid output from the classifier (0–1).
    confidence_threshold Layer-1 minimum confidence to accept.
    z_threshold         Layer-2 maximum feature distance to accept.

    Returns
    -------
    (True,  reason_str)  → image should be REJECTED
    (False, "")          → image passes both checks, show prediction
    """
    confidence = raw_prob if raw_prob >= 0.5 else 1.0 - raw_prob

    # ── Layer 1: confidence threshold ─────────────────────────────────────────
    if confidence < confidence_threshold:
        pct = confidence * 100
        return True, (
            f"The model is only {pct:.0f}% confident — too uncertain to give "
            f"a reliable prediction. Please upload a clearer or more relevant image."
        )

    # ── Layer 2: feature-space distance ───────────────────────────────────────
    if feat_extractor is not None and stats is not None:
        arr  = _preprocess(pil_image)
        dist = _feature_distance(feat_extractor, stats, arr)
        if dist > z_threshold:
            return True, (
                f"This image does not appear to belong to any of the known classes "
                f"(feature distance {dist:.1f} > threshold {z_threshold:.1f}). "
                f"Please upload an image of the correct subject."
            )

    return False, ""
