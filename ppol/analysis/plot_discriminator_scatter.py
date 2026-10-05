"""PCA scatter of behavioral-fingerprint clouds (humans vs personas/baseline).

``plot_fingerprint_pca`` is the general renderer: it standardizes, fits a 2D PCA
on a chosen basis, projects every cloud, and draws a black-X centroid per group.
``plot_human_vs_baseline_scatter`` is the discriminator-time (human vs baseline)
wrapper used by the pipeline; the colbench benchmark builds its own multi-condition
cloud list and calls ``plot_fingerprint_pca`` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ppol.fingerprinting import BehavioralFingerprint, REGEX_FEATURES


def _fp_matrix(fps: List[BehavioralFingerprint], feature_names: Sequence[str]) -> np.ndarray:
    if not fps:
        return np.zeros((0, len(feature_names)))
    return np.array(
        [[fp.features.get(n, 0.0) for n in feature_names] for fp in fps],
        dtype=np.float64,
    )


def plot_fingerprint_pca(
    clouds: List[Dict[str, Any]],
    out: Path | str,
    *,
    title: str,
    basis_labels: Optional[List[str]] = None,
    feature_names: Sequence[str] = REGEX_FEATURES,
    seed: int = 0,
    verbose: bool = True,
) -> Optional[Path]:
    """2D PCA scatter of arbitrary labeled fingerprint clouds.

    Each cloud is a dict ``{label, fps, color, marker, alpha?, size?, zorder?,
    legend? (str|None), centroid_group? (str)}``. Standardization mean/std and the
    PCA basis are fit on the clouds named in ``basis_labels`` (all clouds pooled
    when ``None``). Clouds sharing a ``centroid_group`` get one merged black-X
    centroid, coloured by the group's top-``zorder`` (darkest) member.
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        if verbose:
            print("sklearn not available; skipping scatter plot")
        return None

    clouds = [c for c in clouds if c.get("fps")]
    if len(clouds) < 2:
        if verbose:
            print("Too few fingerprint clouds; skipping scatter plot")
        return None

    mats = {c["label"]: _fp_matrix(c["fps"], feature_names) for c in clouds}
    basis = basis_labels or [c["label"] for c in clouds]
    X_ref = np.vstack([mats[l] for l in basis if l in mats])
    mu, sigma = X_ref.mean(axis=0), X_ref.std(axis=0)
    sigma[sigma == 0] = 1.0
    pca = PCA(n_components=2, random_state=seed).fit((X_ref - mu) / sigma)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    proj: Dict[str, np.ndarray] = {}
    for c in clouds:
        P = pca.transform((mats[c["label"]] - mu) / sigma)
        proj[c["label"]] = P
        ax.scatter(P[:, 0], P[:, 1], s=c.get("size", 28), marker=c.get("marker", "o"),
                   color=c["color"], alpha=c.get("alpha", 0.5), edgecolors="none",
                   label=c.get("legend", c["label"]) or "_nolegend_",
                   zorder=c.get("zorder", 3))

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for c in clouds:
        groups.setdefault(c.get("centroid_group", c["label"]), []).append(c)
    for members in groups.values():
        C = np.vstack([proj[c["label"]] for c in members]).mean(axis=0)
        color = max(members, key=lambda c: c.get("zorder", 3))["color"]
        ax.scatter(C[0], C[1], marker="X", s=200, color=color,
                   edgecolor="black", linewidth=1.0, zorder=10)

    var = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({var[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% var)")
    ax.set_title(title)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", framealpha=0.9, markerscale=1.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"Saved scatter → {out_path}")
    return out_path


def plot_human_vs_baseline_scatter(
    *,
    human_fps: List[BehavioralFingerprint],
    baseline_fps: List[BehavioralFingerprint],
    out: Path | str,
    title: str = "Human vs baseline fingerprints (PCA)",
    human_test: Optional[List[BehavioralFingerprint]] = None,
    baseline_test: Optional[List[BehavioralFingerprint]] = None,
    seed: int = 0,
    verbose: bool = True,
) -> Optional[Path]:
    """Discriminator-time human-vs-baseline scatter (PCA fit on the pooled clouds).

    When ``human_test`` / ``baseline_test`` are provided, train points are drawn
    lighter and test points darker (τ² train/test protocol), with one centroid per
    Human / Baseline group.
    """
    if human_test or baseline_test:
        clouds = [
            {"label": "Human (train)", "fps": human_fps, "marker": "^", "color": "#bff0a8",
             "alpha": 0.75, "size": 30, "zorder": 5, "legend": None, "centroid_group": "Human"},
            {"label": "Human (test)", "fps": human_test or [], "marker": "^", "color": "#52c41a",
             "alpha": 0.95, "size": 34, "zorder": 6, "legend": "Human", "centroid_group": "Human"},
            {"label": "Baseline (train)", "fps": baseline_fps, "marker": "o", "color": "#ee9999",
             "alpha": 0.78, "size": 28, "zorder": 2, "legend": None, "centroid_group": "Baseline"},
            {"label": "Baseline (test)", "fps": baseline_test or [], "marker": "o", "color": "#c62828",
             "alpha": 0.95, "size": 32, "zorder": 3, "legend": "Baseline", "centroid_group": "Baseline"},
        ]
    else:
        clouds = [
            {"label": "Human", "fps": human_fps, "marker": "^", "color": "#52c41a",
             "alpha": 0.85, "size": 32, "zorder": 5},
            {"label": "Baseline", "fps": baseline_fps, "marker": "o", "color": "#c62828",
             "alpha": 0.85, "size": 28, "zorder": 3},
        ]
    return plot_fingerprint_pca(clouds, out, title=title, basis_labels=None,
                                seed=seed, verbose=verbose)
