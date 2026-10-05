"""Evolution train/val curve plotter (two-panel, paper style).

Top panel:    combined score — train (dotted) + baseline hline + val-by-N.
Bottom panel: component metrics — train P(human)/coverage (dashed) + baseline
              P(human) hline + avg-val P(human)/coverage (solid, min/max band).

Domain-agnostic: reads only the standard curve rows. ``plot_scores`` keeps the
signature ``ppol.evolution.fitness._refresh_plots`` calls, so the live training
plot uses this same style.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _val_iter(r: Dict[str, Any]) -> int:
    return int(r.get("iteration", 0))


def _build_iter_to_attempt(train_rows: List[Dict[str, Any]]) -> Dict[int, int]:
    """Map each successful iteration to a 1-based contiguous index (drops gaps)."""
    iters = sorted({int(r["iteration"]) for r in train_rows})
    return {it: i + 1 for i, it in enumerate(iters)}


def _set_iteration_ticks(ax, *, iterations, map_x, fontsize=12, label_every=5) -> None:
    if not iterations:
        return
    max_iter = max(int(i) for i in iterations)
    desired = [1] + list(range(label_every, max_iter + 1, label_every)) if label_every > 1 \
        else sorted({int(i) for i in iterations})
    ticks, labels, seen = [], [], set()
    for it in desired:
        x = float(map_x(int(it)))
        if x in seen:
            continue
        seen.add(x)
        ticks.append(x)
        labels.append(str(it))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fontsize)


def plot_scores_train_val_nsweep(
    train_rows: List[Dict[str, Any]],
    val_rows_by_n: Dict[int, List[Dict[str, Any]]],
    output_path: "str | Path",
    *,
    baseline_hl: Optional[float] = None,
    baseline_score: Optional[float] = None,
    verbose: bool = True,
    contiguous_x: bool = True,
) -> Optional[str]:
    """Two-panel train/val evolution figure. Returns the output path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not train_rows and not any(val_rows_by_n.get(k) for k in val_rows_by_n):
        return None

    iter_to_attempt = _build_iter_to_attempt(train_rows) if contiguous_x else {}
    sorted_iters = sorted(iter_to_attempt)

    def map_x(it: int) -> float:
        if not contiguous_x:
            return float(it)
        if it in iter_to_attempt:
            return float(iter_to_attempt[it])
        if not sorted_iters:
            return float(it)
        return float(iter_to_attempt[min(sorted_iters, key=lambda x: abs(x - it))])

    t_real = [int(r["iteration"]) for r in train_rows]
    t_it = np.array([map_x(i) for i in t_real], float)
    t_final = np.array([r.get("final_combined_score", r.get("train_combined", 0.0)) for r in train_rows])
    t_hl = np.array([r.get("train_human_likeness", 0.0) for r in train_rows])
    t_cov = np.array([r.get("train_intra_diversity", np.nan) for r in train_rows])

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # --- Top: combined score ---
    if train_rows:
        ax0.plot(t_it, t_final, marker="o", linestyle=":", color="0.2",
                 linewidth=1.8, alpha=0.95, label="Train Score")
    if baseline_score is not None and np.isfinite(float(baseline_score)):
        ax0.axhline(float(baseline_score), linestyle="--", color="#6b7280",
                    linewidth=1.4, alpha=0.95, label="Baseline Val Score")
    n_styles = {10: ("#7f0000", "^"), 8: ("#cb181d", "s"), 5: ("#fb6a4a", "o")}
    for n in sorted(val_rows_by_n):
        rows = val_rows_by_n.get(n) or []
        if not rows:
            continue
        vx = np.array([map_x(_val_iter(r)) for r in rows], float)
        vc = np.array([float(r.get("val_combined_score", np.nan)) for r in rows])
        color, marker = n_styles.get(n, ("#b91c1c", "o"))
        ax0.plot(vx, vc, linestyle="-", color=color, marker=marker, markersize=6,
                 linewidth=2.0, alpha=0.9, label=f"Val Score (N={n})")
    ax0.set_ylabel("Combined Score", fontsize=18)
    ax0.set_ylim(0.0, 0.85)
    ax0.set_yticks(np.arange(0.0, 0.81, 0.1))
    ax0.tick_params(axis="both", labelsize=14)
    ax0.legend(loc="upper left", fontsize=13)
    ax0.grid(True, alpha=0.3)

    # --- Bottom: component metrics ---
    if train_rows:
        ax1.plot(t_it, t_hl, linestyle="--", color="#e66100", marker="o",
                 markersize=3.5, linewidth=1.4, alpha=0.85, label="Train P(human)")
        if np.any(np.isfinite(t_cov)):
            ax1.plot(t_it, t_cov, linestyle="--", color="#2171b5", marker="d",
                     markersize=3, linewidth=1.2, alpha=0.75, label="Train Coverage")
    if baseline_hl is not None and np.isfinite(float(baseline_hl)):
        ax1.axhline(float(baseline_hl), linestyle="--", color="#7c3aed",
                    linewidth=1.3, alpha=0.9, label="Baseline P(human)")

    def _agg(field):
        by_iter: Dict[int, List[float]] = {}
        for n in sorted(val_rows_by_n):
            for r in (val_rows_by_n.get(n) or []):
                try:
                    v = float(r.get(field, np.nan))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(v):
                    by_iter.setdefault(_val_iter(r), []).append(v)
        its = sorted(by_iter)
        if not its:
            return (np.array([]),) * 4
        x = np.array([map_x(i) for i in its], float)
        return (x,
                np.array([np.mean(by_iter[i]) for i in its]),
                np.array([np.min(by_iter[i]) for i in its]),
                np.array([np.max(by_iter[i]) for i in its]))

    xh, mh, lh, hh = _agg("val_human_likeness")
    if xh.size:
        ax1.fill_between(xh, lh, hh, color="#fde68a", alpha=0.3, linewidth=0)
        ax1.plot(xh, mh, linestyle="-", color="#b45309", marker="o", markersize=6,
                 linewidth=2.0, alpha=0.95, label="Avg. Val P(human)")
    xd, md, ld, hd = _agg("val_intra_diversity")
    if xd.size:
        ax1.fill_between(xd, ld, hd, color="#bfdbfe", alpha=0.28, linewidth=0)
        ax1.plot(xd, md, linestyle="-", color="#1d4ed8", marker="o", markersize=5,
                 linewidth=1.9, alpha=0.9, label="Avg. Val Coverage")

    ax1.set_xlabel("Iteration", fontsize=18)
    ax1.set_ylabel("Component Metrics", fontsize=18)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_yticks(np.arange(0.0, 0.81, 0.1))
    ax1.tick_params(axis="both", labelsize=14)
    ax1.legend(loc="upper left", fontsize=13, ncol=2)
    ax1.grid(True, alpha=0.3)

    tick_iters = list(t_real)
    for rows in val_rows_by_n.values():
        tick_iters.extend(_val_iter(r) for r in (rows or []))
    _set_iteration_ticks(ax1, iterations=tick_iters, map_x=map_x)

    plt.tight_layout()
    out = str(output_path)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"Saved evolution plot → {out}")
    return out


def plot_scores(train_rows, val_rows, out_png, *, title=None, verbose=False,
                contiguous_x=True, baseline_hl=None, baseline_score=None):
    """Fitness-compatible wrapper: group val rows by n_personas → two-panel figure."""
    by_n: Dict[int, List[Dict[str, Any]]] = {}
    for r in (val_rows or []):
        by_n.setdefault(int(r.get("n_personas", 10)), []).append(r)
    return plot_scores_train_val_nsweep(
        train_rows or [], by_n, out_png,
        baseline_hl=baseline_hl, baseline_score=baseline_score,
        verbose=verbose, contiguous_x=contiguous_x,
    )
