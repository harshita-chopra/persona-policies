"""
Plot fitness and component metrics vs OpenEvolve iteration.

Reads:
  - ``training/simulations/iter_NNNN/log.json`` — per-trial TRAIN metrics (preferred).
  - ``training/results/train_curve.jsonl`` — fallback if simulations logs are missing.
  - ``training/results/val_curve.jsonl`` — full-val on the current elite when it changes.

Val x-axis uses ``iteration`` (the OpenEvolve trial that **produced** the elite;
read from ``best_program_info.json`` when the row was written).

``refresh_evolution_plots`` uses the same train source as the CLI (simulations
first) so incremental PNG updates match a manual replot.

Usage::

  python -m persona_policies.analysis.plot_evolution_scores
  python -m persona_policies.analysis.plot_evolution_scores --output /path/to/plot.png
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from persona_policies.config import PersonaPoliciesConfig, domain_list
from persona_policies.discriminator import BehavioralDiscriminator
from persona_policies.fingerprinting import BehavioralFingerprint, BehavioralFingerprintExtractor
from persona_policies.tau_human_loader import load_domain_dialogues
from persona_policies.tau_train_context import split_train_val, task_id_from_tau_human_instance_key


_ITER_DIR = re.compile(r"^iter_(\d+)$")


def _metrics_phase(log: Dict[str, Any], phase: str) -> Dict[str, Any]:
    block = log.get(phase)
    if not isinstance(block, dict):
        return {}
    m = block.get("metrics")
    return m if isinstance(m, dict) else {}


def collect_train_rows(simulations_dir: Path) -> List[Dict[str, Any]]:
    """One row per ``iter_NNNN/`` with train-phase metrics."""
    rows: List[Dict[str, Any]] = []
    if not simulations_dir.is_dir():
        return rows
    for child in sorted(simulations_dir.iterdir()):
        if not child.is_dir():
            continue
        m = _ITER_DIR.match(child.name)
        if not m:
            continue
        log_path = child / "log.json"
        if not log_path.is_file():
            continue
        try:
            log = json.loads(log_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        it = int(log.get("iteration", int(m.group(1))))
        tm = _metrics_phase(log, "train")
        rows.append(
            {
                "iteration": it,
                "outcome": str(log.get("outcome", "")),
                "final_combined_score": float(log.get("final_combined_score", float("nan"))),
                "train_combined": float(tm.get("combined_score", float("nan"))),
                "train_human_likeness": float(tm.get("human_likeness", float("nan"))),
                "train_intra_diversity": (
                    float(tm["intra_set_diversity"])
                    if tm.get("intra_set_diversity") is not None
                    else float("nan")
                ),
            }
        )
    rows.sort(key=lambda r: r["iteration"])
    return rows


def _json_dicts_from_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Parse JSONL; each line may contain one dict or several glued together (partial write)."""
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    dec = json.JSONDecoder()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        i, n = 0, len(line)
        while i < n:
            while i < n and line[i].isspace():
                i += 1
            if i >= n:
                break
            try:
                obj, end = dec.raw_decode(line, i)
            except json.JSONDecodeError:
                break
            if isinstance(obj, dict):
                rows.append(obj)
            i = end
    return rows


def _val_plot_iteration(r: Dict[str, Any]) -> int:
    """X-axis index: ``iteration`` is the trial that produced the elite
    (iter 0 = initial program)."""
    return int(r.get("iteration", 0))


def collect_val_rows(val_curve_path: Path) -> List[Dict[str, Any]]:
    """One row per new-best validation from ``val_curve.jsonl``."""
    rows = _json_dicts_from_jsonl(val_curve_path)
    rows.sort(key=lambda r: (int(r.get("epoch", 0)), _val_plot_iteration(r)))
    return rows


def collect_val_rows_by_n(
    val_curve_path: Path,
    val_sweep_path: Path,
    *,
    include_n: tuple[int, ...] = (5, 8, 10),
) -> Dict[int, List[Dict[str, Any]]]:
    """Merge official val + sweep rows and return rows grouped by ``n_personas``.

    Official ``val_curve.jsonl`` may omit ``n_personas`` in older rows; default those
    to 10 (the evolution-time val setting in this project).
    """
    by_key: Dict[tuple[int, int], Dict[str, Any]] = {}
    # Prefer official val rows for n=10 at duplicate (iteration, n).
    for r in _json_dicts_from_jsonl(val_sweep_path):
        it = int(r.get("iteration", -1))
        n = int(r.get("n_personas", 10))
        if it >= 0:
            by_key[(it, n)] = r
    for r in _json_dicts_from_jsonl(val_curve_path):
        it = int(r.get("iteration", -1))
        n = int(r.get("n_personas", 10))
        if it >= 0:
            by_key[(it, n)] = r
    out: Dict[int, List[Dict[str, Any]]] = {int(n): [] for n in include_n}
    for (it, n), r in by_key.items():
        if n in out:
            out[n].append(r)
    for n in out:
        out[n].sort(key=lambda r: _val_plot_iteration(r))
    return out


def _baseline_hl_and_score(config: PersonaPoliciesConfig) -> Optional[Tuple[float, float]]:
    """Return (baseline_hl, baseline_score_like) for current config domain.

    Uses the *same* evolution train/val split inside official train tasks:
      - train discriminator on train-task human + train-task baseline sim
      - score baseline sim on val-task episodes only

    This avoids leaking evolution-val tasks into discriminator training.
    Tau sim baseline has no generated persona set, so there is no comparable
    behavioral coverage term. The baseline score line is therefore the val-split
    baseline HL, using the same held-out val tasks as evolution.
    """
    fp_path = Path(config.baseline_fingerprints_path)
    if not fp_path.is_absolute():
        fp_path = _REPO / fp_path
    if not fp_path.is_file():
        return None
    try:
        train_ids, val_ids = split_train_val(
            seed=config.seed,
            val_fraction=config.val_fraction,
            domain=config.taubench_domain,
            taubench_root=Path(config.taubench_root),
        )
        train_set = {str(x) for x in train_ids}
        val_set = {str(x) for x in val_ids}
        if not train_set or not val_set:
            return None

        domains = domain_list(config.taubench_domain)
        qualify = len(domains) > 1

        def task_key(row_domain: str | None, task_id: str | None) -> str:
            tid = str(task_id or "")
            if qualify:
                return f"{row_domain}:{tid}" if row_domain else tid
            return tid

        # Baseline fingerprints from reference artifacts.
        raw = json.loads(fp_path.read_text(encoding="utf-8"))
        sim_train: List[BehavioralFingerprint] = []
        sim_val: List[BehavioralFingerprint] = []
        for row in raw:
            feats = row.get("fingerprint") if isinstance(row, dict) and "fingerprint" in row else row
            if not isinstance(feats, dict):
                continue
            tid = (
                task_key(row.get("domain"), row.get("task_id"))
                if isinstance(row, dict)
                else ""
            )
            fp = BehavioralFingerprint(features=dict(feats))
            if tid in train_set:
                sim_train.append(fp)
            elif tid in val_set:
                sim_val.append(fp)
        if not sim_train or not sim_val:
            return None

        # Human fingerprints, split by task-id derived from tau human instance key.
        human_path = Path(config.tau_bench_human_path)
        if not human_path.is_absolute():
            human_path = _REPO / human_path
        if not human_path.is_file():
            return None
        extractor = BehavioralFingerprintExtractor()
        human_train: List[BehavioralFingerprint] = []
        for dom in domains:
            for d in load_domain_dialogues(str(human_path), dom):
                trace = d.get("conversation") or d.get("turns") or []
                if trace:
                    tid = task_id_from_tau_human_instance_key(
                        str(d.get("instance_id", "")),
                        dom,
                    )
                    if task_key(dom, tid) in train_set:
                        human_train.append(extractor.compute_fingerprint(trace))
        if not human_train:
            return None

        # Train fresh in current env (no pickle/version issues), then score val baseline.
        disc = BehavioralDiscriminator()
        disc.train(human_train, sim_train, verbose=False)
        hl = float(np.mean([disc.predict_human_probability(fp) for fp in sim_val]))
        score = hl
        return hl, score
    except Exception:
        return None


def load_train_rows_for_plot(config: PersonaPoliciesConfig) -> List[Dict[str, Any]]:
    """Train series for plotting: same source as ``main()`` (simulations logs first).

    ``train_curve.jsonl`` only appends successful fitness steps; using it alone
    can shift the train line vs ``iter_*/log.json`` and make val look misaligned.
    """
    sim_dir = Path(config.simulations_dir)
    if not sim_dir.is_absolute():
        sim_dir = _REPO / sim_dir
    rows = collect_train_rows(sim_dir)
    if rows:
        return rows
    from persona_policies.evolution import metrics_plot as _metrics_plot

    tc = Path(config.training_results_dir) / "train_curve.jsonl"
    if not tc.is_absolute():
        tc = _REPO / tc
    return _metrics_plot._load_train_curve_jsonl(tc)


def _build_iter_to_attempt(train_rows: List[Dict[str, Any]]) -> Dict[int, int]:
    """Map each successful OpenEvolve iter to its 1-based attempt index.

    Attempts are ordered by iteration so the x-axis stays monotonic; failed
    iterations (those without an ``iter_NNNN/log.json``) are squeezed out.
    """
    iters = sorted({int(r["iteration"]) for r in train_rows})
    return {it: i + 1 for i, it in enumerate(iters)}


def _set_iteration_ticks(
    ax: plt.Axes,
    *,
    iterations: List[int],
    map_x,
    fontsize: int = 12,
    label_every: int = 5,
) -> None:
    """Show readable real-iteration labels on the x-axis."""
    x_to_iters: Dict[float, List[int]] = {}
    for it in sorted({int(i) for i in iterations}):
        x_to_iters.setdefault(float(map_x(int(it))), []).append(int(it))
    if not x_to_iters:
        return
    max_iter = max(int(i) for i in iterations)
    if label_every > 1:
        desired_iters = [1] + list(range(label_every, max_iter + 1, label_every))
    else:
        desired_iters = sorted({int(i) for i in iterations})

    ticks: List[float] = []
    labels: List[str] = []
    seen_ticks: set[float] = set()
    for it in desired_iters:
        x = float(map_x(int(it)))
        if x in seen_ticks:
            continue
        seen_ticks.add(x)
        ticks.append(x)
        labels.append(str(it))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fontsize)


def plot_scores(
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    output_path: Path,
    title: str | None = None,
    *,
    verbose: bool = True,
    contiguous_x: bool = True,
) -> None:
    """Render the evolution scores plot.

    ``contiguous_x`` (default: True) reindexes the x-axis to ``1..N`` over
    successful attempts. Val points whose producing iter isn't in the train set
    snap to the nearest successful attempt. Annotations always show the real
    OpenEvolve iter.
    """
    if not train_rows and not val_rows:
        raise SystemExit(
            f"Nothing to plot: no iter_*/log.json and no val_curve.jsonl found."
        )

    # --- x-axis mapping ---
    iter_to_attempt = _build_iter_to_attempt(train_rows) if contiguous_x else {}
    sorted_train_iters = sorted(iter_to_attempt) if contiguous_x else []

    def map_x(it: int) -> float:
        if not contiguous_x:
            return float(it)
        if it in iter_to_attempt:
            return float(iter_to_attempt[it])
        if not sorted_train_iters:
            return float(it)
        # Val rows can reference iters with no simulations log (e.g. resumed
        # runs); snap to the nearest successful attempt so the point is still
        # placed sensibly on the contiguous axis.
        nearest = min(sorted_train_iters, key=lambda x: abs(x - it))
        return float(iter_to_attempt[nearest])

    # --- train arrays ---
    t_real = np.array([int(r["iteration"]) for r in train_rows], dtype=int) if train_rows else np.array([], dtype=int)
    t_it = np.array([map_x(int(r["iteration"])) for r in train_rows], dtype=float) if train_rows else np.array([])
    t_final = np.array([r["final_combined_score"] for r in train_rows]) if train_rows else np.array([])
    t_comb = np.array([r["train_combined"] for r in train_rows]) if train_rows else np.array([])
    t_hl = np.array([r["train_human_likeness"] for r in train_rows]) if train_rows else np.array([])
    t_id = np.array([r["train_intra_diversity"] for r in train_rows]) if train_rows else np.array([])

    # --- val arrays (one point per new best; x = elite_iteration / producing trial) ---
    v_real = np.array([_val_plot_iteration(r) for r in val_rows], dtype=int)
    v_it = np.array([map_x(int(x)) for x in v_real], dtype=float)
    v_comb = np.array([float(r.get("val_combined_score", float("nan"))) for r in val_rows])
    v_hl = np.array([float(r.get("val_human_likeness", float("nan"))) for r in val_rows])
    v_id = np.array([float(r.get("val_intra_diversity", float("nan"))) for r in val_rows])

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    ax0 = axes[0]
    if train_rows:
        ax0.plot(t_it, t_final, "o-", color="0.2", linewidth=1.8, alpha=0.9,
                 label="train combined (fitness = selection signal)")
        if np.any(np.isfinite(t_comb) & (t_comb != t_final)):
            ax0.plot(t_it, t_comb, "s--", color="C0", alpha=0.5, label="train combined (raw)")
    if val_rows:
        ax0.plot(v_it, v_comb, "^-", color="C3", linewidth=2.2, markersize=9,
                 label="full-val combined on BEST (monitor)")
        for x, real in zip(v_it, v_real):
            ax0.axvline(x, color="C3", alpha=0.12, linestyle=":")
            ax0.annotate(
                f"val elite it{int(real)}",
                xy=(x, 0.02),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="bottom",
                fontsize=8,
                color="C3",
                alpha=0.8,
            )
    ax0.set_ylabel("Combined Score", fontsize=18)
    ax0.set_ylim(0.0, 0.85)
    ax0.set_yticks(np.arange(0.0, 0.81, 0.1))
    ax0.tick_params(axis="both", labelsize=14)
    ax0.legend(loc="upper left", fontsize=13)
    ax0.grid(True, alpha=0.3)
    ax0.set_title(title or "Evolution: scores vs iteration")

    # Second panel: same metric colors (orange = P(human), blue = coverage);
    # val = solid, train = dashed.
    _COLOR_HL = "#e66100"  # orange
    _COLOR_DV = "#2171b5"  # blue

    ax1 = axes[1]
    if train_rows:
        ax1.plot(
            t_it,
            t_hl,
            linestyle="--",
            color=_COLOR_HL,
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            alpha=0.9,
            label="Train P(human)",
        )
        if np.any(np.isfinite(t_id)):
            ax1.plot(
                t_it,
                t_id,
                linestyle="--",
                color=_COLOR_DV,
                marker="d",
                markersize=3,
                linewidth=1.2,
                alpha=0.8,
                label="Train Coverage",
            )
    if val_rows:
        ax1.plot(
            v_it,
            v_hl,
            linestyle="-",
            color=_COLOR_HL,
            marker="^",
            markersize=7,
            linewidth=2.0,
            alpha=0.95,
            label="Val P(human) (best)",
        )
        if np.any(np.isfinite(v_id)):
            ax1.plot(
                v_it,
                v_id,
                linestyle="-",
                color=_COLOR_DV,
                marker="s",
                markersize=6,
                linewidth=2.0,
                alpha=0.9,
                label="Val Coverage (best)",
            )
    ax1.set_xlabel("Iteration", fontsize=18)
    ax1.set_ylabel("Component Metrics", fontsize=18)
    ax1.set_ylim(0.0, 0.85)
    ax1.set_yticks(np.arange(0.0, 0.81, 0.1))
    ax1.tick_params(axis="both", labelsize=14)
    ax1.legend(loc="upper left", fontsize=13, ncol=2)
    ax1.grid(True, alpha=0.3)

    _set_iteration_ticks(
        ax1,
        iterations=[int(x) for x in t_real] + [int(x) for x in v_real],
        map_x=map_x,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    if not verbose:
        return
    print(f"Saved plot → {output_path}")
    if val_rows:
        if contiguous_x:
            print(
                f"  val_curve rows: {len(val_rows)} "
                f"(real iters {int(v_real.min())}..{int(v_real.max())})"
            )
        else:
            print(
                f"  val_curve rows: {len(val_rows)} "
                f"(val at iters {int(v_it.min())}..{int(v_it.max())})"
            )
    if train_rows:
        if contiguous_x:
            print(
                f"  train rows:    {len(train_rows)} "
                f"(real iters {int(t_real.min())}..{int(t_real.max())}, "
                f"attempts 1..{len(train_rows)})"
            )
        else:
            print(
                f"  train rows:    {len(train_rows)} "
                f"(iters {int(t_it.min())}..{int(t_it.max())})"
            )


def plot_scores_train_val_nsweep(
    train_rows: List[Dict[str, Any]],
    val_rows_by_n: Dict[int, List[Dict[str, Any]]],
    output_path: Path,
    title: str | None = None,
    *,
    baseline_hl: Optional[float] = None,
    baseline_score: Optional[float] = None,
    verbose: bool = True,
    contiguous_x: bool = True,
    layout: str = "vertical",
) -> None:
    """Plot train curve plus val monitor points for n_personas in {5, 8, 10}."""
    if not train_rows and not any(val_rows_by_n.get(k) for k in val_rows_by_n):
        raise SystemExit("Nothing to plot for train/val N-sweep.")
    if layout not in {"vertical", "side_by_side"}:
        raise ValueError("layout must be 'vertical' or 'side_by_side'")

    iter_to_attempt = _build_iter_to_attempt(train_rows) if contiguous_x else {}
    sorted_train_iters = sorted(iter_to_attempt) if contiguous_x else []

    def map_x(it: int) -> float:
        if not contiguous_x:
            return float(it)
        if it in iter_to_attempt:
            return float(iter_to_attempt[it])
        if not sorted_train_iters:
            return float(it)
        nearest = min(sorted_train_iters, key=lambda x: abs(x - it))
        return float(iter_to_attempt[nearest])

    t_real = np.array([int(r["iteration"]) for r in train_rows], dtype=int) if train_rows else np.array([], dtype=int)
    t_it = np.array([map_x(int(r["iteration"])) for r in train_rows], dtype=float) if train_rows else np.array([])
    t_final = np.array([r["final_combined_score"] for r in train_rows]) if train_rows else np.array([])
    t_hl = np.array([r["train_human_likeness"] for r in train_rows]) if train_rows else np.array([])
    t_id = np.array([r["train_intra_diversity"] for r in train_rows]) if train_rows else np.array([])

    if layout == "side_by_side":
        fig, axes = plt.subplots(1, 2, figsize=(18, 6.2), sharey=True)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    ax0, ax1 = axes

    # Top: train combined dotted (requested), plus val combined by N.
    if train_rows:
        ax0.plot(
            t_it,
            t_final,
            marker="o",
            linestyle=":",
            color="0.2",
            linewidth=1.8,
            alpha=0.95,
            label="Train Score",
        )
    if baseline_score is not None and np.isfinite(float(baseline_score)):
        ax0.axhline(
            float(baseline_score),
            linestyle="--",
            color="#6b7280",
            linewidth=1.4,
            alpha=0.95,
            label="Baseline Val Score",
        )

    # Aggregate val across n={5,8,10} for readability:
    # one mean line + light min/max band at each producing iteration.
    def _agg(field: str):
        by_iter: Dict[int, List[float]] = {}
        for n in sorted(val_rows_by_n):
            for r in (val_rows_by_n.get(n) or []):
                it = int(_val_plot_iteration(r))
                try:
                    v = float(r.get(field, float("nan")))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(v):
                    by_iter.setdefault(it, []).append(v)
        its = sorted(by_iter.keys())
        if not its:
            return np.array([]), np.array([]), np.array([]), np.array([])
        x = np.array([map_x(it) for it in its], dtype=float)
        mu = np.array([float(np.mean(by_iter[it])) for it in its], dtype=float)
        lo = np.array([float(np.min(by_iter[it])) for it in its], dtype=float)
        hi = np.array([float(np.max(by_iter[it])) for it in its], dtype=float)
        return x, mu, lo, hi

    # Keep top panel as separate n-curves (requested).
    n_styles = {
        10: {"color": "#7f0000", "marker": "^"},
        8: {"color": "#cb181d", "marker": "s"},
        5: {"color": "#fb6a4a", "marker": "o"},
    }
    for n in sorted(val_rows_by_n):
        rows = val_rows_by_n.get(n) or []
        if not rows:
            continue
        v_real = np.array([_val_plot_iteration(r) for r in rows], dtype=int)
        v_it = np.array([map_x(int(x)) for x in v_real], dtype=float)
        v_comb = np.array([float(r.get("val_combined_score", float("nan"))) for r in rows])
        st = n_styles.get(n, {"color": "#b91c1c", "marker": "o"})
        ax0.plot(
            v_it,
            v_comb,
            linestyle="-",
            color=st["color"],
            marker=st["marker"],
            markersize=6,
            linewidth=2.0,
            alpha=0.9,
            label=f"Val Score (N={n})",
        )
    ax0.set_ylabel("Combined Score", fontsize=18)
    if layout == "side_by_side":
        ax0.set_xlabel("Iteration", fontsize=18)
    ax0.set_ylim(0.0, 0.85)
    ax0.set_yticks(np.arange(0.0, 0.81, 0.1))
    ax0.tick_params(axis="both", labelsize=14)
    ax0.legend(loc="upper left", fontsize=13, ncol=(2 if layout == "side_by_side" else 1))
    ax0.grid(True, alpha=0.3)
    # Keep title empty for a cleaner figure.

    # Bottom: train components + val components by N.
    _COLOR_HL = "#e66100"
    _COLOR_DV = "#2171b5"
    if train_rows:
        ax1.plot(
            t_it, t_hl, linestyle="--", color=_COLOR_HL, marker="o",
            markersize=3.5, linewidth=1.4, alpha=0.85, label="Train P(human)",
        )
        if np.any(np.isfinite(t_id)):
            ax1.plot(
                t_it, t_id, linestyle="--", color=_COLOR_DV, marker="d",
                markersize=3, linewidth=1.2, alpha=0.75, label="Train Coverage",
            )
    if baseline_hl is not None and np.isfinite(float(baseline_hl)):
        ax1.axhline(
            float(baseline_hl),
            linestyle="--",
            color="#7c3aed",
            linewidth=1.3,
            alpha=0.9,
            label="Baseline P(human)",
        )
    x_h, m_h, l_h, h_h = _agg("val_human_likeness")
    if x_h.size:
        ax1.fill_between(
            x_h, l_h, h_h, color="#fde68a", alpha=0.3, linewidth=0,
            label="_nolegend_",
        )
        ax1.plot(
            x_h,
            m_h,
            linestyle="-",
            color="#b45309",
            marker="o",
            markersize=6,
            linewidth=2.0,
            alpha=0.95,
            label="Avg. Val P(human)",
        )
    x_d, m_d, l_d, h_d = _agg("val_intra_diversity")
    if x_d.size:
        ax1.fill_between(
            x_d, l_d, h_d, color="#bfdbfe", alpha=0.28, linewidth=0,
            label="_nolegend_",
        )
        ax1.plot(
            x_d,
            m_d,
            linestyle="-",
            color="#1d4ed8",
            marker="o",
            markersize=5,
            linewidth=1.9,
            alpha=0.9,
            label="Avg. Val Coverage",
        )

    ax1.set_xlabel("Iteration", fontsize=18)
    ax1.set_ylabel("Component Metrics", fontsize=18)
    ax1.set_ylim(0.0, 0.85)
    ax1.set_yticks(np.arange(0.0, 0.81, 0.1))
    ax1.tick_params(axis="both", labelsize=14)
    ax1.legend(loc="upper left", fontsize=13, ncol=2)
    ax1.grid(True, alpha=0.3)

    tick_iters: List[int] = [int(x) for x in t_real]
    for rows in val_rows_by_n.values():
        tick_iters.extend(int(_val_plot_iteration(r)) for r in (rows or []))
    if layout == "side_by_side":
        _set_iteration_ticks(
            ax0,
            iterations=tick_iters,
            map_x=map_x,
        )
    _set_iteration_ticks(
        ax1,
        iterations=tick_iters,
        map_x=map_x,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"Saved plot → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot evolution scores from simulations logs + val curve")
    parser.add_argument(
        "--simulations-dir",
        type=str,
        default=None,
        help="Override path to …/training/simulations (default: config)",
    )
    parser.add_argument(
        "--val-curve",
        type=str,
        default=None,
        help="Override path to val_curve.jsonl (default: <training_results_dir>/val_curve.jsonl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output PNG path (default: training/results/evolution_scores.png)",
    )
    parser.add_argument(
        "--raw-iter",
        action="store_true",
        help="Use real OpenEvolve iteration numbers on the x-axis (default is "
             "contiguous 1..N over successful attempts).",
    )
    args = parser.parse_args()

    cfg = PersonaPoliciesConfig()
    cfg.ensure_output_dirs()

    if args.simulations_dir:
        cfg.simulations_dir = args.simulations_dir

    val_curve = (
        Path(args.val_curve)
        if args.val_curve
        else Path(cfg.training_results_dir) / "val_curve.jsonl"
    )
    if not val_curve.is_absolute():
        val_curve = _REPO / val_curve

    out = (
        Path(args.output)
        if args.output
        else Path(cfg.training_results_dir) / "evolution_scores.png"
    )
    if not out.is_absolute():
        out = _REPO / out

    train_rows = load_train_rows_for_plot(cfg)
    val_rows = collect_val_rows(val_curve)
    plot_scores(
        train_rows,
        val_rows,
        out,
        title="Evolution: scores vs iteration",
        contiguous_x=not args.raw_iter,
    )
    # Additional plot: train + val monitor points across n={5,8,10} using
    # val_curve (official n=10) + val_n_personas_sweep.jsonl (n=5/8/10 sweeps).
    val_sweep = Path(cfg.training_results_dir) / "val_n_personas_sweep.jsonl"
    if not val_sweep.is_absolute():
        val_sweep = _REPO / val_sweep
    out2 = out.with_name("train_val_evolution_scores.png")
    val_by_n = collect_val_rows_by_n(val_curve, val_sweep, include_n=(5, 8, 10))
    if any(val_by_n.get(k) for k in (5, 8, 10)):
        b_ref = _baseline_hl_and_score(cfg)
        plot_scores_train_val_nsweep(
            train_rows,
            val_by_n,
            out2,
            title="Evolution: train + val by n_personas (5/8/10)",
            baseline_hl=(b_ref[0] if b_ref is not None else None),
            baseline_score=(b_ref[1] if b_ref is not None else None),
            contiguous_x=not args.raw_iter,
        )


if __name__ == "__main__":
    main()
