"""
Persona Policies Benchmark Runner
=================================
Evaluate an evolved ``best_program.py`` on the τ² **test** split (see
``PersonaPoliciesConfig.task_split_benchmark``). Reuses the evolution-time
``_run_batch`` so test metrics are apples-to-apples with train/val.

**Val split re-evaluation (for N ablations, same as evolution val):** use
``--val-sweep`` with ``--val-curve`` (defaults to ``<training_results>/val_curve.jsonl``).
Iterations and ``…/openevolve/`` are inferred. Any ``(iteration, n_personas)`` already
in the curve (official N) or in the sweep output file is skipped. Appends JSON lines
compatible with ``val_curve`` (extra key ``sweep: true``).

Compares the generated personas' behavioral fingerprints against:
  - humans   (``config.tau_bench_human_path``, domain-filtered)
  - baseline (default-simulator τ² rollouts, no persona, from ``reference_data/``)

Outputs under ``config.testing_dir/<stem>/``:
  log.json               test metrics + per-episode details
  trajectories.json
  fingerprints.json      flat list of per-rollout fingerprints
  generated_personas.json full generated persona metadata and injected prompts
  summary.txt            headline scores
  feature_bars.png       per-feature mean: human vs baseline vs test
  human_likeness_bars.png  mean P(human) by source
  fingerprint_scatter.png  2D PCA fit on human + baseline refs, with test projected

Usage::

  python -m persona_policies.benchmark \
      --best-program persona_policies/outputs/training_<version>/openevolve/best/best_program.py

  python -m persona_policies.benchmark --val-sweep \
      --val-curve persona_policies/outputs/training_<version>/results/val_curve.jsonl \
      --n-personas-list 5,8,10

  Iteration indices and ``…/openevolve/`` are inferred from the curve path. Rows in
  ``val_curve.jsonl`` (official val at ``config.n_personas``) and existing lines in
  the sweep output are not re-computed.

  For each (iteration, N) run, per-run artifacts (same style as in-training
  ``…/validation/iter_*/``) are written under
  ``…/validation/sweep_iter_NNNN_nM/`` (``log.json``, ``trajectories.json``,
  ``best_program.py``).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from persona_policies.config import PersonaPoliciesConfig
from persona_policies.config import canonical_domain_name, domain_list
from persona_policies.discriminator import BehavioralDiscriminator
from persona_policies.evaluator import PersonaPolicyEvaluator
from persona_policies.evolution.fitness import (
    _attach_diversity_axes_snapshot,
    _diversity_components_batch,
    _find_task_idx,
    _load_human_reference,
    _load_evolved_module,
    _phase_log_data,
    _run_batch,
    _task_id_for_log,
    _trajectory_json_episodes,
)
from persona_policies.tau_train_context import split_train_val
from persona_policies.fingerprinting import (
    BehavioralFingerprint,
    BehavioralFingerprintExtractor,
    REGEX_FEATURES,
)
from persona_policies.tau_human_loader import load_domain_dialogues
from persona_policies.tau_train_context import (
    format_user_scenario_c,
    load_split_task_ids,
    task_id_from_tau_human_instance_key,
    take_task_batch_sequential,
)


BASELINE_PROGRAMS = {
    "seed_initial_generator": _REPO_ROOT / "persona_policies/evolution/initial_generator.py",
    "direct_llm_personas": _REPO_ROOT / "persona_policies/evolution/baselines/direct_llm_personas.py",
}


# Features surfaced in the bar plot (subset of REGEX_FEATURES).
_BAR_FEATURES: List[str] = [
    "words_per_turn",
    "short_utterance_rate",
    "politeness_rate",
    "formality_rate",
    "acknowledgment_rate",
    "uncertainty_rate",
    "certainty_rate",
    "pushback_rate",
    "clarification_question_rate",
    "emotional_expression_rate",
    "identity_confusion_rate",
    "verbosity_cv",
]
_BAR_FEATURES = [f for f in _BAR_FEATURES if f in REGEX_FEATURES]


def _derive_stem(best_path: Path) -> str:
    for p in best_path.parents:
        if p.name.startswith("training_"):
            return p.name[len("training_"):] or p.name
    return best_path.parent.name


def _parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for part in str(s).replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _rel_repo(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(p)


def _read_val_curve_rows(val_curve_path: Path) -> List[Dict]:
    if not val_curve_path.is_file():
        return []
    rows: List[Dict] = []
    for line in val_curve_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _val_score(row: Dict[str, Any]) -> Optional[float]:
    raw = row.get("val_combined_score", row.get("combined_score"))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def select_best_program_by_val_n_average(
    *,
    config: PersonaPoliciesConfig,
    val_curve_path: Path,
    sweep_path: Path,
    openevolve_dir: Path,
    n_personas_list: List[int],
) -> tuple[Path, Dict[str, Any]]:
    """Pick checkpoint best_program.py with the highest mean val score over requested N values."""
    n_values = sorted({int(n) for n in n_personas_list if int(n) > 0})
    if not n_values:
        raise SystemExit("No valid N values for val-average selection.")

    rows = []
    for path in (val_curve_path, sweep_path):
        if path.is_file():
            rows.extend(_read_val_curve_rows(path))
    by_iter: Dict[int, Dict[int, float]] = {}
    for row in rows:
        if "iteration" not in row:
            continue
        score = _val_score(row)
        if score is None:
            continue
        try:
            it = int(row["iteration"])
            n = int(row.get("n_personas", config.n_personas))
        except (TypeError, ValueError):
            continue
        by_iter.setdefault(it, {})[n] = score

    candidates: List[tuple[float, int, Dict[int, float]]] = []
    for it, n_to_score in by_iter.items():
        if all(n in n_to_score for n in n_values):
            avg = float(np.mean([n_to_score[n] for n in n_values]))
            candidates.append((avg, it, n_to_score))
    if not candidates:
        raise SystemExit(
            "No iteration has val scores for all requested N values "
            f"{n_values}. Run --val-sweep --n-personas-list "
            f"{','.join(str(n) for n in n_values)} first."
        )

    avg, iteration, n_to_score = max(candidates, key=lambda x: (x[0], x[1]))
    best_path = openevolve_dir / "checkpoints" / f"checkpoint_{iteration}" / "best_program.py"
    if not best_path.is_file():
        raise SystemExit(f"Selected checkpoint is missing best_program.py: {best_path}")
    selection = {
        "selection_rule": "max_mean_val_combined_score_over_n_personas",
        "iteration": int(iteration),
        "n_personas_list": n_values,
        "mean_val_combined_score": float(avg),
        "val_combined_by_n": {str(n): float(n_to_score[n]) for n in n_values},
        "best_program_path": str(best_path),
    }
    return best_path, selection


def _iterations_from_val_curve(val_curve_path: Path) -> List[int]:
    iters = {int(r["iteration"]) for r in _read_val_curve_rows(val_curve_path) if "iteration" in r}
    return sorted(iters)


def _iterations_from_train_curve_hl(
    train_curve_path: Path,
    *,
    threshold: float,
) -> List[int]:
    """Unique iterations with train human-likeness above ``threshold``."""
    rows = _read_val_curve_rows(train_curve_path)
    iters: set[int] = set()
    for r in rows:
        if "iteration" not in r:
            continue
        hl = r.get("train_human_likeness", r.get("human_likeness"))
        try:
            hl_f = float(hl)
        except (TypeError, ValueError):
            continue
        if hl_f > float(threshold):
            iters.add(int(r["iteration"]))
    return sorted(iters)


def _refresh_score_plots(
    cfg: PersonaPoliciesConfig,
    *,
    val_curve_path: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """Regenerate evolution_scores.png and train_val_evolution_scores.png."""
    import matplotlib

    matplotlib.use("Agg")
    from persona_policies.analysis.plot_evolution_scores import (
        collect_val_rows,
        collect_val_rows_by_n,
        load_train_rows_for_plot,
        plot_scores,
        plot_scores_train_val_nsweep,
        _baseline_hl_and_score,
    )

    out = output_path or (Path(cfg.training_results_dir) / "evolution_scores.png")
    out = out if out.is_absolute() else _REPO_ROOT / out
    val_curve_path = val_curve_path if val_curve_path.is_absolute() else _REPO_ROOT / val_curve_path
    sweep_path = Path(cfg.training_results_dir) / "val_n_personas_sweep.jsonl"
    sweep_path = sweep_path if sweep_path.is_absolute() else _REPO_ROOT / sweep_path

    train_rows = load_train_rows_for_plot(cfg)
    val_rows = collect_val_rows(val_curve_path)
    plot_scores(
        train_rows,
        val_rows,
        out,
        title="Evolution: scores vs iteration",
        contiguous_x=True,
    )

    out2 = out.with_name("train_val_evolution_scores.png")
    val_by_n = collect_val_rows_by_n(val_curve_path, sweep_path, include_n=(5, 8, 10))
    if any(val_by_n.get(k) for k in (5, 8, 10)):
        b_ref = _baseline_hl_and_score(cfg)
        plot_scores_train_val_nsweep(
            train_rows,
            val_by_n,
            out2,
            title="Evolution: train + val by n_personas (5/8/10)",
            baseline_hl=(b_ref[0] if b_ref is not None else None),
            baseline_score=(b_ref[1] if b_ref is not None else None),
            contiguous_x=True,
        )
    return out2


def run_val_sweep_plus_plot(
    cfg: PersonaPoliciesConfig,
    *,
    openevolve_dir: Path,
    n_personas_list: List[int],
    val_curve_path: Path,
    train_curve_path: Optional[Path],
    train_hl_threshold: float,
    output_jsonl: Path,
    plot_output: Optional[Path] = None,
) -> Path:
    """Run val sweep for val-curve iters + high-HL train iters, then refresh plots."""
    val_iters = _iterations_from_val_curve(val_curve_path)
    if val_iters:
        print(
            f"[VAL-SWEEP+PLOT] sweep 1/2: {len(val_iters)} val_curve iterations",
            flush=True,
        )
        run_val_n_sweep(
            cfg,
            openevolve_dir,
            val_iters,
            n_personas_list,
            val_curve_path=val_curve_path,
            train_curve_path=train_curve_path,
            train_hl_threshold=train_hl_threshold,
            output_jsonl=output_jsonl,
        )
    else:
        print("[VAL-SWEEP+PLOT] no val_curve iterations found; skipping sweep 1/2", flush=True)

    print(
        f"[VAL-SWEEP+PLOT] sweep 2/2: train_human_likeness > {train_hl_threshold}",
        flush=True,
    )
    run_val_n_sweep(
        cfg,
        openevolve_dir,
        None,
        n_personas_list,
        val_curve_path=val_curve_path,
        train_curve_path=train_curve_path,
        train_hl_threshold=train_hl_threshold,
        output_jsonl=output_jsonl,
    )

    out = _refresh_score_plots(cfg, val_curve_path=val_curve_path, output_path=plot_output)
    print(f"[VAL-SWEEP+PLOT] wrote {out}", flush=True)
    return out


def _covered_iter_n(
    n_default: int,
    val_curve_path: Path,
    sweep_out_path: Path,
) -> set[tuple[int, int]]:
    """(iteration, n_personas) already present in official val curve or an existing sweep file."""
    covered: set[tuple[int, int]] = set()
    for path in (val_curve_path, sweep_out_path):
        for row in _read_val_curve_rows(path) if path.is_file() else []:
            it = int(row.get("iteration", -1))
            if it < 0:
                continue
            n = int(row.get("n_personas", n_default))
            covered.add((it, n))
    return covered


def _openevolve_dir_from_val_curve(val_curve_path: Path) -> Path:
    """``…/training_*/results/val_curve.jsonl`` → ``…/training_*/openevolve``."""
    return val_curve_path.resolve().parent.parent / "openevolve"


def _epoch_and_steps(
    config: PersonaPoliciesConfig,
    iteration: int,
    val_curve_path: Optional[Path],
) -> Tuple[int, int]:
    """Match ``val_curve.jsonl`` row for ``iteration`` if available; else derive from pool size."""
    if val_curve_path is not None and val_curve_path.is_file():
        for row in _read_val_curve_rows(val_curve_path):
            if int(row.get("iteration", -1)) == int(iteration):
                return int(row["epoch"]), int(row["steps_per_epoch"])
    train_ids, _ = split_train_val(
        config.seed,
        config.val_fraction,
        config.taubench_domain,
        Path(config.taubench_root),
    )
    n_train = len(train_ids)
    nb = max(1, int(getattr(config, "eval_batch_size", 1) or 1))
    steps = max(1, math.ceil(n_train / nb))
    epoch = (int(iteration) - 1) // steps + 1
    return epoch, steps


def _write_val_sweep_validation_artifacts(
    config: PersonaPoliciesConfig,
    module: Any,
    iteration: int,
    n_personas: int,
    best_path: Path,
    epoch: int,
    steps_per_epoch: int,
    val_metrics: Dict[str, Any],
    val_trajs: List,
    val_personas: List,
    val_tasks: List[Dict[str, Any]],
    val_ctxs: List[str],
    val_details: List,
    val_fingerprints_by_task: List,
    row: Dict[str, Any],
) -> Path:
    """Like in-training ``…/validation/iter_*/``, but one folder per (iter, N) sweep."""
    val_dir = (
        Path(config.simulations_dir).parent
        / "validation"
        / f"sweep_iter_{int(iteration):04d}_n{int(n_personas)}"
    )
    val_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(str(best_path), str(val_dir / "best_program.py"))
    except OSError:
        pass
    val_log = (
        _phase_log_data(val_tasks, val_personas, val_metrics, val_ctxs, val_details)
        if val_tasks
        else None
    )
    _attach_diversity_axes_snapshot(val_log or {}, module)
    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "iteration": int(iteration),
        "steps_per_epoch": int(steps_per_epoch),
        "timestamp": time.time(),
        "n_personas": int(n_personas),
        "sweep": True,
        "val": val_log,
        "summary": row,
    }
    (val_dir / "log.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    if val_trajs:
        val_traj_payload: Dict[str, Any] = {
            "fingerprint_vector_len": len(REGEX_FEATURES),
            "fingerprint_order_ref": "persona_policies.fingerprinting.REGEX_FEATURES",
            "val": _trajectory_json_episodes(val_trajs, val_fingerprints_by_task),
        }
        (val_dir / "trajectories.json").write_text(
            json.dumps(val_traj_payload, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
    return val_dir


def run_val_n_sweep(
    config: PersonaPoliciesConfig,
    openevolve_dir: Path,
    checkpoint_iterations: Optional[List[int]],
    n_personas_list: List[int],
    *,
    val_curve_path: Path,
    train_curve_path: Optional[Path] = None,
    train_hl_threshold: float = 0.55,
    output_jsonl: Optional[Path] = None,
) -> Path:
    """Replay evolution **val** (held-out train slice) for each checkpoint × N; append ``val_curve``-style JSONL.

    Each line matches ``val_curve.jsonl`` keys, plus ``\"sweep\": true``, for plotting with the
    same schema as in-training val (but ``n_personas`` and scores vary by ablation N).
    """
    config.ensure_output_dirs()
    val_curve_path = val_curve_path if val_curve_path.is_absolute() else _REPO_ROOT / val_curve_path
    if not val_curve_path.is_file():
        raise SystemExit(f"val curve not found: {val_curve_path}")

    if train_curve_path is not None:
        train_curve_path = train_curve_path if train_curve_path.is_absolute() else _REPO_ROOT / train_curve_path

    if checkpoint_iterations is not None:
        iters = checkpoint_iterations
        iter_source = "override"
    else:
        tc = train_curve_path or (val_curve_path.parent / "train_curve.jsonl")
        iters = _iterations_from_train_curve_hl(tc, threshold=float(train_hl_threshold))
        iter_source = f"train_curve hl>{float(train_hl_threshold):.3g}"
        if not iters:
            iters = _iterations_from_val_curve(val_curve_path)
            iter_source = "val_curve (fallback)"
    if not iters:
        raise SystemExit(f"No candidate iterations from {val_curve_path} / train curve")

    openevolve_dir = openevolve_dir.resolve()
    if not openevolve_dir.is_dir():
        raise SystemExit(f"OpenEvolve dir not found: {openevolve_dir}")

    out_path = output_jsonl or (Path(config.training_results_dir) / "val_n_personas_sweep.jsonl")
    out_path = out_path if out_path.is_absolute() else _REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ref = max(1, int(config.n_personas))
    covered = _covered_iter_n(n_ref, val_curve_path, out_path)

    _, val_ids = split_train_val(
        config.seed,
        config.val_fraction,
        config.taubench_domain,
        Path(config.taubench_root),
    )
    n_val = len(val_ids)
    if n_val == 0:
        raise SystemExit("No val task ids (split_train_val).")

    evaluator = PersonaPolicyEvaluator(config)
    evaluator.runner.use_split(config.task_split_evolution)

    val_rng = random.Random(config.seed)
    w_roll = max(1, int(getattr(config, "parallel_episode_workers", 1) or 1))

    print(
        f"\n[VAL-SWEEP] val_curve={val_curve_path}\n"
        f"            openevolve={openevolve_dir}\n"
        f"            val_tasks={n_val}  iters={iters} (source={iter_source})  "
        f"n_personas={n_personas_list}\n"
        f"            τ² rollout workers={w_roll} (config.parallel_episode_workers or "
        f"--parallel-episode-workers; matches evolution _run_batch)\n"
        f"            covered_pairs={len(covered)}  out={out_path}\n"
    )

    with out_path.open("a", encoding="utf-8") as sink:
        for it in iters:
            to_run: List[int] = []
            for n in n_personas_list:
                ni = max(1, int(n))
                if (int(it), ni) not in covered:
                    to_run.append(ni)
            if not to_run:
                print(
                    f"  [skip] iter={it}  all requested N already in {val_curve_path.name} or {out_path.name}",
                    flush=True,
                )
                continue

            ck_dir = openevolve_dir / f"checkpoints/checkpoint_{int(it)}"
            best = ck_dir / "best_program.py"
            if not best.is_file():
                print(f"  [skip] no file: {best}", flush=True)
                continue
            module = _load_evolved_module(str(best))
            if module is None:
                print(f"  [skip] import failed: {best}", flush=True)
                continue

            epoch, steps_per_epoch = _epoch_and_steps(config, int(it), val_curve_path)

            for n in to_run:
                print(
                    f"  [run] iter={it}  epoch={epoch}  n_personas={n} …",
                    flush=True,
                )
                (
                    val_metrics,
                    val_trajs,
                    val_personas,
                    val_tasks,
                    val_ctxs,
                    val_details,
                    _judge_pols,
                    val_fps_by_task,
                ) = _run_batch(
                    module,
                    val_ids,
                    config,
                    evaluator,
                    val_rng,
                    "val-best",
                    sequential_start=0,
                    batch_size=n_val,
                    n_personas=n,
                    label=f"sweep_iter{it}_n{n}",
                )
                if val_metrics.get("error"):
                    print(
                        f"  [error] iter={it} n={n}: {val_metrics.get('stderr', val_metrics)}",
                        flush=True,
                    )
                    continue

                row = {
                    "epoch": int(epoch),
                    "iteration": int(it),
                    "steps_per_epoch": int(steps_per_epoch),
                    "timestamp": time.time(),
                    "n_val_tasks": int(n_val),
                    "n_personas": int(n),
                    "best_program_path": _rel_repo(best),
                    "val_combined_score": float(val_metrics.get("combined_score", 0.0)),
                    "val_human_likeness": float(val_metrics.get("human_likeness", 0.0)),
                    "val_intra_diversity": float(val_metrics.get("intra_set_diversity", 0.0)),
                    "val_success_rate": float(val_metrics.get("persona_success_rate", 0.0)),
                    "val_n_episodes": int(val_metrics.get("n_episodes", 0)),
                    "sweep": True,
                }
                art_msg = ""
                try:
                    vdir = _write_val_sweep_validation_artifacts(
                        config,
                        module,
                        int(it),
                        int(n),
                        best,
                        int(epoch),
                        int(steps_per_epoch),
                        val_metrics,
                        val_trajs,
                        val_personas,
                        val_tasks,
                        val_ctxs,
                        val_details,
                        val_fps_by_task,
                        row,
                    )
                    art_msg = f"  validation → {vdir.name}/"
                except OSError as e:
                    art_msg = f"  (validation artifacts failed: {e})"
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                covered.add((int(it), int(n)))
                print(
                    f"  [ok]  combined={row['val_combined_score']:.4f}  "
                    f"HL={row['val_human_likeness']:.4f}  "
                    f"intra={row['val_intra_diversity']:.4f}"
                    f"{art_msg and '  ' + art_msg or ''}",
                    flush=True,
                )

    print(f"\n[VAL-SWEEP] appended rows → {out_path}")
    return out_path


def _load_human_fingerprints(
    cfg: PersonaPoliciesConfig,
    task_ids: Optional[set[str]] = None,
) -> List[BehavioralFingerprint]:
    """Per-dialogue human fingerprints, filtered by domain and optionally task id."""
    p = Path(cfg.tau_bench_human_path)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    if not p.is_file():
        return []
    extractor = BehavioralFingerprintExtractor()
    fps: List[BehavioralFingerprint] = []
    domains = domain_list(cfg.taubench_domain)
    qualify = len(domains) > 1
    for dom in domains:
        for d in load_domain_dialogues(str(p), dom):
            if task_ids is not None:
                instance_id = str(d.get("instance_id", ""))
                tid = task_id_from_tau_human_instance_key(instance_id, dom)
                key = f"{dom}:{tid}" if qualify and tid else tid
                if key not in task_ids:
                    continue
            trace = d.get("conversation") or d.get("turns") or []
            if trace:
                fps.append(extractor.compute_fingerprint(trace))
    return fps


def _existing_baseline_fingerprint_path(
    cfg: PersonaPoliciesConfig,
    *,
    prefer_legacy: bool = False,
) -> Optional[Path]:
    """Return configured baseline fingerprints, falling back to the legacy artifact name."""
    p = Path(cfg.baseline_fingerprints_path)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    stem = cfg.baseline_stem
    if prefer_legacy and "_user_" in stem:
        legacy_stem = stem.split("_user_", 1)[0]
        fallback = Path(cfg.reference_data_dir) / f"{legacy_stem}_fingerprints.json"
        if not fallback.is_absolute():
            fallback = _REPO_ROOT / fallback
        if fallback.is_file():
            return fallback
    if p.is_file():
        return p

    if "_user_" in stem:
        legacy_stem = stem.split("_user_", 1)[0]
        fallback = Path(cfg.reference_data_dir) / f"{legacy_stem}_fingerprints.json"
        if not fallback.is_absolute():
            fallback = _REPO_ROOT / fallback
        if fallback.is_file():
            return fallback
    return None


def _load_human_fingerprints_by_split(
    cfg: PersonaPoliciesConfig,
) -> tuple[List[BehavioralFingerprint], List[BehavioralFingerprint]]:
    """Human fingerprints split by official tau2 train/test for the configured domain(s)."""
    p = Path(cfg.tau_bench_human_path)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    if not p.is_file():
        return [], []
    from persona_policies.tau_train_context import official_split_for_task_id

    extractor = BehavioralFingerprintExtractor()
    train: List[BehavioralFingerprint] = []
    test: List[BehavioralFingerprint] = []
    for dom in domain_list(cfg.taubench_domain):
        for d in load_domain_dialogues(str(p), dom):
            trace = d.get("conversation") or d.get("turns") or []
            if not trace:
                continue
            tid = task_id_from_tau_human_instance_key(str(d.get("instance_id", "")), dom)
            if not tid:
                continue
            try:
                sp = official_split_for_task_id(dom, tid, Path(cfg.taubench_root))
            except FileNotFoundError:
                sp = None
            fp = extractor.compute_fingerprint(trace)
            if sp == "train":
                train.append(fp)
            elif sp == "test":
                test.append(fp)
    return train, test


def _load_baseline_fingerprints(
    cfg: PersonaPoliciesConfig,
    task_ids: Optional[set[str]] = None,
) -> List[BehavioralFingerprint]:
    """Default-simulator fingerprints from ``reference_data/baseline_*_fingerprints.json``."""
    p = _existing_baseline_fingerprint_path(cfg)
    if p is None:
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    fps: List[BehavioralFingerprint] = []
    domains = domain_list(cfg.taubench_domain)
    qualify = len(domains) > 1
    for row in raw:
        if task_ids is not None and isinstance(row, dict):
            row_task_id = row.get("task_id")
            row_domain = row.get("domain")
            key = (
                f"{row_domain}:{row_task_id}"
                if qualify and row_domain is not None and row_task_id is not None
                else str(row_task_id or "")
            )
            if row_task_id is not None and key not in task_ids:
                continue
        feats = row.get("fingerprint") if isinstance(row, dict) and "fingerprint" in row else row
        if isinstance(feats, dict):
            fps.append(BehavioralFingerprint(features=dict(feats)))
    return fps


def _load_baseline_fingerprints_by_split(
    cfg: PersonaPoliciesConfig,
) -> tuple[List[BehavioralFingerprint], List[BehavioralFingerprint]]:
    """Tau simulator baseline fingerprints split by stored official split metadata."""
    p = _existing_baseline_fingerprint_path(cfg, prefer_legacy=True)
    if p is None:
        return [], []
    raw = json.loads(p.read_text(encoding="utf-8"))
    train: List[BehavioralFingerprint] = []
    test: List[BehavioralFingerprint] = []
    for row in raw:
        feats = row.get("fingerprint") if isinstance(row, dict) and "fingerprint" in row else row
        if not isinstance(feats, dict):
            continue
        fp = BehavioralFingerprint(features=dict(feats))
        if isinstance(row, dict) and row.get("split") == "train":
            train.append(fp)
        elif isinstance(row, dict) and row.get("split") == "test":
            test.append(fp)
    return train, test


def _load_persona_validation_fingerprints(
    cfg: PersonaPoliciesConfig,
    selection: Optional[Dict[str, Any]],
) -> List[BehavioralFingerprint]:
    """Fingerprints from the selected checkpoint's full-val artifacts (official train-heldout)."""
    if not selection or "iteration" not in selection:
        return []
    try:
        it = int(selection["iteration"])
    except (TypeError, ValueError):
        return []
    p = Path(cfg.training_root) / "validation" / f"iter_{it:04d}" / "trajectories.json"
    if not p.is_absolute():
        p = _REPO_ROOT / p
    if not p.is_file():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    out: List[BehavioralFingerprint] = []
    for row in raw.get("val", []):
        vec = row.get("fingerprint") if isinstance(row, dict) else None
        if isinstance(vec, list):
            out.append(
                BehavioralFingerprint(
                    features={
                        name: float(vec[i]) if i < len(vec) else 0.0
                        for i, name in enumerate(REGEX_FEATURES)
                    }
                )
            )
        elif isinstance(vec, dict):
            out.append(BehavioralFingerprint(features=dict(vec)))
    return out


def _fp_matrix(fps: List[BehavioralFingerprint], names: List[str]) -> np.ndarray:
    if not fps:
        return np.zeros((0, len(names)))
    return np.array([[fp.features.get(n, 0.0) for n in names] for fp in fps], dtype=np.float64)


def _mean_p_human(fps: List[BehavioralFingerprint], disc: BehavioralDiscriminator) -> float:
    if not fps:
        return float("nan")
    return float(np.mean([disc.predict_human_probability(fp) for fp in fps]))


def _plot_feature_bars(
    by_source: Dict[str, np.ndarray],
    feature_names: List[str],
    out: Path,
) -> None:
    """Grouped bar: per feature, one bar per source (mean across rollouts/dialogues)."""
    sources = [s for s in ("human", "baseline", "test") if s in by_source and by_source[s].shape[0]]
    if not sources or not feature_names:
        return
    means = {s: by_source[s].mean(axis=0) for s in sources}
    x = np.arange(len(feature_names))
    width = 0.8 / len(sources)
    colors = {"human": "#2ca02c", "baseline": "#d62728", "test": "#1f77b4"}
    fig, ax = plt.subplots(figsize=(max(10, 0.9 * len(feature_names)), 5.5))
    for i, s in enumerate(sources):
        ax.bar(
            x + (i - (len(sources) - 1) / 2) * width,
            means[s],
            width,
            label=f"{s} (n={by_source[s].shape[0]})",
            color=colors.get(s, None),
            alpha=0.85,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in feature_names], fontsize=8)
    ax.set_ylabel("Mean value")
    ax.set_title("Behavioral features: test personas vs humans vs baseline simulator")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_human_likeness_bars(p_by_source: Dict[str, float], out: Path) -> None:
    labels = [s for s in ("human", "baseline", "test") if not np.isnan(p_by_source.get(s, float("nan")))]
    vals = [p_by_source[s] for s in labels]
    colors = {"human": "#2ca02c", "baseline": "#d62728", "test": "#1f77b4"}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, vals, color=[colors[l] for l in labels], alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean P(human) from discriminator")
    ax.set_title("Human-likeness by source (τ² test split)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_train_test_fingerprint_scatter(
    *,
    human_train: List[BehavioralFingerprint],
    human_test: List[BehavioralFingerprint],
    baseline_train: List[BehavioralFingerprint],
    baseline_test: List[BehavioralFingerprint],
    persona_train: List[BehavioralFingerprint],
    persona_test: List[BehavioralFingerprint],
    out: Path,
    title: str,
    seed: int = 0,
) -> None:
    """Retail-style PCA scatter: train/test refs plus persona train/test, same legend/colors."""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("sklearn not available; skipping PCA scatter")
        return

    series = [
        ("Humans Train", human_train, "^", "#bff0a8", 0.75, 30, 5),
        ("Humans Test", human_test, "^", "#52c41a", 0.95, 34, 6),
        ("Base-Simulators Train", baseline_train, "o", "#ee9999", 0.78, 28, 2),
        ("Base-Simulators Test", baseline_test, "o", "#c62828", 0.95, 32, 3),
        ("Persona Policies Train", persona_train, "o", "#c7e3f5", 0.75, 28, 1),
        ("Persona Policies Test", persona_test, "o", "#4b91cf", 0.95, 30, 1),
    ]
    matrices = [(label, _fp_matrix(fps, list(REGEX_FEATURES)), marker, color, alpha, size, zorder)
                for label, fps, marker, color, alpha, size, zorder in series if fps]
    if len(matrices) < 2:
        return

    reference_labels = {
        "Humans Train",
        "Base-Simulators Train",
    }
    reference_mats = [X for label, X, *_ in matrices if label in reference_labels]
    if len(reference_mats) < 2:
        return
    X_ref = np.vstack(reference_mats)
    mu, sigma = X_ref.mean(axis=0), X_ref.std(axis=0)
    sigma[sigma == 0] = 1.0
    pca = PCA(n_components=2, random_state=seed)
    pca.fit((X_ref - mu) / sigma)

    fig, ax = plt.subplots(figsize=(8.8, 6))
    projected_by_label: Dict[str, np.ndarray] = {}
    for label, X, marker, color, alpha, size, zorder in matrices:
        P = pca.transform((X - mu) / sigma)
        projected_by_label[label] = P
        
        if label == "Humans Test":
            disp_label = "Humans"
        elif label == "Base-Simulators Test":
            disp_label = "Base-Simulator"
        elif label == "Persona Policies Test":
            disp_label = "PPol"
        else:
            disp_label = "_nolegend_"

        ax.scatter(
            P[:, 0],
            P[:, 1],
            s=size,
            marker=marker,
            color=color,
            alpha=alpha,
            edgecolors="none",
            label=disp_label,
            zorder=zorder,
        )

    centroid_groups = [
        ("Base-Simulators", ["Base-Simulators Train", "Base-Simulators Test"], "#c62828", 7),
        ("Persona Policies", ["Persona Policies Train", "Persona Policies Test"], "#1f77b4", 7),
        ("Humans", ["Humans Train", "Humans Test"], "#52c41a", 8),
    ]
    for _name, labels, color, zorder in centroid_groups:
        parts = [projected_by_label[l] for l in labels if l in projected_by_label]
        if not parts:
            continue
        C = np.vstack(parts).mean(axis=0)
        ax.scatter(
            C[0],
            C[1],
            marker="X",
            s=230,
            color=color,
            edgecolor="black",
            linewidth=1.2,
            zorder=zorder,
        )

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({100*var[0]:.1f}% var)", fontsize=21)
    ax.set_ylabel(f"PC2 ({100*var[1]:.1f}% var)", fontsize=21)
    ax.tick_params(axis="both", labelsize=17)
    ax.legend(loc="upper left", fontsize=19, markerscale=1.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def _write_benchmark_outputs(
    *,
    config: PersonaPoliciesConfig,
    out_dir: Path,
    label: str,
    program_path: Optional[Path],
    metrics: Dict[str, Any],
    trajectories: List,
    task_dicts: List[Dict[str, Any]],
    fingerprints_by_task: List[List[BehavioralFingerprint]],
    n_personas: int,
    persona_batch: Optional[List[List[str]]] = None,
    task_contexts: Optional[List[str]] = None,
    details_batch: Optional[List[List[Dict[str, Any]]]] = None,
    selection: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write benchmark logs, plots, and summary for generator and no-persona runs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    test_fps: List[BehavioralFingerprint] = [
        fp for task_fps in fingerprints_by_task for fp in task_fps
    ]
    per_ep = metrics.get("per_episode", []) or []
    turns = [
        float(ep.get("n_turns"))
        for ep in per_ep
        if isinstance(ep, dict) and isinstance(ep.get("n_turns"), (int, float))
    ]
    if turns:
        metrics.setdefault("mean_turns", float(np.mean(turns)))
        cap = config.max_turns_per_episode
        if cap:
            metrics.setdefault("cap_hit_rate", float(np.mean([t >= cap for t in turns])))

    payload: Dict[str, Any] = {
        "benchmark_label": label,
        "domain": config.taubench_domain,
        "split": config.task_split_benchmark,
        "n_tasks": int(metrics.get("n_tasks_in_batch", len(task_dicts))),
        "n_personas_per_task": int(metrics.get("n_personas_per_task", n_personas)),
        "total_rollouts": int(metrics.get("total_rollouts", len(trajectories))),
        "metrics": {k: v for k, v in metrics.items() if k != "per_episode"},
        "per_episode": per_ep,
        "task_ids": [str(t.get("id")) for t in task_dicts],
    }
    generated_personas_path: Optional[Path] = None
    if persona_batch is not None and task_contexts is not None and details_batch is not None:
        payload["test"] = _phase_log_data(
            task_dicts,
            persona_batch,
            metrics,
            task_contexts,
            details_batch,
        )
        generated_personas = _generated_personas_export(
            label=label,
            config=config,
            program_path=program_path,
            selection=selection,
            task_dicts=task_dicts,
            persona_batch=persona_batch,
            task_contexts=task_contexts,
            details_batch=details_batch,
        )
        if generated_personas["total_personas"] > 0:
            generated_personas_path = out_dir / "generated_personas.json"
            generated_personas_path.write_text(
                json.dumps(generated_personas, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
    if program_path is not None:
        payload["program_path"] = str(program_path)
    if selection is not None:
        payload["selection"] = selection
    (out_dir / "log.json").write_text(
        json.dumps(payload, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "trajectories.json").write_text(
        json.dumps(
            {"trajectories": [list(t) for t in trajectories]},
            default=str,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (out_dir / "fingerprints.json").write_text(
        json.dumps([fp.features for fp in test_fps], indent=2),
        encoding="utf-8",
    )

    benchmark_task_ids = {
        str(t.get("_combined_id", t.get("id")))
        for t in task_dicts
        if t.get("_combined_id", t.get("id")) is not None
    }
    human_fps = _load_human_fingerprints(config, benchmark_task_ids)
    baseline_fps = _load_baseline_fingerprints(config, benchmark_task_ids)
    if not human_fps:
        print(f"  (no human fingerprints at {config.tau_bench_human_path}; feature bars will omit humans)")
    if not baseline_fps:
        print(f"  (no baseline fingerprints at {config.baseline_fingerprints_path}; feature bars will omit baseline)")

    disc = BehavioralDiscriminator.load(config.discriminator_model_path)
    p_human = {
        "human": _mean_p_human(human_fps, disc),
        "baseline": _mean_p_human(baseline_fps, disc),
        "test": _mean_p_human(test_fps, disc),
    }
    fp_by_source = {
        "human": _fp_matrix(human_fps, _BAR_FEATURES),
        "baseline": _fp_matrix(baseline_fps, _BAR_FEATURES),
        "test": _fp_matrix(test_fps, _BAR_FEATURES),
    }
    _plot_feature_bars(fp_by_source, _BAR_FEATURES, out_dir / "feature_bars.png")
    _plot_human_likeness_bars(p_human, out_dir / "human_likeness_bars.png")
    human_train_fps, human_test_fps = _load_human_fingerprints_by_split(config)
    baseline_train_fps, baseline_test_fps = _load_baseline_fingerprints_by_split(config)
    persona_train_fps = _load_persona_validation_fingerprints(config, selection)
    _plot_train_test_fingerprint_scatter(
        human_train=human_train_fps,
        human_test=human_test_fps,
        baseline_train=baseline_train_fps,
        baseline_test=baseline_test_fps,
        persona_train=persona_train_fps,
        persona_test=test_fps,
        out=out_dir / "fingerprint_scatter.png",
        title=f"{canonical_domain_name(config.taubench_domain).replace('_', ' ').title()} Behavioral Fingerprints",
    )

    headline_keys = (
        "combined_score",
        "human_likeness",
        "intra_set_diversity",
        "persona_success_rate",
        "mean_turns",
        "cap_hit_rate",
    )
    lines = [
        "=== Benchmark on tau2 test split ===",
        f"label: {label}",
        f"program: {program_path}" if program_path is not None else "program: none",
        f"domain/split: {config.taubench_domain}/{config.task_split_benchmark}",
        f"tasks: {int(metrics.get('n_tasks_in_batch', len(task_dicts)))}  "
        f"personas/task: {int(metrics.get('n_personas_per_task', n_personas))}  "
        f"rollouts: {int(metrics.get('total_rollouts', len(trajectories)))}",
        "",
        "Test metrics:",
    ]
    if selection is not None:
        lines += [
            f"selection: {selection.get('selection_rule')}",
            f"selected_iter: {selection.get('iteration')}",
            f"selection_val_avg: {float(selection.get('mean_val_combined_score', 0.0)):.4f}",
            "",
        ]
    if generated_personas_path is not None:
        lines += [
            f"generated_personas: {generated_personas_path}",
            "",
        ]
    for k in headline_keys:
        v = metrics.get(k)
        if isinstance(v, (int, float)):
            lines.append(f"  {k:24s} {float(v):.4f}")
    lines += [
        "",
        "Mean P(human) by source:",
        f"  human    {p_human['human']:.4f}    (n={len(human_fps)})",
        f"  baseline {p_human['baseline']:.4f}    (n={len(baseline_fps)})",
        f"  test     {p_human['test']:.4f}    (n={len(test_fps)})",
    ]
    summary = "\n".join(lines)
    (out_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)
    print(f"\n[BENCH] artifacts -> {out_dir}")
    return out_dir


def _generated_personas_export(
    *,
    label: str,
    config: PersonaPoliciesConfig,
    program_path: Optional[Path],
    selection: Optional[Dict[str, Any]],
    task_dicts: List[Dict[str, Any]],
    persona_batch: List[List[str]],
    task_contexts: List[str],
    details_batch: List[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Standalone persona export for offline inspection and selection.

    ``log.json`` also contains this information in newer runs, but it is buried after
    per-episode metrics. This file is intentionally focused on the generated persona
    descriptions, metadata, and exact injected instruction text.
    """
    tasks: List[Dict[str, Any]] = []
    total_personas = 0
    for ti, td in enumerate(task_dicts):
        task_context = task_contexts[ti] if ti < len(task_contexts) else ""
        personas = persona_batch[ti] if ti < len(persona_batch) else []
        details = details_batch[ti] if ti < len(details_batch) else []
        persona_rows: List[Dict[str, Any]] = []
        for pi, ptext in enumerate(personas):
            if not ptext:
                continue
            meta = dict(details[pi]) if pi < len(details) and isinstance(details[pi], dict) else {}
            meta["persona_idx"] = pi
            meta["expanded_instruction"] = str(ptext)
            persona_rows.append(meta)
        if not persona_rows:
            continue
        total_personas += len(persona_rows)
        tasks.append(
            {
                "task_id": _task_id_for_log(td, ti),
                "task_context": task_context,
                "personas": persona_rows,
            }
        )

    return {
        "benchmark_label": label,
        "domain": config.taubench_domain,
        "split": config.task_split_benchmark,
        "program_path": str(program_path) if program_path is not None else None,
        "selection": selection,
        "n_tasks_with_personas": len(tasks),
        "total_personas": total_personas,
        "tasks": tasks,
    }


def _source_testing_dirs(names: str) -> List[Path]:
    out: List[Path] = []
    for part in str(names or "").split(","):
        s = part.strip()
        if not s:
            continue
        p = Path(s)
        if not p.is_absolute():
            p = _REPO_ROOT / "persona_policies" / "outputs" / "testing" / p
        out.append(p)
    return out


def merge_testing_runs(
    config: PersonaPoliciesConfig,
    *,
    source_names: str,
    out_name: str,
) -> Path:
    """Merge existing per-domain benchmark artifacts and rescore under ``config``.

    This avoids rerunning duplicated conversations when a combined-domain benchmark can
    be assembled from already-run component domains.
    """
    config.ensure_output_dirs()
    source_dirs = _source_testing_dirs(source_names)
    if len(source_dirs) < 2:
        raise SystemExit("--merge-testing-runs requires at least two comma-separated source folders")
    missing = [str(p) for p in source_dirs if not (p / "log.json").is_file()]
    if missing:
        raise SystemExit("Missing source testing log.json:\n  " + "\n  ".join(missing))

    disc = BehavioralDiscriminator.load(config.discriminator_model_path)
    source_payloads = [json.loads((p / "log.json").read_text(encoding="utf-8")) for p in source_dirs]
    all_fps: List[BehavioralFingerprint] = []
    all_trajectories: List = []
    merged_per_episode: List[Dict[str, Any]] = []
    test_tasks: List[Dict[str, Any]] = []
    persona_batch: List[List[str]] = []
    task_contexts: List[str] = []
    details_batch: List[List[Dict[str, Any]]] = []
    fingerprints_by_task: List[List[BehavioralFingerprint]] = []
    successes: List[bool] = []
    failure_modes: List[Optional[str]] = []

    task_offset = 0
    episode_offset = 0
    for src_dir, payload in zip(source_dirs, source_payloads, strict=True):
        src_domain = str(payload.get("domain") or "")
        traj_payload = json.loads((src_dir / "trajectories.json").read_text(encoding="utf-8"))
        src_trajs = traj_payload.get("trajectories", [])
        src_fps_raw = json.loads((src_dir / "fingerprints.json").read_text(encoding="utf-8"))
        src_fps = [BehavioralFingerprint(features=dict(x)) for x in src_fps_raw]
        if len(src_trajs) != len(src_fps):
            raise SystemExit(
                f"trajectory/fingerprint mismatch in {src_dir}: "
                f"{len(src_trajs)} trajectories vs {len(src_fps)} fingerprints"
            )

        test_block = payload.get("test") if isinstance(payload.get("test"), dict) else {}
        src_tasks = test_block.get("tasks") if isinstance(test_block.get("tasks"), list) else []
        if not src_tasks:
            raise SystemExit(
                f"{src_dir}/log.json does not contain generated personas under test.tasks. "
                "Rerun that source benchmark with the current script before merging."
            )

        local_fps_by_task: Dict[int, List[BehavioralFingerprint]] = {
            i: [] for i in range(len(src_tasks))
        }
        for ep_i, ep in enumerate(payload.get("per_episode", []) or []):
            row = dict(ep)
            fp = src_fps[ep_i] if ep_i < len(src_fps) else None
            if fp is not None:
                row["p_human"] = disc.predict_human_probability(fp)
                all_fps.append(fp)
            if ep_i < len(src_trajs):
                all_trajectories.append(src_trajs[ep_i])
            old_ti = int(row.get("task_batch_idx", -1)) if row.get("task_batch_idx") is not None else -1
            if fp is not None and old_ti >= 0:
                local_fps_by_task.setdefault(old_ti, []).append(fp)
            row["episode_idx"] = episode_offset + len(merged_per_episode)
            if old_ti >= 0:
                row["task_batch_idx"] = task_offset + old_ti
            row["source_testing_dir"] = src_dir.name
            merged_per_episode.append(row)
            successes.append(bool(row.get("success")))
            failure_modes.append(row.get("failure_mode"))

        for i, task_row in enumerate(src_tasks):
            raw_tid = str(task_row.get("task_id", ""))
            combined_tid = raw_tid if ":" in raw_tid else f"{src_domain}:{raw_tid}"
            personas = task_row.get("personas") if isinstance(task_row.get("personas"), list) else []
            p_texts = [str(p.get("expanded_instruction", "")) for p in personas if isinstance(p, dict)]
            p_meta = [dict(p) for p in personas if isinstance(p, dict)]
            test_tasks.append({
                "id": raw_tid.split(":", 1)[-1],
                "_domain": src_domain,
                "_combined_id": combined_tid,
                "user_scenario": None,
            })
            persona_batch.append(p_texts)
            task_contexts.append(str(task_row.get("task_context", "")))
            details_batch.append(p_meta)
            fingerprints_by_task.append(local_fps_by_task.get(i, []))
        task_offset += len(src_tasks)
        episode_offset += len(src_trajs)

    if not all_fps:
        raise SystemExit("No fingerprints loaded from source testing runs.")

    H_ref, d_ref = _load_human_reference(config)
    div = _diversity_components_batch(details_batch, fingerprints_by_task, H_ref, d_ref)
    intra_div = float(div.get("intra_set_diversity", 0.0))
    p_human = [disc.predict_human_probability(fp) for fp in all_fps]
    human_likeness = float(np.mean(p_human))
    combined = (
        float(config.lambda_human_likeness) * human_likeness
        + float(config.lambda_intra_diversity) * intra_div
    )
    turns = [
        float(ep.get("n_turns"))
        for ep in merged_per_episode
        if isinstance(ep.get("n_turns"), (int, float))
    ]
    metrics: Dict[str, Any] = {
        "combined_score": combined,
        "human_likeness": human_likeness,
        "intra_set_diversity": intra_div,
        "persona_success_rate": float(np.mean(successes)) if successes else 0.0,
        "n_episodes": len(all_trajectories),
        "failure_mode_distribution": {},
        "per_episode": merged_per_episode,
        "n_tasks_in_batch": float(len(test_tasks)),
        "n_personas_per_task": float(max((len(p) for p in persona_batch), default=0)),
        "total_rollouts": float(len(all_trajectories)),
        "mean_turns": float(np.mean(turns)) if turns else 0.0,
        "cap_hit_rate": (
            float(np.mean([t >= config.max_turns_per_episode for t in turns]))
            if turns and config.max_turns_per_episode
            else 0.0
        ),
        "merged_from_testing_dirs": [p.name for p in source_dirs],
    }

    out_dir = Path(config.testing_dir) / out_name
    return _write_benchmark_outputs(
        config=config,
        out_dir=out_dir,
        label=out_name,
        program_path=None,
        metrics=metrics,
        trajectories=all_trajectories,
        task_dicts=test_tasks,
        fingerprints_by_task=fingerprints_by_task,
        n_personas=int(metrics["n_personas_per_task"] or 0),
        persona_batch=persona_batch,
        task_contexts=task_contexts,
        details_batch=details_batch,
        selection={
            "selection_rule": "merged_existing_testing_runs",
            "source_testing_dirs": [p.name for p in source_dirs],
        },
    )


def run(
    config: PersonaPoliciesConfig,
    best_program_path: Path,
    n_tasks: Optional[int] = None,
    n_personas: Optional[int] = None,
    out_name: Optional[str] = None,
    selection: Optional[Dict[str, Any]] = None,
) -> Path:
    config.ensure_output_dirs()

    module = _load_evolved_module(str(best_program_path))
    if module is None:
        raise SystemExit(f"Could not load best_program: {best_program_path}")

    test_ids = load_split_task_ids(
        config.taubench_domain,
        config.task_split_benchmark,
        Path(config.taubench_root),
    )
    if not test_ids:
        raise SystemExit(
            f"No tau2 {config.taubench_domain} {config.task_split_benchmark} "
            "task ids (check split_tasks.json)."
        )
    batch = n_tasks if (n_tasks is not None and n_tasks > 0) else len(test_ids)
    n = n_personas if (n_personas is not None and n_personas > 0) else config.n_personas

    evaluator = PersonaPolicyEvaluator(config)
    evaluator.runner.use_split(config.task_split_benchmark)

    out_dir = Path(config.testing_dir) / (out_name or _derive_stem(best_program_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n[BENCH] best={best_program_path}\n"
        f"        split={config.task_split_benchmark} "
        f"tasks={batch}/{len(test_ids)} personas/task={n} "
        f"workers={config.parallel_episode_workers}\n"
        f"        out={out_dir}"
    )

    rng = random.Random(config.seed)
    metrics, trajectories, persona_batch, task_dicts, task_contexts, details_batch, persona_policies, fingerprints_by_task = _run_batch(
        module, test_ids, config, evaluator, rng, "test",
        sequential_start=0, batch_size=batch, n_personas=n, label=out_dir.name,
    )

    return _write_benchmark_outputs(
        config=config,
        out_dir=out_dir,
        label=out_dir.name,
        program_path=best_program_path,
        metrics=metrics,
        trajectories=trajectories,
        task_dicts=task_dicts,
        persona_batch=persona_batch,
        task_contexts=task_contexts,
        details_batch=details_batch,
        fingerprints_by_task=fingerprints_by_task,
        n_personas=n,
        selection=selection,
    )


def run_no_persona_baseline(
    config: PersonaPoliciesConfig,
    n_tasks: Optional[int] = None,
    out_name: Optional[str] = None,
) -> Path:
    """Benchmark tau2 default user simulator with no persona policy injection."""
    config.ensure_output_dirs()
    test_ids = load_split_task_ids(
        config.taubench_domain,
        config.task_split_benchmark,
        Path(config.taubench_root),
    )
    if not test_ids:
        raise SystemExit(
            f"No tau2 {config.taubench_domain} {config.task_split_benchmark} "
            "task ids (check split_tasks.json)."
        )
    batch = n_tasks if (n_tasks is not None and n_tasks > 0) else len(test_ids)
    task_dicts = take_task_batch_sequential(
        test_ids,
        batch,
        0,
        config.taubench_domain,
        Path(config.taubench_root),
    )
    if not task_dicts:
        raise SystemExit("No task dicts loaded for no_persona baseline.")

    evaluator = PersonaPolicyEvaluator(config)
    evaluator.runner.use_split(config.task_split_benchmark)
    out_dir = Path(config.testing_dir) / (out_name or "baseline_no_persona")

    print(
        f"\n[BENCH] baseline=no_persona\n"
        f"        split={config.task_split_benchmark} "
        f"tasks={len(task_dicts)}/{len(test_ids)} personas/task=1 "
        f"workers={config.parallel_episode_workers}\n"
        f"        out={out_dir}"
    )

    jobs: List[tuple[int, int, str, Dict[str, Any]]] = []
    for ti, task in enumerate(task_dicts):
        tid = str(task.get("_combined_id", task.get("id", ti)))
        task_idx = _find_task_idx(evaluator, tid)
        if task_idx is None:
            print(f"  Skipping task {tid} (not in runner)")
            continue
        jobs.append((ti, task_idx, tid, task))
    if not jobs:
        raise SystemExit("No no_persona baseline jobs matched runner tasks.")

    def _run_one(job: tuple[int, int, str, Dict[str, Any]]) -> tuple[int, Dict[str, Any]]:
        ti, task_idx, tid, _task = job
        try:
            result = evaluator.runner.run_episode(
                task_idx=task_idx,
                persona_policy_text=None,
                verbose=False,
            )
            fp = evaluator.extractor.compute_fingerprint(result["trajectory"])
            return ti, {
                "ok": True,
                "task_id": tid,
                "trajectory": result["trajectory"],
                "success": result["success"],
                "failure_mode": result.get("failure_mode"),
                "reward": float(result.get("reward", 0.0)),
                "n_turns": int(result.get("n_turns", 0)),
                "fingerprint": fp,
            }
        except Exception as exc:
            return ti, {"ok": False, "task_id": tid, "error": str(exc)}

    results: Dict[int, Dict[str, Any]] = {}
    n_workers = max(1, int(getattr(config, "parallel_episode_workers", 1) or 1))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one, job): job for job in jobs}
        for fut in as_completed(futures):
            ti, row = fut.result()
            results[ti] = row
            if not row.get("ok"):
                print(f"  task {row.get('task_id')}: FAILED ({row.get('error')})")

    trajectories: List = []
    successes: List[bool] = []
    failure_modes: List[Any] = []
    fingerprints: List[BehavioralFingerprint] = []
    fingerprints_by_task: List[List[BehavioralFingerprint]] = [[] for _ in task_dicts]
    task_contexts = [format_user_scenario_c(t) for t in task_dicts]
    successful_records: List[Dict[str, Any]] = []
    for ti, task in enumerate(task_dicts):
        row = results.get(ti)
        if not row or not row.get("ok"):
            continue
        trajectories.append(row["trajectory"])
        successes.append(bool(row["success"]))
        failure_modes.append(row.get("failure_mode"))
        fp = row["fingerprint"]
        fingerprints.append(fp)
        fingerprints_by_task[ti].append(fp)
        successful_records.append({
            "task_id": row["task_id"],
            "task_batch_idx": ti,
            "persona_idx": 0,
            "reward": row["reward"],
            "n_turns": row["n_turns"],
            "success": row["success"],
            "failure_mode": row.get("failure_mode"),
        })

    if not trajectories:
        raise SystemExit("No successful no_persona baseline trajectories.")

    H_ref, d_ref = _load_human_reference(config)
    details_batch_div = [
        [{"expanded_instruction": None}] if fingerprints_by_task[ti] else []
        for ti in range(len(task_dicts))
    ]
    div = _diversity_components_batch(details_batch_div, fingerprints_by_task, H_ref, d_ref)
    intra_div = float(div.get("intra_set_diversity", 0.0))

    metrics, _ = evaluator._metrics_from_episodes(
        "",
        trajectories,
        successes,
        failure_modes,
        persona_policies=[""] * len(trajectories),
        task_contexts=[
            task_contexts[r["task_batch_idx"]]
            for r in successful_records
            if isinstance(r.get("task_batch_idx"), int)
        ],
        intra_set_diversity=intra_div,
        n_variants=len(trajectories),
        precomputed_fingerprints=fingerprints,
    )
    per_ep = metrics.get("per_episode", [])
    for i, ep in enumerate(per_ep):
        if i < len(successful_records):
            ep.update({k: v for k, v in successful_records[i].items() if k not in ep})
    metrics["n_tasks_in_batch"] = float(len(task_dicts))
    metrics["n_personas_per_task"] = 1.0
    metrics["total_rollouts"] = float(len(trajectories))

    return _write_benchmark_outputs(
        config=config,
        out_dir=out_dir,
        label="baseline_no_persona",
        program_path=None,
        metrics=metrics,
        trajectories=trajectories,
        task_dicts=task_dicts,
        persona_batch=[[None] for _ in task_dicts],
        task_contexts=[format_user_scenario_c(t) for t in task_dicts],
        details_batch=[[{"expanded_instruction": None}] for _ in task_dicts],
        fingerprints_by_task=fingerprints_by_task,
        n_personas=1,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate best_program.py on τ² test split, or re-run val ablations (see --val-sweep).",
    )
    p.add_argument(
        "--val-sweep",
        action="store_true",
        help="Run held-out val (same as evolution val) for checkpoint(s) at multiple n_personas; "
        "appends val_curve-compatible JSONL (see --val-curve, --openevolve-dir).",
    )
    p.add_argument(
        "--val-sweep-plus-plot",
        action="store_true",
        help="One-command workflow: sweep n_personas for every val_curve iteration and "
        "for train iterations above --train-hl-threshold, then regenerate plots.",
    )
    p.add_argument(
        "--best-program",
        default=None,
        help="Path to best_program.py (not used with --val-sweep or --baseline).",
    )
    p.add_argument(
        "--best-by-val-avg",
        action="store_true",
        help="Benchmark the checkpoint whose mean val_combined_score over --n-personas-list "
        "(default 5,8,10) is highest. Uses --val-curve, --val-sweep-out, and --openevolve-dir.",
    )
    p.add_argument(
        "--merge-testing-runs",
        action="store_true",
        help="Merge existing testing folders, rescore under --domain, and regenerate artifacts without rerunning conversations.",
    )
    p.add_argument(
        "--source-testing",
        default=None,
        help="With --merge-testing-runs: comma-separated testing folder names or paths.",
    )
    p.add_argument(
        "--baseline",
        choices=("no_persona", "seed_initial_generator", "direct_llm_personas"),
        default=None,
        help="Run a built-in baseline on the test split.",
    )
    p.add_argument("--n-tasks", type=int, default=None, help="Default: full test split")
    p.add_argument(
        "--n-personas",
        type=int,
        default=None,
        help="Test benchmark: default config.n_personas. Ignored if --n-personas-list is set (use --val-sweep).",
    )
    p.add_argument(
        "--n-personas-list",
        type=str,
        default=None,
        help="With --val-sweep: comma-separated N values (default: 5,8,10). N matching config and already in val_curve is skipped.",
    )
    p.add_argument(
        "--openevolve-dir",
        type=str,
        default=None,
        help="With --val-sweep: directory containing checkpoints/ (default: parent of val_curve → …/openevolve).",
    )
    p.add_argument(
        "--checkpoint-iterations",
        type=str,
        default=None,
        help="With --val-sweep: explicit iteration list, e.g. 1,9,19 (overrides auto-pick).",
    )
    p.add_argument(
        "--val-curve",
        type=str,
        default=None,
        help="val_curve.jsonl (default: <config.training_results_dir>/val_curve.jsonl); used for metadata + dedupe.",
    )
    p.add_argument(
        "--train-curve",
        type=str,
        default=None,
        help="With --val-sweep: train_curve.jsonl path (default: sibling of val_curve).",
    )
    p.add_argument(
        "--train-hl-threshold",
        type=float,
        default=None,
        metavar="T",
        help="With --val-sweep auto-pick: include iterations where train_human_likeness > T. "
        "Default: config val_sweep_train_hl_threshold (see persona_policies/config.py). "
        "If no rows match, fallback is all iterations present in val_curve.",
    )
    p.add_argument(
        "--val-sweep-out",
        type=str,
        default=None,
        help="Output JSONL path (default: <training_results_dir>/val_n_personas_sweep.jsonl).",
    )
    p.add_argument("--domain", default=None, help="Override config.taubench_domain")
    p.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Validation fraction inside official train split for --val-sweep (default: config.val_fraction)",
    )
    p.add_argument("--out-name", default=None, help="Test benchmark: subdir under testing/")
    p.add_argument(
        "--parallel-episode-workers",
        type=int,
        default=None,
        metavar="N",
        help="Thread pool size for τ² rollouts per batch (default: config.parallel_episode_workers; "
        "same as evolution / train simulation).",
    )
    args = p.parse_args()

    cfg = PersonaPoliciesConfig()
    if args.domain:
        cfg.taubench_domain = args.domain
        cfg.refresh_domain_artifact_paths()
    if args.val_fraction is not None:
        cfg.val_fraction = float(args.val_fraction)
    if args.parallel_episode_workers is not None and int(args.parallel_episode_workers) > 0:
        cfg.parallel_episode_workers = int(args.parallel_episode_workers)

    if args.val_sweep or args.val_sweep_plus_plot:
        cfg.ensure_output_dirs()
        vc = Path(args.val_curve) if args.val_curve else Path(cfg.training_results_dir) / "val_curve.jsonl"
        if not vc.is_absolute():
            vc = _REPO_ROOT / vc
        odir = Path(args.openevolve_dir) if args.openevolve_dir else _openevolve_dir_from_val_curve(vc)
        if not odir.is_absolute():
            odir = _REPO_ROOT / odir
        npl = args.n_personas_list or "5,8,10"
        cit: Optional[List[int]] = None
        if args.checkpoint_iterations and str(args.checkpoint_iterations).strip():
            cit = _parse_int_list(args.checkpoint_iterations)
        outj = Path(args.val_sweep_out) if args.val_sweep_out else None
        if outj is not None and not outj.is_absolute():
            outj = _REPO_ROOT / outj
        tc = Path(args.train_curve) if args.train_curve else None
        if tc is not None and not tc.is_absolute():
            tc = _REPO_ROOT / tc
        hl_t = (
            float(args.train_hl_threshold)
            if args.train_hl_threshold is not None
            else float(cfg.val_sweep_train_hl_threshold)
        )
        if args.val_sweep_plus_plot:
            run_val_sweep_plus_plot(
                cfg,
                openevolve_dir=odir,
                n_personas_list=_parse_int_list(npl),
                val_curve_path=vc,
                train_curve_path=tc,
                train_hl_threshold=hl_t,
                output_jsonl=outj or (Path(cfg.training_results_dir) / "val_n_personas_sweep.jsonl"),
                plot_output=None,
            )
            return
        run_val_n_sweep(
            cfg,
            odir,
            cit,
            _parse_int_list(npl),
            val_curve_path=vc,
            train_curve_path=tc,
            train_hl_threshold=hl_t,
            output_jsonl=outj,
        )
        return

    if args.best_by_val_avg:
        cfg.ensure_output_dirs()
        vc = Path(args.val_curve) if args.val_curve else Path(cfg.training_results_dir) / "val_curve.jsonl"
        if not vc.is_absolute():
            vc = _REPO_ROOT / vc
        sweep_out = (
            Path(args.val_sweep_out)
            if args.val_sweep_out
            else Path(cfg.training_results_dir) / "val_n_personas_sweep.jsonl"
        )
        if not sweep_out.is_absolute():
            sweep_out = _REPO_ROOT / sweep_out
        odir = Path(args.openevolve_dir) if args.openevolve_dir else _openevolve_dir_from_val_curve(vc)
        if not odir.is_absolute():
            odir = _REPO_ROOT / odir
        npl = _parse_int_list(args.n_personas_list or "5,8,10")
        best_path, selection = select_best_program_by_val_n_average(
            config=cfg,
            val_curve_path=vc,
            sweep_path=sweep_out,
            openevolve_dir=odir,
            n_personas_list=npl,
        )
        print(
            f"[SELECT] iter={selection['iteration']} "
            f"mean_val={selection['mean_val_combined_score']:.4f} "
            f"by_n={selection['val_combined_by_n']} "
            f"program={best_path}",
            flush=True,
        )
        run(
            cfg,
            best_path,
            n_tasks=args.n_tasks,
            n_personas=args.n_personas,
            out_name=args.out_name or (
                f"{canonical_domain_name(cfg.taubench_domain)}_best_val_avg_"
                f"n{'_'.join(str(n) for n in selection['n_personas_list'])}"
                f"_iter{int(selection['iteration']):04d}"
            ),
            selection=selection,
        )
        return

    if args.merge_testing_runs:
        if not args.source_testing:
            raise SystemExit("--merge-testing-runs requires --source-testing a,b")
        if not args.out_name:
            raise SystemExit("--merge-testing-runs requires --out-name")
        merge_testing_runs(
            cfg,
            source_names=args.source_testing,
            out_name=args.out_name,
        )
        return

    if args.baseline:
        if args.baseline == "no_persona":
            run_no_persona_baseline(
                cfg,
                n_tasks=args.n_tasks,
                out_name=args.out_name or f"baseline_{args.baseline}",
            )
            return
        run(
            cfg,
            BASELINE_PROGRAMS[args.baseline],
            n_tasks=args.n_tasks,
            n_personas=args.n_personas,
            out_name=args.out_name or f"baseline_{args.baseline}",
        )
        return

    if not args.best_program:
        raise SystemExit(
            "Provide --best-program, use --baseline, or use --val-sweep "
            "(see --val-curve)."
        )

    best = Path(args.best_program)
    if not best.is_absolute():
        best = _REPO_ROOT / best
    if not best.is_file():
        raise SystemExit(f"best_program not found: {best}")

    run(cfg, best, n_tasks=args.n_tasks, n_personas=args.n_personas, out_name=args.out_name)


if __name__ == "__main__":
    main()
