"""OpenEvolve ``evaluate(program_path)``.

**Per-iteration flow (one OE iter == one training "batch step"):**
  1. Load the evolved G(c, D, N) program at ``program_path``.
  2. Training step: sample a random minibatch of train tasks. For each task,
     build context c, call G(c, D, N) to get N persona strings, then run one 
     episode per (persona, task) pair in parallel. Metrics aggregate → ``train_score``.
  3. Return ``combined_score = train_score`` so OpenEvolve's archive / islands use 
     train-only performance to decide whether to keep the candidate as an elite.
  4. Validation (monitoring only): at the start of each ``evaluate()``, if the 
     current elite program on disk has changed since the last validation,
     spawn a daemon thread (no ``join()``) that runs full val and appends to
     ``training/results/val_curve.jsonl``. Training never waits on val.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openevolve.evaluation_result import EvaluationResult
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from persona_policies.config import PersonaPoliciesConfig
from persona_policies.config import domain_list
from persona_policies.evolution.metrics_plot import (
    append_train_curve_row,
    refresh_evolution_plots,
)
from persona_policies.fingerprinting import BehavioralFingerprint, REGEX_FEATURES

_EVAL_BATCH_SEQ = 0
_SPLIT_LOGGED = False

# Max chars of task context stored per task under training/simulations/.../log.json.
_ITERATION_LOG_TASK_CONTEXT_CHARS = 2500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _early_stop_marker(config: PersonaPoliciesConfig) -> Path:
    return Path(config.openevolve_output_dir) / "EARLY_STOP"


def _early_stop_requested(config: PersonaPoliciesConfig) -> bool:
    return _early_stop_marker(config).is_file()


def _early_stop_result(iteration: Optional[int] = None) -> EvaluationResult:
    metrics: Dict[str, float] = {
        "combined_score": 0.0,
        "train_score": 0.0,
        "early_stop_requested": 1.0,
    }
    if iteration is not None:
        metrics["iteration"] = float(iteration)
    return EvaluationResult(metrics=metrics, artifacts={})


def _train_n_personas_for_epoch(config: PersonaPoliciesConfig, epoch: int) -> int:
    """Returns the number of personas to use for training at a given epoch, using ``config.n_personas_schedule`` (see ``PersonaPoliciesConfig``). Validation always uses ``config.n_personas``."""
    n_default = max(1, int(config.n_personas))
    if not config.curriculum:
        return n_default
    n = n_default
    for thresh, n_p in sorted(config.n_personas_schedule, key=lambda t: int(t[0])):
        if int(epoch) >= int(thresh):
            n = max(1, int(n_p))
    return n


def _scoring_weights_for_n_personas(
    config: PersonaPoliciesConfig,
    n_personas: int,
) -> Tuple[float, float]:
    """Curriculum-aware train weights: ramp diversity pressure up with N."""
    n_final = max(1, int(config.n_personas))
    n_current = max(1, int(n_personas))
    ratio = min(1.0, n_current / n_final)
    lambda_b = float(config.lambda_intra_diversity) * ratio
    lambda_h = 1.0 - lambda_b
    return lambda_h, lambda_b


def _persona_reasoning_field(d: Dict[str, Any]) -> Any:
    """Stage-1 ``reasoning`` field when present."""
    if not isinstance(d, dict):
        return None
    return d.get("reasoning")


def _load_evolved_module(program_path: str):
    """Import the evolved program as a module."""
    spec = importlib.util.spec_from_file_location("evolved_persona", program_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _error_result(msg: str = "") -> EvaluationResult:
    # Include every key used as an OpenEvolve ``feature_dimension`` (see
    # openevolve_config.yaml) so MAP-Elites can always bin an errored program
    # instead of crashing with "... not found in program metrics".
    out: Dict[str, Any] = {
        "combined_score": 0.01,
        "error": 1.0,
        "human_likeness": 0.0,
        "intra_set_diversity": 0.0,
    }
    if msg:
        print(f"[EVAL ERROR] {msg}")
    return EvaluationResult(metrics=out)


def _find_task_idx(evaluator: Any, task_id: str):
    for idx in range(len(evaluator.runner._tasks)):
        t = evaluator.runner._tasks[idx]
        key_fn = getattr(evaluator.runner, "get_task_key", None)
        key = key_fn(idx) if callable(key_fn) else str(t.id)
        if str(key) == str(task_id) or str(t.id) == str(task_id):
            return idx
    return None


def _task_id_for_log(task_dict: Dict[str, Any], fallback: int = 0) -> str:
    return str(task_dict.get("_combined_id", task_dict.get("id", fallback)))


def _module_has_generator(module: Any) -> bool:
    if callable(getattr(module, "generate_personas_detailed", None)):
        return True
    s1, s2 = getattr(module, "stage1_archetypes", None), getattr(module, "stage2_expand", None)
    return callable(s1) and callable(s2)


def _attach_diversity_axes_snapshot(log: Optional[Dict[str, Any]], module: Any) -> None:
    """In-place: add ``diversity_axis_keys`` (and short labels) from ``DIVERSITY_AXES`` if present."""
    if not log:
        return
    ax = getattr(module, "DIVERSITY_AXES", None)
    if not isinstance(ax, list) or not ax:
        return
    keys = []
    axes_compact: List[Dict[str, Any]] = []
    for row in ax:
        if not isinstance(row, dict):
            continue
        k = row.get("behavior") or row.get("key")
        if k:
            keys.append(str(k))
        pres = row.get("presence") if isinstance(row.get("presence"), dict) else {}
        axes_compact.append({
            "behavior": k,
            "definition": (row.get("definition") or "")[:400],
            "presence": {
                "true": (pres.get("true") or "")[:200],
                "false": (pres.get("false") or "")[:200],
            },
        })
    log["diversity_axis_keys"] = keys
    log["diversity_axes"] = axes_compact


def _string_only_persona_meta(text: str) -> Dict[str, Any]:
    return {
        "persona_id": None,
        "description": None,
        "axis_placement": None,
        "reasoning": None,
        "expanded_instruction": text,
        "stage1_metadata_available": False,
    }


def _regex_feature_vector(fp: BehavioralFingerprint) -> np.ndarray:
    return np.array([fp.features.get(f, 0.0) for f in REGEX_FEATURES], dtype=np.float64)


def _fingerprint_to_vec_list(fp: BehavioralFingerprint) -> List[float]:
    """19 floats in ``REGEX_FEATURES`` order (same as ``BehavioralFingerprint.to_vector``)."""
    return [float(fp.features.get(name, 0.0)) for name in REGEX_FEATURES]


def _flatten_fingerprints_by_task(
    fps_by_task: List[List[BehavioralFingerprint]],
) -> List[BehavioralFingerprint]:
    """Same order as ``_run_batch`` flat ``all_trajectories`` (task-major, persona-minor)."""
    return [fp for row in fps_by_task for fp in row]


def _trajectory_json_episodes(
    trajectories: List,
    fps_by_task: Optional[List[List[BehavioralFingerprint]]],
) -> List[Dict[str, Any]]:
    """One entry per successful rollout: raw trajectory + 1-D fingerprint vector (no feature names)."""
    if not trajectories:
        return []
    if not fps_by_task:
        return [{"trajectory": t} for t in trajectories]
    flat = _flatten_fingerprints_by_task(fps_by_task)
    if len(flat) != len(trajectories):
        print(
            f"[WARN] trajectory/fingerprint length mismatch: "
            f"{len(trajectories)} trajs vs {len(flat)} fingerprints — saving trajectories only.",
            flush=True,
        )
        return [{"trajectory": t} for t in trajectories]
    return [
        {"trajectory": t, "fingerprint": _fingerprint_to_vec_list(fp)}
        for t, fp in zip(trajectories, flat, strict=True)
    ]


# ---------------------------------------------------------------------------
# Human reference fingerprints (for Chamfer-based diversity)
# ---------------------------------------------------------------------------

# Per-domain cache: (H, d_ref). H is (n_human, len(REGEX_FEATURES)) in REGEX_FEATURES order.
# d_ref = mean pairwise Euclidean distance among H (the typical human-to-human gap).
_HUMAN_REF_CACHE: Dict[str, Tuple[np.ndarray, float]] = {}


def _load_human_reference(config: PersonaPoliciesConfig) -> Tuple[np.ndarray, float]:
    """Load train-split human fingerprints for ``config.taubench_domain`` (cached per domain).

    Uses the official ``split_tasks.json`` labels to exclude test-split humans; dialogues
    with unknown task ids fall back into ``train`` (they're not held out for val anyway).

    Fail loud: Chamfer diversity is part of the active fitness signal. If human
    references can't be loaded (missing file, empty domain, etc.), we raise so the run stops at startup.
    """
    domain = config.taubench_domain
    if domain in _HUMAN_REF_CACHE:
        return _HUMAN_REF_CACHE[domain]

    from persona_policies.fingerprinting import BehavioralFingerprintExtractor
    from persona_policies.tau_human_loader import load_domain_dialogues
    from persona_policies.tau_train_context import (
        official_split_for_task_id,
        split_train_val,
        task_id_from_tau_human_instance_key,
    )

    extractor = BehavioralFingerprintExtractor()
    taubench_root = Path(config.taubench_root)
    _, evolution_val_ids = split_train_val(
        seed=config.seed,
        val_fraction=config.val_fraction,
        domain=domain,
        taubench_root=taubench_root,
    )
    evolution_val_set = set(evolution_val_ids)
    qualify_task_ids = len(domain_list(domain)) > 1
    vectors: List[np.ndarray] = []
    n_test = 0
    n_val = 0
    n_unlabeled = 0
    for dom in domain_list(domain):
        dialogues = load_domain_dialogues(str(config.tau_bench_human_path), dom)
        for d in dialogues:
            trace = d.get("conversation", d.get("turns", []))
            if len(trace) < 2:
                continue
            key = str(d.get("instance_id", ""))
            tid = task_id_from_tau_human_instance_key(key, dom) if key else ""
            try:
                sp = official_split_for_task_id(dom, tid, taubench_root) if tid else None
            except FileNotFoundError:
                sp = None
            if sp == "test":
                n_test += 1
                continue
            task_key = f"{dom}:{tid}" if qualify_task_ids and tid else str(tid or "")
            if task_key in evolution_val_set:
                n_val += 1
                continue
            if sp is None:
                n_unlabeled += 1
            fp = extractor.compute_fingerprint(trace, user_role="user", agent_role="assistant")
            vectors.append(fp.to_vector(REGEX_FEATURES))

    if not vectors:
        raise RuntimeError(
            f"[CHAMFER] no train-split human fingerprints loaded for "
            f"domain={domain!r} from {config.tau_bench_human_path!r}. "
            f"Chamfer diversity is part of the active fitness, so running "
            f"with zero humans would silently change the objective. Fix "
            f"the human dataset or switch off chamfer before retrying."
        )

    H = np.asarray(vectors, dtype=np.float64)
    if H.shape[0] >= 2:
        from scipy.spatial.distance import pdist
        d_ref = float(pdist(H, metric="euclidean").mean())
        if d_ref < 1e-9:
            d_ref = 1.0
    else:
        d_ref = 1.0

    print(
        f"[CHAMFER] human reference (domain={domain}): {H.shape[0]} train humans, "
        f"d_ref={d_ref:.4f}  (excluded val={n_val}, test={n_test}, "
        f"unlabeled-kept-as-train={n_unlabeled})",
        flush=True,
    )
    _HUMAN_REF_CACHE[domain] = (H, d_ref)
    return _HUMAN_REF_CACHE[domain]


def _chamfer_score(
    fingerprints: List[BehavioralFingerprint],
    H: np.ndarray,
    d_ref: float,
) -> float:
    """Two-sided Chamfer distance between personas and humans, rescaled to a [0, 1] reward.

    ``err = mean_h min_p ||h - p||  +  mean_p min_h ||p - h||``   (both in the 19-D
    regex fingerprint space, L2). Reward = ``1 - min(1, err / (2*d_ref))`` so it sits on
    the same scale as ``human_likeness``. ``d_ref`` is the mean
    pairwise distance within the human cloud — dividing by ``2*d_ref`` makes a score of
    1 mean "typical human spread" and lower values = worse coverage + drift.
    """
    if not fingerprints or H.shape[0] == 0 or d_ref <= 0:
        return 0.0
    P = np.asarray(
        [fp.to_vector(REGEX_FEATURES) for fp in fingerprints],
        dtype=np.float64,
    )
    if P.shape[0] == 0:
        return 0.0
    from scipy.spatial.distance import cdist
    D = cdist(H, P, metric="euclidean")           # (|H|, |P|)
    # cover_humans: every real human should have a near persona.
    # stay_on_manifold: every persona should sit near some real human.
    err = float(D.min(axis=1).mean() + D.min(axis=0).mean())
    scaled = err / (2.0 * d_ref)
    return float(max(0.0, 1.0 - min(1.0, scaled)))


def _diversity_components_per_task(
    personas_meta: List[Dict[str, Any]],
    fingerprints: Optional[List[BehavioralFingerprint]],
    H: np.ndarray,
    d_ref: float,
) -> float:
    """Return Chamfer-based behavioral coverage for one task.

    Chamfer coverage is defined for a singleton set too, which lets the
    default no-persona simulator receive a nonzero coverage score when its
    single rollout lies near the human reference cloud. Empty task/persona
    groups still score zero.
    """
    if not personas_meta:
        return 0.0
    return float(_chamfer_score(fingerprints or [], H, d_ref))


def _diversity_components_batch(
    details_batch: List[List[Dict[str, Any]]],
    fingerprints_batch: List[List[BehavioralFingerprint]],
    H: np.ndarray,
    d_ref: float,
) -> Dict[str, float]:
    """Mean of per-task components across the batch, keyed by metric name."""
    rows = [
        _diversity_components_per_task(ps, fps, H, d_ref)
        for ps, fps in zip(details_batch, fingerprints_batch, strict=True)
        if ps
    ]
    if not rows:
        return {"intra_set_diversity": 0.0}
    n = float(len(rows))
    score = sum(rows) / n
    return {"intra_set_diversity": score}


def _row_to_meta(row: Any) -> Tuple[str, Dict[str, Any]]:
    """Normalize a persona row (dict or string) to ``(expanded_text, meta_dict)``."""
    if not isinstance(row, dict):
        t = str(row).strip()
        return t, _string_only_persona_meta(t)
    exp = str(
        row.get("expanded_instruction")
        or row.get("stage2_expanded_instruction")
        or ""
    ).strip()
    ap = row.get("axis_placement")
    return exp, {
        "persona_id": row.get("persona_id"),
        "description": row.get("description"),
        "axis_placement": dict(ap) if isinstance(ap, dict) else ap,
        "reasoning": _persona_reasoning_field(row),
        "expanded_instruction": exp,
        "stage1_metadata_available": True,
    }


def _generate_task_personas(
    module: Any,
    c: str,
    n: int,
    axes: Any,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Call G(c, axes, n) and return ``(expanded strings, parallel log-dict metadata)``.

    Prefers ``generate_personas_detailed(c, axes, n)``; falls back to
    ``stage1_archetypes`` + ``stage2_expand`` if an evolved module removed it.
    """
    gd = getattr(module, "generate_personas_detailed", None)
    if callable(gd) and axes is not None:
        rows: List[Any] = list(gd(c, axes, n))
    else:
        s1 = getattr(module, "stage1_archetypes", None)
        s2 = getattr(module, "stage2_expand", None)
        if not (callable(s1) and callable(s2) and axes is not None):
            raise RuntimeError("evolved module must expose generate_personas_detailed(c, axes, n) or stage1_archetypes + stage2_expand")
        rows = [
            {**arch, "expanded_instruction": s2(arch, c, axes)}
            for arch in s1(c, axes, n)
            if isinstance(arch, dict)
        ]
    texts: List[str] = []
    meta: List[Dict[str, Any]] = []
    for row in rows:
        t, m = _row_to_meta(row)
        texts.append(t)
        meta.append(m)
    return texts, meta


# ---------------------------------------------------------------------------
# Per-iteration log: program + per-task personas + metrics
# ---------------------------------------------------------------------------

def _save_iteration_log(
    config: PersonaPoliciesConfig,
    iteration: int,
    program_path: str,
    train_data: Dict[str, Any] | None,
    val_data: Dict[str, Any] | None,
    outcome: str,
    final_score: float,
    *,
    train_trajectories: Optional[List] = None,
    train_fingerprints_by_task: Optional[List[List[BehavioralFingerprint]]] = None,
    val_trajectories: Optional[List] = None,
    val_fingerprints_by_task: Optional[List[List[BehavioralFingerprint]]] = None,
    reflection_text: str = "",
) -> bool:
    """Write per-iteration artifacts under ``…/training/simulations/iter_NNNN/``.

    In the current train-only fitness flow this is called **once per iter** with
    ``outcome="ok"`` (or ``"error"`` on train failure). Full-val on the current
    elite is logged separately under ``training/validation/iter_NNNN/`` and
    ``training/results/val_curve.jsonl`` when a new best program appears.

    Files:
      - ``program.py`` — copy of the evolved generator
      - ``log.json`` — compact log (train / val, personas, metrics)
      - ``trajectories.json`` — list of episodes ``{trajectory, fingerprint}`` per rollout;
        ``fingerprint`` is a length-19 float list in ``REGEX_FEATURES`` order (see
        ``fingerprint_vector_len`` / ``fingerprint_order_ref`` at JSON root).
      - ``reflection.txt`` — evaluator reflection (final call only, when provided)
    """
    log_dir = Path(config.simulations_dir) / f"iter_{iteration:04d}"
    log_dir.mkdir(parents=True, exist_ok=True)

    src = Path(program_path)
    if src.is_file():
        shutil.copy2(str(src), str(log_dir / "program.py"))

    entry: Dict[str, Any] = {
        "iteration": iteration,
        "program_path": program_path,
        "timestamp": time.time(),
        "outcome": outcome,
        "final_combined_score": final_score,
    }
    if train_data:
        entry["train"] = train_data
    if val_data:
        entry["val"] = val_data

    (log_dir / "log.json").write_text(
        json.dumps(entry, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # Trajectories are large — save separately so log.json stays manageable
    traj_data: Dict[str, Any] = {
        "fingerprint_vector_len": len(REGEX_FEATURES),
        "fingerprint_order_ref": "persona_policies.fingerprinting.REGEX_FEATURES",
    }
    if train_trajectories:
        traj_data["train"] = _trajectory_json_episodes(
            train_trajectories, train_fingerprints_by_task
        )
    if val_trajectories:
        traj_data["val"] = _trajectory_json_episodes(
            val_trajectories, val_fingerprints_by_task
        )
    if traj_data.get("train") or traj_data.get("val"):
        (log_dir / "trajectories.json").write_text(
            json.dumps(traj_data, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    if reflection_text:
        (log_dir / "reflection.txt").write_text(reflection_text, encoding="utf-8")


def _phase_log_data(
    task_dicts: List[Dict[str, Any]],
    persona_batch: List[List[str]],
    metrics: Dict[str, Any],
    task_contexts: List[str],
    details_batch: List[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Per-task rows + full ``metrics`` for ``log.json``."""
    tasks = []
    for i, td in enumerate(task_dicts):
        c = task_contexts[i] if i < len(task_contexts) else ""
        row: Dict[str, Any] = {
            "task_id": _task_id_for_log(td),
            "task_context": (c or "")[:_ITERATION_LOG_TASK_CONTEXT_CHARS],
        }
        personas_out: List[Dict[str, Any]] = []
        ps = persona_batch[i] if i < len(persona_batch) else []
        ds = details_batch[i] if i < len(details_batch) else []
        for j, ptext in enumerate(ps):
            base: Dict[str, Any] = dict(ds[j]) if j < len(ds) and isinstance(ds[j], dict) else {}
            base["expanded_instruction"] = ptext
            personas_out.append(base)
        row["personas"] = personas_out
        tasks.append(row)
    return {"tasks": tasks, "metrics": metrics}


# ---------------------------------------------------------------------------
# _run_batch — generates personas for a batch of tasks and scores them
# ---------------------------------------------------------------------------

def _run_batch(
    module,
    task_ids: List[str],
    config: PersonaPoliciesConfig,
    evaluator,
    rng: random.Random,
    phase: str,
    *,
    sequential_start: Optional[int] = None,
    batch_size: Optional[int] = None,
    n_personas: Optional[int] = None,
    label: str = "",
    explicit_task_dicts: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[
    Dict[str, Any],
    List,
    List[List[str]],
    List[Dict[str, Any]],
    List[str],
    List[List[Dict[str, Any]]],
    List[str],
    List[List[BehavioralFingerprint]],
]:
    """G(c,D,N) per task + parallel τ²; sliding batch if ``sequential_start`` is int else random.

    The 8th return element is ``fingerprints_by_task``: a list of per-task lists
    of fingerprints (one per (task, persona) rollout), aligned with ``task_dicts``.
    Empty inner lists are kept if all rollouts for a task failed.
    """
    from persona_policies.tau_train_context import (
        format_user_scenario_c,
        sample_task_batch,
        take_task_batch_sequential,
    )

    nb = max(1, batch_size if batch_size is not None else config.eval_batch_size)
    n = max(1, n_personas if n_personas is not None else config.n_personas)

    if explicit_task_dicts is not None:
        task_dicts = [t for t in explicit_task_dicts if t]
    elif sequential_start is not None:
        task_dicts = take_task_batch_sequential(
            task_ids,
            nb,
            sequential_start,
            config.taubench_domain,
            Path(config.taubench_root),
        )
    else:
        task_dicts = sample_task_batch(
            task_ids,
            nb,
            rng,
            config.taubench_domain,
            Path(config.taubench_root),
        )
    if not task_dicts:
        return {"combined_score": 0.01, "error": 1.0}, [], [], [], [], [], [], []

    task_contexts = [format_user_scenario_c(t) for t in task_dicts]
    axes = getattr(module, "DIVERSITY_AXES", None)

    # Persona generation is LLM-bound (1 Stage-1 + N Stage-2 calls per task).
    # Fan out across tasks in parallel so B tasks don't serialize B LLM roundtrips.
    persona_batch: List[List[str]] = [[] for _ in task_contexts]
    details_batch: List[List[Dict[str, Any]]] = [[] for _ in task_contexts]
    n_gen_workers = min(
        len(task_contexts),
        max(1, int(getattr(config, "parallel_episode_workers", 1) or 1)),
    )
    try:
        if len(task_contexts) > 1 and n_gen_workers > 1:
            with ThreadPoolExecutor(max_workers=n_gen_workers) as pool:
                futs = {
                    pool.submit(_generate_task_personas, module, c, n, axes): ti
                    for ti, c in enumerate(task_contexts)
                }
                for fut in as_completed(futs):
                    ti = futs[fut]
                    texts, meta = fut.result()
                    persona_batch[ti] = texts
                    details_batch[ti] = meta
        else:
            for ti, c in enumerate(task_contexts):
                texts, meta = _generate_task_personas(module, c, n, axes)
                persona_batch[ti] = texts
                details_batch[ti] = meta
    except Exception as e:
        tb = traceback.format_exc()
        return (
            {"combined_score": 0.01, "error": 1.0, "stderr": f"{e}\n{tb[-500:]}"},
            [],
            [],
            task_dicts,
            task_contexts,
            [[] for _ in task_dicts],
            [],
            [],
        )

    for ti in range(len(persona_batch)):
        ps = persona_batch[ti]
        ds = details_batch[ti]
        new_ps: List[str] = []
        new_ds: List[Dict[str, Any]] = []
        for j, p in enumerate(ps):
            p = str(p).strip()
            if p and len(p) >= 40:
                new_ps.append(p)
                base = dict(ds[j]) if j < len(ds) and isinstance(ds[j], dict) else _string_only_persona_meta(p)
                base["expanded_instruction"] = p
                new_ds.append(base)
        persona_batch[ti] = new_ps
        details_batch[ti] = new_ds
    total_personas = sum(len(ps) for ps in persona_batch)
    if total_personas == 0:
        return (
            {"combined_score": 0.01, "error": 1.0},
            [],
            persona_batch,
            task_dicts,
            task_contexts,
            details_batch,
            [],
            [],
        )

    banner = f"[{phase.upper()}]"
    if label:
        banner = f"{banner} {label}"
    n_workers = max(1, int(getattr(config, "parallel_episode_workers", 1) or 1))
    print(f"\n{'='*60}")
    print(
        f"{banner} G(c,D,N): {len(task_contexts)} tasks x {n} personas "
        f"= {total_personas} rollouts  (workers={n_workers})"
    )
    print("=" * 60)

    # ---- τ² episodes (parallel) ----
    # Build the flat job list; preserve (ti, pi) so we can reassemble in order.
    jobs: List[Tuple[int, int, int, str, str]] = []  # (ti, pi, task_idx, task_id, persona_text)
    skipped_tasks: List[str] = []
    for ti, (task_dict, personas) in enumerate(zip(task_dicts, persona_batch)):
        task_id = _task_id_for_log(task_dict, ti)
        task_idx = _find_task_idx(evaluator, task_id)
        if task_idx is None:
            skipped_tasks.append(task_id)
            continue
        for pi, persona_text in enumerate(personas):
            jobs.append((ti, pi, task_idx, task_id, persona_text))

    for tid in skipped_tasks:
        tqdm.write(f"  Skipping task {tid} (not in runner)")

    def _run_one(job):
        ti, pi, task_idx, task_id, persona_text = job
        max_steps = (
            getattr(config, "train_max_steps_per_episode", None)
            if phase == "train"
            else None
        )
        try:
            result = evaluator.runner.run_episode(
                task_idx=task_idx,
                persona_policy_text=persona_text,
                max_turns=max_steps,
                verbose=False,
            )
            fp = evaluator.extractor.compute_fingerprint(result["trajectory"])
            return (ti, pi, {
                "ok": True,
                "task_id": task_id,
                "trajectory": result["trajectory"],
                "success": result["success"],
                "failure_mode": result.get("failure_mode"),
                "reward": float(result.get("reward", 0.0)),
                "n_turns": int(result.get("n_turns", 0)),
                "fingerprint": fp,
            })
        except Exception as e:
            return (ti, pi, {
                "ok": False,
                "task_id": task_id,
                "error": str(e),
            })

    results_by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
    if jobs:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_one, j): j for j in jobs}
            pbar = tqdm(
                total=len(futures),
                desc=f"{banner} rollouts",
                unit="ep",
                dynamic_ncols=True,
            )
            try:
                for fut in as_completed(futures):
                    ti, pi, r = fut.result()
                    results_by_key[(ti, pi)] = r
                    if r.get("ok"):
                        pbar.set_postfix_str(
                            f"t={r['task_id']} p={pi+1}/{n} "
                            f"rew={r['reward']:.1f} turns={r['n_turns']}",
                            refresh=False,
                        )
                    else:
                        tqdm.write(
                            f"  task {r.get('task_id')} persona {pi+1}/{n}: "
                            f"FAILED ({r.get('error')})"
                        )
                    pbar.update(1)
            finally:
                pbar.close()

    # Reassemble in deterministic (task-major, persona-minor) order so downstream
    # per_episode / fingerprint / trajectory lists stay aligned with persona_batch.
    all_trajectories: List = []
    all_successes: List[bool] = []
    all_failure_modes: List = []
    all_fingerprints: List[Any] = []
    fingerprints_by_task: List[List[BehavioralFingerprint]] = [[] for _ in persona_batch]
    judge_persona_policies: List[str] = []
    judge_task_contexts: List[str] = []
    episode_records: List[Dict[str, Any]] = []
    successful_episode_records: List[Dict[str, Any]] = []
    for ti, (task_dict, personas) in enumerate(zip(task_dicts, persona_batch)):
        task_id = _task_id_for_log(task_dict, ti)
        tc_for_task = task_contexts[ti] if ti < len(task_contexts) else ""
        for pi, persona_text in enumerate(personas):
            r = results_by_key.get((ti, pi))
            if r is None:
                episode_records.append({
                    "task_id": task_id,
                    "persona_idx": pi,
                    "error": "skipped (task not in runner)",
                })
                continue
            if not r.get("ok"):
                episode_records.append({
                    "task_id": task_id,
                    "persona_idx": pi,
                    "error": r.get("error", "unknown"),
                })
                continue
            all_trajectories.append(r["trajectory"])
            all_successes.append(r["success"])
            all_failure_modes.append(r.get("failure_mode"))
            fp = r["fingerprint"]
            all_fingerprints.append(fp)
            fingerprints_by_task[ti].append(fp)
            judge_persona_policies.append(str(persona_text))
            judge_task_contexts.append(str(tc_for_task))
            ok_record = {
                "task_id": task_id,
                "task_batch_idx": ti,
                "persona_idx": pi,
                "reward": r["reward"],
                "n_turns": r["n_turns"],
                "success": r["success"],
                "failure_mode": r.get("failure_mode"),
            }
            episode_records.append(ok_record)
            successful_episode_records.append(ok_record)

    if not all_trajectories:
        return (
            {"combined_score": 0.01, "error": 1.0},
            [],
            persona_batch,
            task_dicts,
            task_contexts,
            details_batch,
            [],
            [],
        )

    # ---- Intra-set diversity: Chamfer distance vs humans (τ²) ----
    H_ref, d_ref = _load_human_reference(config)
    div = _diversity_components_batch(details_batch, fingerprints_by_task, H_ref, d_ref)
    intra_div = div["intra_set_diversity"]
    scoring_weights = (
        _scoring_weights_for_n_personas(config, n)
        if phase == "train"
        else None
    )

    # ---- Aggregate metrics (includes per_episode detail from evaluator) ----
    metrics, _ = evaluator._metrics_from_episodes(
        persona_batch[0][0] if persona_batch and persona_batch[0] else "",
        all_trajectories,
        all_successes,
        all_failure_modes,
        persona_policies=judge_persona_policies,
        task_contexts=judge_task_contexts,
        intra_set_diversity=intra_div,
        n_variants=total_personas,
        precomputed_fingerprints=all_fingerprints,
        scoring_weights=scoring_weights,
    )
    # Merge successful episode metadata (task_id, persona_idx, reward, turns)
    # into per_episode. Failed/skipped records are not present in per_episode.
    per_ep = metrics.get("per_episode", [])
    for i, ep in enumerate(per_ep):
        if i < len(successful_episode_records):
            ep.update({
                k: v for k, v in successful_episode_records[i].items()
                if k not in ep
            })
            ti = successful_episode_records[i].get("task_batch_idx")
            pi = successful_episode_records[i].get("persona_idx")
            if isinstance(ti, int) and isinstance(pi, int):
                try:
                    details_batch[ti][pi]["human_likeness"] = ep.get("p_human")
                except (IndexError, TypeError):
                    pass

    metrics["n_tasks_in_batch"] = float(len(task_dicts))
    metrics["n_personas_per_task"] = float(n)
    metrics["total_rollouts"] = float(len(all_trajectories))

    print(f"\n--- {banner} Results ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")

    return (
        metrics,
        all_trajectories,
        persona_batch,
        task_dicts,
        task_contexts,
        details_batch,
        judge_persona_policies,
        fingerprints_by_task,
    )


# ---------------------------------------------------------------------------
# _build_result — package metrics + reflection into EvaluationResult
# ---------------------------------------------------------------------------

_OE_METRICS_WHITELIST = frozenset({
    "combined_score",
    "human_likeness",
    "intra_set_diversity",
})


def _build_result(
    config: PersonaPoliciesConfig,
    evaluator,
    metrics: Dict[str, Any],
    trajectories: List,
    rollout_persona_policies: List[str],
    persona_fingerprints_by_task: Optional[List[List[BehavioralFingerprint]]] = None,
    task_dicts: Optional[List[Dict[str, Any]]] = None,
    task_contexts: Optional[List[str]] = None,
    details_batch: Optional[List[List[Dict[str, Any]]]] = None,
) -> Tuple[EvaluationResult, str]:
    # Reflection keeps the full metrics view (incl. persona_success_rate) so the
    # LLM can comment on it qualitatively.
    m = {k: v for k, v in metrics.items() if k not in ("baseline_success_rate",)}

    # OpenEvolve's prompt renders ``EvaluationResult.metrics`` into both the
    # current-program ``{metrics}`` block and every past program's performance
    # line in ``{evolution_history}``. Keep only fitness-meaningful scalars so
    # the mutator isn't buried in housekeeping (epoch, batch_in_epoch, counts).
    out: Dict[str, Any] = {}
    for k in _OE_METRICS_WHITELIST:
        v = metrics.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)

    reflection = ""
    if trajectories and rollout_persona_policies:
        try:
            n = min(len(trajectories), len(rollout_persona_policies))
            per_ep_records = metrics.get("per_episode") or []
            persona_episodes = []
            for i in range(n):
                rec = per_ep_records[i] if i < len(per_ep_records) else {}
                persona_episodes.append(
                    {
                        "persona": rollout_persona_policies[i],
                        "trajectory": trajectories[i],
                        "fingerprint": rec.get("fingerprint"),
                        "p_human": rec.get("p_human"),
                    }
                )
            task_persona_contexts: List[Dict[str, Any]] = []
            for ti, td in enumerate(task_dicts or []):
                personas: List[Dict[str, Any]] = []
                for pi, meta in enumerate((details_batch or [[]])[ti] if details_batch and ti < len(details_batch) else []):
                    personas.append({
                        "persona_idx": pi,
                        "axis_placement": meta.get("axis_placement") if isinstance(meta, dict) else None,
                        "human_likeness": meta.get("human_likeness") if isinstance(meta, dict) else None,
                    })
                task_persona_contexts.append({
                    "task_id": _task_id_for_log(td, ti) if isinstance(td, dict) else str(ti),
                    "task_context": task_contexts[ti] if task_contexts and ti < len(task_contexts) else "",
                    "original_context": td.get("user_scenario") if isinstance(td, dict) else None,
                    "personas": personas,
                })
            reflection = evaluator.llm_evolution_reflection(
                m,
                persona_episodes,
                task_persona_contexts=task_persona_contexts,
            )
        except Exception as e:
            # Reflection is advisory context for the mutator, not a fitness
            # input, so we don't hard-fail — but never swallow silently:
            # a dead reflection LLM means the mutator is flying half-blind.
            print(
                f"[REFLECTION] generation failed ({type(e).__name__}: {str(e)[:200]}); "
                f"skipping reflection this iter",
                flush=True,
            )
            reflection = ""

    artifacts: Dict[str, str] = {}
    refl = (reflection or "").strip()
    if refl:
        artifacts["evaluator_reflection"] = refl[:config.max_evolution_feedback_chars]

    return EvaluationResult(metrics=out, artifacts=artifacts), reflection


# ---------------------------------------------------------------------------
# Elite monitoring + validation
# ---------------------------------------------------------------------------
# OpenEvolve writes ``best/`` only at the very end of the run; during evolution
# the live elite code is copied into ``checkpoints/checkpoint_*/best_program.py``.

def _find_best_program_path(config: PersonaPoliciesConfig) -> Optional[Path]:
    """Current on-disk elite: final ``best/`` if present, else latest checkpoint."""
    root = Path(config.openevolve_output_dir)
    canonical = root / "best" / "best_program.py"
    if canonical.is_file():
        return canonical
    cands = sorted(
        (p for p in (root / "checkpoints").glob("checkpoint_*/best_program.py") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    return cands[-1] if cands else None


def _best_program_producing_iter(best_path: Path) -> Optional[int]:
    """Iteration that produced the current elite, from OpenEvolve's ``best_program_info.json``."""
    info = best_path.parent / "best_program_info.json"
    if not info.is_file():
        return None
    try:
        return int(json.loads(info.read_text(encoding="utf-8"))["iteration"])
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def _best_program_identity_key(best_path: Path) -> Optional[str]:
    """Stable id for the elite program (OpenEvolve program ``id``, else code SHA256)."""
    info = best_path.parent / "best_program_info.json"
    if info.is_file():
        try:
            pid = json.loads(info.read_text(encoding="utf-8")).get("id")
            if pid is not None and str(pid) != "":
                return f"id:{pid}"
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    try:
        return f"sha256:{hashlib.sha256(best_path.read_bytes()).hexdigest()}"
    except OSError:
        return None


_VAL_LOCK = threading.Lock()
_VAL_THREAD: Optional[threading.Thread] = None
_LAST_VAL_ELITE_ID: Optional[str] = None
# When a new elite appears while a full val is still in progress, stash it here; the
# val worker's ``finally`` runs the latest pending job (see _drain_val_pending).
_VAL_PENDING: Optional[Tuple[Path, int, int, int]] = None
_ELITE_ID: Optional[str] = None
_ELITE_SINCE_ITER: Optional[int] = None

# Persisted across processes so resume does not re-run full val for the same elite.
def _last_val_elite_marker(config: PersonaPoliciesConfig) -> Path:
    return Path(config.training_results_dir) / "last_val_elite_id.txt"


def _val_worker(
    best_path: Path,
    openevolve_iteration: int,
    epoch: int,
    steps_per_epoch: int,
) -> None:
    """Runs full val in background; re-enqueues a pending val if a newer elite arrived mid-run."""
    from persona_policies.evaluator import PersonaPolicyEvaluator
    from persona_policies.tau_train_context import split_train_val

    cfg = PersonaPoliciesConfig()
    cfg.ensure_output_dirs()
    _, v_ids = split_train_val(
        seed=cfg.seed,
        val_fraction=cfg.val_fraction,
        domain=cfg.taubench_domain,
        taubench_root=Path(cfg.taubench_root),
    )
    try:
        _run_full_validation_on_best(
            cfg,
            PersonaPolicyEvaluator(cfg),
            v_ids,
            best_path,
            openevolve_iteration,
            epoch,
            steps_per_epoch,
        )
    except Exception as e:
        print(f"[VAL] failed: {e}\n{traceback.format_exc()[-1200:]}", flush=True)
    finally:
        with _VAL_LOCK:
            global _VAL_THREAD
            _VAL_THREAD = None
        _drain_val_pending()


def _drain_val_pending() -> None:
    """If another elite was queued while val ran, start the latest pending full-val (non-blocking)."""
    global _VAL_THREAD, _LAST_VAL_ELITE_ID, _VAL_PENDING
    pnd: Optional[Tuple[Path, int, int, int]] = None
    with _VAL_LOCK:
        if _VAL_THREAD is not None and _VAL_THREAD.is_alive():
            return
        pnd = _VAL_PENDING
        _VAL_PENDING = None
    if pnd is None:
        return
    p, it, e, spe = pnd
    ident2 = _best_program_identity_key(p) if p.is_file() else None
    if ident2 is None:
        return
    with _VAL_LOCK:
        if _VAL_THREAD is not None and _VAL_THREAD.is_alive():
            _VAL_PENDING = pnd
            return
        if ident2 == _LAST_VAL_ELITE_ID:
            return
        _LAST_VAL_ELITE_ID = ident2
        cfg0 = PersonaPoliciesConfig()
        mk = _last_val_elite_marker(cfg0)
        mk.parent.mkdir(parents=True, exist_ok=True)
        mk.write_text(ident2, encoding="utf-8")
        _VAL_THREAD = threading.Thread(
            target=_val_worker, args=(p, it, e, spe), name=f"val-queued-{it}", daemon=True
        )
    print(
        f"[VAL] running queued full-val (elite changed during previous val) → "
        f"iter {it} best={p.parent.name if p.parent else p}",
        flush=True,
    )
    _VAL_THREAD.start()


def _handle_elite_monitoring(
    config: PersonaPoliciesConfig,
    iteration: int,
    val_ids: List[str],
    epoch: int,
    steps_per_epoch: int,
) -> bool:
    """Early-stop on stagnant elite; kick off a daemon full-val when elite id is new.

    Non-blocking: main thread never waits (no ``join()``). If a new elite appears while
    a previous val is still running, the new elite is *queued* and run when that val ends.
    """
    global _VAL_THREAD, _LAST_VAL_ELITE_ID, _ELITE_ID, _ELITE_SINCE_ITER, _VAL_PENDING
    # val_ids is unused here — worker recomputes via split_train_val to match in-training val
    _ = val_ids

    if _LAST_VAL_ELITE_ID is None:
        marker = _last_val_elite_marker(config)
        if marker.is_file():
            _LAST_VAL_ELITE_ID = marker.read_text(encoding="utf-8").strip() or None

    p = _find_best_program_path(config)
    ident = _best_program_identity_key(p) if p is not None else None
    if ident is None:
        return False

    n_stop = int(getattr(config, "early_stop_iters_without_new_best", 0) or 0)
    if n_stop > 0:
        if _ELITE_ID != ident:
            _ELITE_ID, _ELITE_SINCE_ITER = ident, iteration
        elif _ELITE_SINCE_ITER is None:
            _ELITE_SINCE_ITER = iteration
        elif iteration - _ELITE_SINCE_ITER >= n_stop:
            print(
                f"\n[EARLY-STOP] no new elite for {n_stop} iters "
                f"(since iter {_ELITE_SINCE_ITER}, now {iteration}).\n",
                flush=True,
            )
            # Signal the OpenEvolve controller (this worker's parent) to shut down
            # cleanly instead of killing just this worker (which corrupts the pool).
            try:
                _early_stop_marker(config).touch()
                os.kill(os.getppid(), signal.SIGTERM)
            except Exception:
                pass
            return True

    with _VAL_LOCK:
        if _VAL_THREAD is not None and not _VAL_THREAD.is_alive():
            # Finished thread not cleared in rare error paths: allow new val
            _VAL_THREAD = None

        if _VAL_THREAD is not None and _VAL_THREAD.is_alive():
            if ident == _LAST_VAL_ELITE_ID:
                return False
            # New on-disk best while a full val is still in flight — queue, do not drop.
            _VAL_PENDING = (p, int(iteration), int(epoch), int(steps_per_epoch))
            print(
                f"[VAL] full val still in progress; queued elite (will run when it finishes) "
                f"→ {p.parent.name if p and p.parent else p}",
                flush=True,
            )
            return False
        if ident == _LAST_VAL_ELITE_ID:
            return False
        _LAST_VAL_ELITE_ID = ident
        marker = _last_val_elite_marker(config)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(ident, encoding="utf-8")

        _VAL_THREAD = threading.Thread(
            target=_val_worker,
            args=(p, int(iteration), int(epoch), int(steps_per_epoch)),
            name=f"val-{iteration}",
            daemon=True,
        )
    _VAL_THREAD.start()
    return False


def _run_full_validation_on_best(
    config: PersonaPoliciesConfig,
    evaluator,
    val_ids: List[str],
    best_path: Path,
    iteration: int,
    epoch: int,
    steps_per_epoch: int,
) -> None:
    """Full validation on ``best_path``; append ``val_curve.jsonl`` + optional ``validation/iter_*``."""
    try:
        best_module = _load_evolved_module(str(best_path))
    except Exception as e:
        print(f"[VAL] load best: {e}", flush=True)
        return
    if best_module is None or not _module_has_generator(best_module):
        return

    n_val = len(val_ids)
    if n_val == 0:
        return

    # Producing iter is what OpenEvolve recorded in best_program_info.json;
    # fall back to iteration - 1 only if that file is missing.
    producing_iter = _best_program_producing_iter(best_path) or max(1, iteration - 1)

    val_rng = random.Random(config.seed)
    print(
        f"[VAL] full val on elite from trial {producing_iter} "
        f"({n_val} tasks × {config.n_personas} personas)…",
        flush=True,
    )
    (
        val_metrics,
        val_trajs,
        val_personas,
        val_tasks,
        val_ctxs,
        val_details,
        _,
        val_fingerprints_by_task,
    ) = _run_batch(
        best_module,
        val_ids,
        config,
        evaluator,
        val_rng,
        "val-best",
        sequential_start=0,
        batch_size=n_val,
        n_personas=config.n_personas,
        label=str(producing_iter),
    )

    if val_metrics.get("error"):
        print(
            f"[VAL] elite trial {producing_iter}: val errored — skipping log.",
            flush=True,
        )
        return
    val_score = float(val_metrics.get("combined_score", 0.0))
    row = {
        "epoch": int(epoch),
        "iteration": int(producing_iter),
        "steps_per_epoch": int(steps_per_epoch),
        "timestamp": time.time(),
        "n_val_tasks": int(n_val),
        "n_personas": int(config.n_personas),
        "best_program_path": str(best_path),
        "val_combined_score": val_score,
        "val_human_likeness": float(val_metrics.get("human_likeness", 0.0)),
        "val_intra_diversity": float(val_metrics.get("intra_set_diversity", 0.0)),
        "val_success_rate": float(val_metrics.get("persona_success_rate", 0.0)),
        "val_n_episodes": int(val_metrics.get("n_episodes", 0)),
    }
    curve_path = Path(config.training_results_dir) / "val_curve.jsonl"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    with curve_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    val_dir = Path(config.simulations_dir).parent / "validation" / f"iter_{producing_iter:04d}"
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
    _attach_diversity_axes_snapshot(val_log or {}, best_module)
    payload = {
        "epoch": epoch,
        "iteration": producing_iter,
        "steps_per_epoch": steps_per_epoch,
        "timestamp": time.time(),
        "val": val_log,
        "summary": row,
    }
    (val_dir / "log.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if val_trajs:
        val_traj_payload: Dict[str, Any] = {
            "fingerprint_vector_len": len(REGEX_FEATURES),
            "fingerprint_order_ref": "persona_policies.fingerprinting.REGEX_FEATURES",
            "val": _trajectory_json_episodes(val_trajs, val_fingerprints_by_task),
        }
        (val_dir / "trajectories.json").write_text(
            json.dumps(val_traj_payload, ensure_ascii=False, default=str), encoding="utf-8"
        )
    print(
        f"[VAL] elite trial {producing_iter}: combined={val_score:.4f} hl={row['val_human_likeness']:.3f} "
        f"intra={row['val_intra_diversity']:.3f} → {curve_path.name}",
        flush=True,
    )


def evaluate(program_path: str) -> EvaluationResult:
    """Run sliding train batch on the candidate; return ``combined_score = train_score``.

    At the start of every call we also (a) optionally early-stop when the on-disk
    elite has been unchanged for ``early_stop_iters_without_new_best`` iterations,
    and (b) fire a daemon full-val pass whenever the on-disk elite identity is new.
    Neither blocks the train batch.
    """
    global _EVAL_BATCH_SEQ, _SPLIT_LOGGED

    config = PersonaPoliciesConfig()
    config.ensure_output_dirs()

    raw = os.environ.get("OPENEVOLVE_ITERATION")
    if raw and raw.isdigit():
        iteration = int(raw)
    else:
        _EVAL_BATCH_SEQ += 1
        iteration = _EVAL_BATCH_SEQ

    if _early_stop_requested(config):
        print(
            f"[EARLY-STOP] marker present; skipping train eval for iter {iteration}",
            flush=True,
        )
        return _early_stop_result(iteration)

    try:
        module = _load_evolved_module(program_path)
    except Exception as e:
        return _error_result(f"Import error: {e}")
    if module is None:
        return _error_result("Could not load module")
    if not _module_has_generator(module):
        return _error_result(
            "No persona generator found (need generate_personas_detailed "
            "or stage1_archetypes + stage2_expand)"
        )

    from persona_policies.evaluator import PersonaPolicyEvaluator
    from persona_policies.tau_train_context import split_train_val

    train_ids, val_ids = split_train_val(
        seed=config.seed,
        val_fraction=config.val_fraction,
        domain=config.taubench_domain,
        taubench_root=Path(config.taubench_root),
    )
    if not _SPLIT_LOGGED:
        assert set(train_ids).isdisjoint(val_ids), "train/val leak"
        print(
            f"[SPLIT] seed={config.seed} val_fraction={config.val_fraction} "
            f"domain={config.taubench_domain} train={len(train_ids)} val={len(val_ids)}",
            flush=True,
        )
        _SPLIT_LOGGED = True

    rng = random.Random(
        config.seed + iteration * 7919 + (hash(program_path) % (2**31))
    )

    nb = max(1, config.eval_batch_size)
    n_train = len(train_ids)
    steps_per_epoch = max(1, math.ceil(n_train / nb))
    batch_idx_in_epoch = ((iteration - 1) % steps_per_epoch) + 1
    epoch = ((iteration - 1) // steps_per_epoch) + 1

    if _handle_elite_monitoring(config, iteration, val_ids, epoch, steps_per_epoch):
        return _early_stop_result(iteration)

    evaluator = PersonaPolicyEvaluator(config)

    train_n_personas = _train_n_personas_for_epoch(config, epoch)
    _cur = "on" if config.curriculum else "off"

    print(
        f"[EVOLUTION FITNESS] iter={iteration}  epoch={epoch}  "
        f"batch={batch_idx_in_epoch}/{steps_per_epoch}  program={Path(program_path).name}  "
        f"random train minibatch (size={nb}, pool={n_train}, "
        f"n_personas={train_n_personas}, curriculum={_cur})",
        flush=True,
    )

    # ---- 3. Training batch step ----
    (
        train_metrics,
        train_trajs,
        train_personas,
        train_tasks,
        train_ctxs,
        train_details,
        train_rollout_personas,
        train_fingerprints_by_task,
    ) = _run_batch(
        module,
        train_ids,
        config,
        evaluator,
        rng,
        "train",
        n_personas=train_n_personas,
        label=f"iter {iteration} epoch {epoch} batch {batch_idx_in_epoch}/{steps_per_epoch}",
    )
    train_score = float(train_metrics.get("combined_score", 0.0))

    if train_metrics.get("error"):
        # Failed iters are reported back to OpenEvolve via _error_result, but we
        # skip writing simulations/iter_NNNN/ so plots/curves only cover
        # successful iters (no NaN gaps, no 0.01 spikes from the error floor).
        return _error_result("Train phase failed")

    train_log = _phase_log_data(
        train_tasks, train_personas, train_metrics, train_ctxs, train_details,
    )
    _attach_diversity_axes_snapshot(train_log, module)

    # ---- 4. Build OE result (combined_score == train_score) + save per-iter log ----
    out_metrics = {k: v for k, v in train_metrics.items() if isinstance(v, (int, float))}
    out_metrics["combined_score"] = train_score
    out_metrics["train_score"] = train_score
    out_metrics["epoch"] = float(epoch)
    out_metrics["batch_in_epoch"] = float(batch_idx_in_epoch)
    out_metrics["steps_per_epoch"] = float(steps_per_epoch)

    result, refl = _build_result(
        config,
        evaluator,
        out_metrics,
        train_trajs,
        train_rollout_personas,
        train_fingerprints_by_task,
        task_dicts=train_tasks,
        task_contexts=train_ctxs,
        details_batch=train_details,
    )
    _save_iteration_log(
        config, iteration, program_path, train_log, None,
        "ok", train_score,
        train_trajectories=train_trajs,
        train_fingerprints_by_task=train_fingerprints_by_task,
        reflection_text=refl,
    )
    print(
        f"\n[TRAIN] iter={iteration} epoch={epoch} "
        f"batch {batch_idx_in_epoch}/{steps_per_epoch}  train_score={train_score:.4f}",
        flush=True,
    )

    append_train_curve_row(
        config,
        iteration=iteration,
        epoch=epoch,
        batch_in_epoch=batch_idx_in_epoch,
        steps_per_epoch=steps_per_epoch,
        program_path=program_path,
        train_metrics=train_metrics,
        train_score=train_score,
    )

    try:
        refresh_evolution_plots(config)
    except Exception as e:
        print(f"[metrics_plot] {e}", flush=True)

    return result
