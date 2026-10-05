"""Runner-agnostic OpenEvolve fitness for ppol persona generators.

Invoked by ``ppol.pipeline.PPol.evolve`` as the OpenEvolve evaluator; accepts
any ``EpisodeRunner`` via the run-config JSON (``PPOL_RUN_CONFIG`` env var).

Per iteration: take a deterministic sliding batch of train tasks, call the
evolved generator G(c, D, N) per task, run one episode per persona, then score
``combined = λ_h · human_likeness + λ_b · chamfer_coverage`` (discriminator
P(human) + two-sided Chamfer vs the human cloud; weights ramp with the
n_personas curriculum). Side-channels: LLM reflection for the mutator,
train/val curves + plots, per-iteration logs, early stop on a stagnant elite.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# OpenEvolve's structured result type; falls back to plain dict if unavailable.
try:
    from openevolve.evaluation_result import EvaluationResult  # type: ignore
except Exception:  # pragma: no cover
    EvaluationResult = None  # type: ignore[assignment]

from ppol.config import default_config


def _defaults():
    """Shared hyperparameter defaults (``PPolConfig``)."""
    return default_config()


# ---------------------------------------------------------------------------
# Module-level state (singletons + iteration counter + val thread)
# ---------------------------------------------------------------------------

_CONFIG:        Dict[str, Any] | None = None
_RUNNER                                = None
_HUMAN_DIST                            = None
_H_MATRIX:      np.ndarray | None      = None
_D_REF:         float | None           = None
_STD_MU:        np.ndarray | None      = None   # per-feature mean (human ref) for z-scoring
_STD_SD:        np.ndarray | None      = None   # per-feature std  (human ref) for z-scoring
_DISCRIMINATOR                         = None

_EVAL_SEQ      = 0                      # ppol-local iteration counter (1-based); offset from checkpoint on resume
_ELITE_ID:     Optional[str] = None
_ELITE_SINCE:  Optional[int] = None

# Run full val on the first eval always, then on any iteration whose combined
# train score clears this bar.
_VAL_SCORE_THRESHOLD = 0.7
_VAL_FIRST_DONE = False


# ---------------------------------------------------------------------------
# Config + singletons
# ---------------------------------------------------------------------------

def _config() -> Dict[str, Any]:
    global _CONFIG
    if _CONFIG is None:
        p = os.environ.get("PPOL_RUN_CONFIG")
        if not p:
            raise RuntimeError("PPOL_RUN_CONFIG env var not set; call PPol.evolve(...) to configure.")
        _CONFIG = json.loads(Path(p).read_text())
    return _CONFIG


def _load_module(path: str, name: str = "ppol_module"):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _runner():
    global _RUNNER
    if _RUNNER is None:
        cfg = _config()
        mod = _load_module(cfg["runner_module"], "ppol_user_runner")
        cls = getattr(mod, cfg["runner_class"])
        _RUNNER = cls(**(cfg.get("runner_kwargs") or {}))
    return _RUNNER


def _human_reference():
    """Load human distribution + individual fingerprint matrix H + d_ref."""
    global _HUMAN_DIST, _H_MATRIX, _D_REF
    if _HUMAN_DIST is None:
        from ppol.fingerprinting import HumanBehavioralDistribution

        cfg = _config()
        ref_path = Path(cfg["human_reference_path"])
        _HUMAN_DIST = HumanBehavioralDistribution.load(str(ref_path))

        sidecar = ref_path.with_suffix(".individual.json")
        if not sidecar.is_file():
            raise FileNotFoundError(
                f"Sidecar with individual fingerprints not found at {sidecar}. "
                "Re-run compute_human_reference() to regenerate it."
            )
        data = json.loads(sidecar.read_text())
        feature_names = _HUMAN_DIST.feature_names
        rows = [[fp.get(k, 0.0) for k in feature_names] for fp in data["fingerprints"]]
        _H_MATRIX = np.asarray(rows, dtype=np.float64)
        # Standardize features (z-score by the human reference) BEFORE clustering /
        # d_ref, so the Chamfer coverage isn't dominated by the high-variance length
        # features (opening_length alone was ~81% of raw Euclidean variance). Params
        # are stored so _chamfer_score transforms persona fingerprints identically.
        # ``standardize_coverage: false`` in the run config keeps the raw 19-D space
        # (used by examples/tau2bench); default true preserves ColBench/WildChat.
        global _STD_MU, _STD_SD
        if bool(cfg.get("standardize_coverage", True)):
            _STD_MU = _H_MATRIX.mean(axis=0)
            _STD_SD = _H_MATRIX.std(axis=0)
            _STD_SD[_STD_SD < 1e-9] = 1.0
            _H_MATRIX = (_H_MATRIX - _STD_MU) / _STD_SD
        else:
            _STD_MU = None
            _STD_SD = None
        # Optional: collapse the human cloud to K "mode" centroids for the coverage
        # target. With N~10 personas per task, two-sided Chamfer vs the full cloud
        # (~1189 pts) is dominated by unreachable humans and saturates flat; covering
        # K<=~N modes makes coverage a live, improvable signal.
        n_clusters = int(cfg.get("n_human_clusters",
                                 os.environ.get("PERSONA_POLICIES_N_HUMAN_CLUSTERS", 0)) or 0)
        if 0 < n_clusters < _H_MATRIX.shape[0]:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(_H_MATRIX)
            _H_MATRIX = km.cluster_centers_.astype(np.float64)
            print(f"[ppol fitness] coverage target: clustered {len(rows)} humans -> "
                  f"{n_clusters} mode centroids", flush=True)
        if _H_MATRIX.shape[0] >= 2:
            from scipy.spatial.distance import pdist
            d = float(pdist(_H_MATRIX, metric="euclidean").mean())
            _D_REF = d if d > 1e-9 else 1.0
        else:
            _D_REF = 1.0
        print(
            f"[ppol fitness] human reference: {_H_MATRIX.shape[0]} coverage points, d_ref={_D_REF:.4f}",
            flush=True,
        )
    return _HUMAN_DIST, _H_MATRIX, _D_REF


def _discriminator():
    global _DISCRIMINATOR
    if _DISCRIMINATOR is None:
        from ppol.discriminator import BehavioralDiscriminator
        path = _config().get("discriminator_path")
        if not path:
            raise RuntimeError(
                "discriminator_path is required in PPOL_RUN_CONFIG; pass it to PPol.evolve(...)."
            )
        _DISCRIMINATOR = BehavioralDiscriminator.load(path)
    return _DISCRIMINATOR


def _fingerprint(trajectory: List[Dict[str, str]]):
    from ppol.fingerprinting import BehavioralFingerprintExtractor
    return BehavioralFingerprintExtractor().compute_fingerprint(trajectory)


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

# On a large train pool one epoch spans many iterations, so an epoch-keyed
# curriculum never advances within a capped run. Above this pool size we step
# the schedule by iteration instead — one stage every _CURRICULUM_ITER_STAGE
# iterations (e.g. N=5 for iters 1-20, N=8 for 21-40, N=10 after).
_CURRICULUM_LARGE_TRAIN = 100
_CURRICULUM_ITER_STAGE = 20


def _n_personas_for(epoch: int, iteration: int, train_size: int) -> int:
    """Look up n_personas from the curriculum schedule.

    Schedule = ``[(epoch_threshold, n_personas), ...]`` ascending.
    - Small train pool: epoch-keyed — for epoch ``E`` the largest threshold
      ``≤ E`` wins.
    - Large train pool (> _CURRICULUM_LARGE_TRAIN): iteration-keyed — step
      through the schedule's N values, one stage per _CURRICULUM_ITER_STAGE
      iterations, holding the final N afterwards.
    ``epoch`` and ``iteration`` are 1-based.
    """
    cfg = _config()
    d = _defaults()
    base = max(1, int(cfg.get("n_personas", d.n_personas)))
    if not cfg.get("curriculum", d.curriculum):
        return base
    schedule = sorted(
        cfg.get("n_personas_schedule") or [list(t) for t in d.n_personas_schedule],
        key=lambda t: int(t[0]),
    )
    if not schedule:
        return base
    if train_size > _CURRICULUM_LARGE_TRAIN:
        ns = [max(1, int(n_p)) for _, n_p in schedule]
        stage = min((max(1, iteration) - 1) // _CURRICULUM_ITER_STAGE, len(ns) - 1)
        return ns[stage]
    n = base
    for thresh, n_p in schedule:
        if int(epoch) >= int(thresh):
            n = max(1, int(n_p))
    return n


def _scoring_weights(n_personas_current: int) -> Tuple[float, float]:
    """Curriculum-aware train weights: ramp lambda_intra with current N."""
    cfg = _config()
    d = _defaults()
    n_final   = max(1, int(cfg.get("n_personas", d.n_personas)))
    n_current = max(1, int(n_personas_current))
    ratio = min(1.0, n_current / n_final)
    lambda_b = float(cfg.get("lambda_intra_diversity", d.lambda_intra_diversity)) * ratio
    lambda_h = 1.0 - lambda_b
    return lambda_h, lambda_b


# ---------------------------------------------------------------------------
# Sliding train batch
# ---------------------------------------------------------------------------

def _sliding_batch(train_tasks: List, iteration: int, batch_size: int) -> Tuple[List, int, int, int]:
    """Deterministic sliding window across train tasks.

    Returns ``(batch, epoch, batch_in_epoch, steps_per_epoch)``.

    Iteration is 1-based. One epoch = full sweep over ``train_tasks``.
    ``epoch`` and ``batch_in_epoch`` are **1-based**.
    """
    n = len(train_tasks)
    if n == 0:
        return [], 0, 0, 0
    bs = max(1, min(int(batch_size), n))
    steps_per_epoch = (n + bs - 1) // bs
    batch_in_epoch = ((iteration - 1) % steps_per_epoch) + 1
    epoch = ((iteration - 1) // steps_per_epoch) + 1
    start = (batch_in_epoch - 1) * bs
    end = min(start + bs, n)
    batch = train_tasks[start:end]
    if len(batch) < bs:
        batch = batch + train_tasks[: bs - len(batch)]
    return batch, epoch, batch_in_epoch, steps_per_epoch


# ---------------------------------------------------------------------------
# Persona generation + episode execution
# ---------------------------------------------------------------------------

def _generate_personas(generator_mod, task, n: int) -> List[Dict[str, Any]]:
    """Return a list of persona dicts with at least ``expanded_instruction`` set."""
    ctx = task.description
    if getattr(task, "context", None):
        ctx = f"{task.description}\n\nContext: {task.context}"

    axes = getattr(generator_mod, "DIVERSITY_AXES", None)
    gd = getattr(generator_mod, "generate_personas_detailed", None)
    if not (callable(gd) and axes is not None):
        raise RuntimeError(
            "Generator must expose DIVERSITY_AXES and generate_personas_detailed(c, axes, n)."
        )
    rows = list(gd(ctx, axes, n))
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = (
            row.get("expanded_instruction")
            or row.get("instruction")
            or row.get("text")
            or ""
        ).strip()
        if not text:
            continue
        out.append({
            "persona_id":          row.get("persona_id"),
            "description":         row.get("description"),
            "axis_placement":      dict(row.get("axis_placement") or {}),
            "reasoning":           row.get("reasoning"),
            "expanded_instruction": text,
        })
    return out


def _run_episode(task, persona_text: str, phase: str = "train"):
    try:
        # Optional train-only turn cap (``train_max_turns`` in the run config),
        # for runners whose ``run_episode`` accepts ``max_turns`` (e.g. τ²-bench).
        cap = _config().get("train_max_turns") if phase == "train" else None
        if cap:
            result = _runner().run_episode(
                task, persona_policy=persona_text, max_turns=int(cap)
            )
        else:
            result = _runner().run_episode(task, persona_policy=persona_text)
        return result, _fingerprint(result.trajectory)
    except Exception as e:
        sys.stderr.write(f"[ppol fitness] episode failed: {e}\n")
        return None, None


# ---------------------------------------------------------------------------
# Scoring components
# ---------------------------------------------------------------------------

def _chamfer_score(fingerprints, H: np.ndarray, d_ref: float, feature_names) -> float:
    """Two-sided Chamfer in 19-D regex space, rescaled to ``[0, 1]``.

    Wrapper over :func:`ppol.evaluation.chamfer_coverage` (the single shared
    implementation). ``H`` is z-scored; ``P`` is z-scored to match via the module
    standardizer ``_STD_MU``/``_STD_SD`` set in :func:`_human_reference`.
    """
    from ppol.evaluation import chamfer_coverage, fingerprints_to_matrix

    P = fingerprints_to_matrix(fingerprints, feature_names)
    return chamfer_coverage(P, H, d_ref, std_mu=_STD_MU, std_sd=_STD_SD)


def _trajectory_excerpt(traj: List[Dict[str, str]], *, max_turns: int = 20, max_chars: int = 2400) -> str:
    lines: List[str] = []
    for t in traj[:max_turns]:
        role = str(t.get("role", "")).upper()
        c = str(t.get("content") or "")
        if not c.strip():
            continue
        if role == "ASSISTANT" and len(c) > 250:
            c = f"{c[:100]} ... {c[-100:]}"
        lines.append(f"{role}: {c}")
    return "\n".join(lines)[:max_chars]


def _format_fingerprint_block(fp) -> str:
    if fp is None or not hasattr(fp, "features"):
        return ""
    feats = fp.features
    names = sorted(feats.keys())
    parts = [f"{n}={float(feats.get(n, 0.0)):.3f}" for n in names]
    return "\n".join("  " + "  ".join(parts[i:i + 5]) for i in range(0, len(parts), 5))


# ---------------------------------------------------------------------------
# LLM reflection
# ---------------------------------------------------------------------------

_REFLECTION_PROMPT = """You are evaluating a set of personas representing human populations in provided task scenarios.

Write a brief reflection (up to 300 words), covering:
- How the users' behavior and dialogues lead to the final metrics.
- Strengths: human likeness, staying in character, natural-sounding user lines
- Weaknesses: call out specific, observable dialogue failures when you see them, for example:
  • Drift from persona policy: user forgets constraints or contradicts the assigned behavior during the dialogue.
  • Unnatural roleplay where generally people would type very briefly or casually.
  • Overly cooperative behavior lacking any realistic friction—no typos or natural pushback when appropriate.
- Use the human likeness probability and other features to explain *why* personas scored high or low given the task.
- Analyze which combination of behaviors among personas lead to higher human-likeness and which combinations conflict.
- Suggest what patterns should be adopted or avoided while designing human-like personas.

Output rules (must follow):
- You must NEVER mention indices or labels ("Task K", "Sample N", "p0/p1"). Describe patterns instead.
- Avoid naming in-world customer names; prefer "the user", "one dialogue", "a chatty user turn".

---
# Metrics
{metrics_block}

---
# This batch of task scenarios:
{task_context_block}

---
# Sample personas and dialogues (highest and lowest human-likeness)
{pairs_block}
"""


def _generate_reflection(
    metrics: Dict[str, Any],
    persona_episodes: List[Dict[str, Any]],
    task_persona_contexts: List[Dict[str, Any]],
) -> str:
    """Call the reflection LLM to produce mutator feedback. Returns "" on failure."""
    cfg = _config()
    d = _defaults()
    model = cfg.get("reflection_model") or d.evolution_feedback_model
    if not model:
        return ""

    # Pick top-2 / bottom-2 by p_human, fallback to random if no scores
    pool = []
    for ep in persona_episodes:
        pp = str(ep.get("persona") or "").strip()
        traj = ep.get("trajectory")
        if not pp or not traj:
            continue
        excerpt = _trajectory_excerpt(traj).strip()
        if not excerpt:
            continue
        pool.append({
            "persona": pp,
            "excerpt": excerpt,
            "fingerprint": ep.get("fingerprint"),
            "p_human": ep.get("p_human"),
        })
    if not pool:
        return ""
    scored = [x for x in pool if isinstance(x.get("p_human"), (int, float))]
    if scored:
        scored.sort(key=lambda x: float(x["p_human"]))
        sampled = (list(reversed(scored[-2:])) + scored[:2])[:4]
    else:
        sampled = pool[:4]

    metrics_block = "\n".join(
        f"- {k}: {metrics[k]:.4f}"
        for k in ("combined_score", "human_likeness", "intra_set_diversity")
        if isinstance(metrics.get(k), (int, float))
    )

    pair_sections = []
    for ep in sampled:
        persona_clip = ep["persona"][:1200] + ("\n[...]" if len(ep["persona"]) > 1200 else "")
        p_h = ep.get("p_human")
        p_h_str = f"{float(p_h):.4f}" if isinstance(p_h, (int, float)) else "n/a"
        fp_block = _format_fingerprint_block(ep.get("fingerprint"))
        fp_section = f"Behavioral features:\n{fp_block}\n\n" if fp_block else ""
        pair_sections.append(
            f"## Sample (human_likeness={p_h_str})\n"
            f"Persona policy:\n{persona_clip}\n\n"
            f"{fp_section}"
            f"Dialogue:\n{ep['excerpt']}"
        )
    pairs_block = "\n\n".join(pair_sections)

    task_sections = []
    for ti, task in enumerate(task_persona_contexts, start=1):
        ctx = str(task.get("task_context") or "")[:2200]
        persona_lines = []
        for p in task.get("personas") or []:
            if not isinstance(p, dict):
                continue
            h = p.get("human_likeness")
            h_s = f"{float(h):.4f}" if isinstance(h, (int, float)) else "n/a"
            axis = p.get("axis_placement")
            behaviors = [k for k, v in axis.items() if bool(v)] if isinstance(axis, dict) else []
            persona_lines.append(
                f"  - human_likeness={h_s}; behaviors: {json.dumps(behaviors, ensure_ascii=False)}"
            )
        task_sections.append(
            f"## Task {ti}\nOriginal context: {ctx}\nGenerated Personas:\n" + "\n".join(persona_lines)
        )
    task_context_block = "\n\n".join(task_sections)

    prompt = _REFLECTION_PROMPT.format(
        metrics_block=metrics_block,
        task_context_block=task_context_block,
        pairs_block=pairs_block,
    )
    try:
        import litellm
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
            max_tokens=int(cfg.get("evolution_feedback_max_tokens", d.evolution_feedback_max_tokens)),
            timeout=120,
        )
        return ((resp.choices[0].message.content or "")).strip()
    except Exception as e:
        sys.stderr.write(f"[ppol fitness] reflection failed: {e}\n")
        return ""


# ---------------------------------------------------------------------------
# Train / val curves and plot refresh
# ---------------------------------------------------------------------------

def _training_dir() -> Path:
    out = Path(_config().get("openevolve_output_dir") or "outputs/openevolve")
    return out.parent / "training"


def _append_train_curve(
    iteration: int, epoch: int, batch_in_epoch: int, steps_per_epoch: int,
    program_path: str, metrics: Dict[str, Any], train_score: float,
) -> None:
    row = {
        "iteration":           int(iteration),
        "epoch":               int(epoch),
        "batch_in_epoch":      int(batch_in_epoch),
        "steps_per_epoch":     int(steps_per_epoch),
        "program":             Path(program_path).name,
        "final_combined_score": float(train_score),
        "train_combined":      float(metrics.get("combined_score", train_score)),
        "train_human_likeness": float(metrics.get("human_likeness", float("nan"))),
        "train_intra_diversity": float(metrics.get("intra_set_diversity", float("nan"))),
    }
    out = _training_dir() / "results" / "train_curve.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _append_val_curve(
    producing_iter: int, epoch: int, steps_per_epoch: int,
    best_program_path: Path, val_metrics: Dict[str, Any], n_val_tasks: int, n_personas: int,
) -> None:
    row = {
        "epoch":               int(epoch),
        "iteration":           int(producing_iter),
        "steps_per_epoch":     int(steps_per_epoch),
        "timestamp":           time.time(),
        "n_val_tasks":         int(n_val_tasks),
        "n_personas":          int(n_personas),
        "best_program_path":   str(best_program_path),
        "val_combined_score":  float(val_metrics.get("combined_score", 0.0)),
        "val_human_likeness":  float(val_metrics.get("human_likeness", 0.0)),
        "val_intra_diversity": float(val_metrics.get("intra_set_diversity", 0.0)),
        "val_n_episodes":      int(val_metrics.get("n_episodes", 0)),
    }
    out = _training_dir() / "results" / "val_curve.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        f.flush()


def _refresh_plots() -> None:
    """Regenerate ``evolution_scores.png`` from the JSONL curves. Best-effort."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        from ppol.analysis.plot_evolution_scores import plot_scores
    except Exception as e:
        print(f"[ppol fitness] plot refresh skipped: {e}", flush=True)
        return

    results_dir = _training_dir() / "results"
    train_path = results_dir / "train_curve.jsonl"
    val_path   = results_dir / "val_curve.jsonl"

    def _load_jsonl(p: Path) -> List[Dict[str, Any]]:
        if not p.is_file():
            return []
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        rows.sort(key=lambda r: int(r.get("iteration", 0)))
        return rows

    train_rows = _load_jsonl(train_path)
    val_rows   = _load_jsonl(val_path)
    if not train_rows and not val_rows:
        return
    out_png = results_dir / "evolution_scores.png"
    try:
        plot_scores(
            train_rows, val_rows, out_png,
            title="Evolution: scores vs iteration",
            verbose=False, contiguous_x=False,
        )
    except Exception as e:
        print(f"[ppol fitness] plot_scores failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# Per-iteration log dumps
# ---------------------------------------------------------------------------

def _save_iteration_log(
    phase: str, iteration: int, epoch: int, batch_in_epoch: int, steps_per_epoch: int,
    program_path: str, metrics: Dict[str, Any], persona_batch: List[List[Dict[str, Any]]],
    task_contexts: List[str], details_batch: List[List[Dict[str, Any]]],
    reflection_text: str = "",
) -> None:
    log_dir = _training_dir() / "simulations" / f"iter_{iteration:04d}"
    log_dir.mkdir(parents=True, exist_ok=True)

    light_metrics = {
        k: v for k, v in metrics.items()
        if k in ("combined_score", "human_likeness", "intra_set_diversity",
                 "persona_success_rate", "n_episodes", "n_tasks_in_batch",
                 "n_personas_per_task")
    }
    payload = {
        "phase": phase,
        "iteration": iteration,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "steps_per_epoch": steps_per_epoch,
        "program": Path(program_path).name,
        "timestamp": time.time(),
        "metrics": light_metrics,
        "tasks": [
            {
                "task_context": ctx,
                "personas": personas,
            }
            for ctx, personas in zip(task_contexts, details_batch)
        ],
    }
    (log_dir / "log.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    if reflection_text:
        (log_dir / "reflection.txt").write_text(reflection_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Best program tracking + val worker
# ---------------------------------------------------------------------------

def _find_best_program() -> Optional[Path]:
    out_dir = Path(_config().get("openevolve_output_dir") or "")
    if not out_dir:
        return None
    p = out_dir / "best" / "best_program.py"
    return p if p.is_file() else None


def _best_program_id(p: Path) -> Optional[str]:
    try:
        return hashlib.sha1(p.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def _run_batch(generator_mod, tasks_subset: List, phase: str, n_personas: int, n_workers: int):
    """Generate personas, run episodes, return per-task fingerprints + episode metadata.

    Shared by train + val paths.
    """
    per_task_fps: List[List] = [[] for _ in tasks_subset]
    per_task_personas: List[List[Dict[str, Any]]] = [[] for _ in tasks_subset]
    per_task_results: List[List[Any]] = [[] for _ in tasks_subset]

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = []
        for ti, task in enumerate(tasks_subset):
            try:
                personas = _generate_personas(generator_mod, task, n_personas)
            except Exception as e:
                sys.stderr.write(f"[ppol fitness/{phase}] gen failed for {task.task_id}: {e}\n")
                continue
            for p in personas:
                per_task_personas[ti].append(p)
                futures.append((ti, p, ex.submit(_run_episode, task, p["expanded_instruction"], phase)))
        for ti, p, f in futures:
            result, fp = f.result()
            if result is None or fp is None:
                continue
            per_task_fps[ti].append(fp)
            per_task_results[ti].append((result, fp, p))

    return per_task_fps, per_task_personas, per_task_results


def _run_val(program_path: Path, producing_iter: int, epoch: int, steps_per_epoch: int) -> None:
    """Run full val on the given candidate and append a val_curve point.

    Runs synchronously (inline in evaluate()): OpenEvolve executes evaluate() in a
    forked worker process, so a background thread here would be killed when the
    pool shuts down before its episodes finish — and its LLM load would starve the
    main loop. Blocking guarantees the point is written and keeps val off the
    critical LLM-bandwidth path."""
    cfg = _config()
    d = _defaults()
    try:
        gen_mod = _load_module(str(program_path), "ppol_val_candidate")
        runner = _runner()
        val_ids = set(cfg.get("val_task_ids") or [])
        val_tasks = [t for t in runner.get_tasks() if t.task_id in val_ids]
        if not val_tasks:
            return
        n_personas = max(1, int(cfg.get("n_personas", d.n_personas)))
        n_workers  = max(1, int(cfg.get("n_workers", d.parallel_episode_workers)))

        print(
            f"[VAL] full val on candidate from iter {producing_iter} "
            f"({len(val_tasks)} tasks × {n_personas} personas)...",
            flush=True,
        )
        per_task_fps, _, _ = _run_batch(gen_mod, val_tasks, "val", n_personas, n_workers)
        all_fps = [fp for grp in per_task_fps for fp in grp]
        if not all_fps:
            print(f"[VAL] iter {producing_iter}: no episodes — skipping log.", flush=True)
            return

        disc = _discriminator()
        _, H, d_ref = _human_reference()
        feature_names = _human_reference()[0].feature_names

        human_likeness = float(np.mean([disc.predict_human_probability(fp) for fp in all_fps]))
        chamfers = [
            _chamfer_score(grp, H, d_ref, feature_names) for grp in per_task_fps if grp
        ]
        intra = float(np.mean(chamfers)) if chamfers else 0.0
        lh = float(cfg.get("lambda_human_likeness", d.lambda_human_likeness))
        lb = float(cfg.get("lambda_intra_diversity", d.lambda_intra_diversity))
        val_metrics = {
            "combined_score":      lh * human_likeness + lb * intra,
            "human_likeness":      human_likeness,
            "intra_set_diversity": intra,
            "n_episodes":          len(all_fps),
        }
        _append_val_curve(producing_iter, epoch, steps_per_epoch,
                          program_path, val_metrics, len(val_tasks), n_personas)
        _refresh_plots()
        print(
            f"[VAL] iter {producing_iter}: combined={val_metrics['combined_score']:.4f} "
            f"hl={human_likeness:.3f} intra={intra:.3f}",
            flush=True,
        )
    except Exception as e:
        print(f"[VAL] failed: {e}\n{traceback.format_exc()[-1200:]}", flush=True)


def _maybe_run_val(
    iteration: int, epoch: int, steps_per_epoch: int,
    program_path: str, train_score: float,
) -> None:
    """Run full val on the FIRST eval ever (to anchor the val curve), and thereafter
    whenever the combined train score clears ``val_score_threshold`` (default 0.7).

    "First" is tracked via a marker FILE, not in-memory: OpenEvolve forks workers, so
    an in-memory flag resets per worker and fires a low-score val on each worker's
    first eval. The file makes it a true one-time, cross-worker, resume-safe anchor."""
    threshold = float(_config().get("val_score_threshold", _VAL_SCORE_THRESHOLD))
    marker = Path(_config().get("openevolve_output_dir", "")) / ".val_first_done"
    first = not marker.exists()
    if not first and float(train_score) <= threshold:
        return
    if first:
        try:
            marker.touch()
        except OSError:
            pass
    p = Path(program_path)
    if not p.is_file():
        print(f"[VAL] skip iter {iteration}: program not found at {p}", flush=True)
        return
    print(f"[VAL] triggered iter {iteration}: train={train_score:.4f} "
          f"({'first-anchor' if first else f'> {threshold:.2f}'})", flush=True)
    _run_val(p, iteration, epoch, steps_per_epoch)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

def _check_early_stop(iteration: int) -> bool:
    """Return True if we've triggered early stop. Best-effort SIGTERM to parent."""
    global _ELITE_ID, _ELITE_SINCE
    cfg = _config()
    d = _defaults()
    n_stop = int(cfg.get("early_stop_iters_without_new_best", d.early_stop_iters_without_new_best) or 0)
    if n_stop <= 0:
        return False
    p = _find_best_program()
    if p is None:
        return False
    ident = _best_program_id(p)
    if ident is None:
        return False
    if _ELITE_ID != ident:
        _ELITE_ID, _ELITE_SINCE = ident, iteration
        return False
    if _ELITE_SINCE is None:
        _ELITE_SINCE = iteration
        return False
    if iteration - _ELITE_SINCE < n_stop:
        return False
    print(
        f"\n[EARLY-STOP] no new elite for {n_stop} iters "
        f"(since iter {_ELITE_SINCE}, now {iteration}).\n",
        flush=True,
    )
    out_dir = Path(cfg.get("openevolve_output_dir") or "")
    try:
        (out_dir / "EARLY_STOP").touch()
        os.kill(os.getppid(), signal.SIGTERM)
    except Exception:
        pass
    return True


# ---------------------------------------------------------------------------
# OpenEvolve entry point
# ---------------------------------------------------------------------------

def _eval_seq_offset() -> int:
    """Return the number of evaluations already completed, so the ppol iteration
    label + sliding batch + curriculum resume from the real eval count.

    We count points in ``train_curve.jsonl`` (one per successful evaluate()),
    NOT OpenEvolve's checkpoint ``last_iteration`` — that counter also ticks on
    failed mutations that never reached evaluate(), so it runs far ahead of the
    real evaluation count (e.g. 70 iterations for ~20 evaluations) and made the
    resumed labels jump to 71+."""
    global _EVAL_SEQ
    if _EVAL_SEQ != 0:
        return 0  # already initialised
    try:
        tc = _training_dir() / "results" / "train_curve.jsonl"
        if tc.is_file():
            its = [json.loads(l).get("iteration", 0)
                   for l in tc.read_text().splitlines() if l.strip()]
            return max(its) if its else 0
    except Exception:
        pass
    return 0


def evaluate(program_path: str):
    """OpenEvolve calls this once per candidate. Returns EvaluationResult or dict."""
    global _EVAL_SEQ
    if _EVAL_SEQ == 0:
        _EVAL_SEQ = _eval_seq_offset()
    _EVAL_SEQ += 1
    iteration = _EVAL_SEQ

    cfg = _config()
    d = _defaults()

    # Load the candidate generator
    try:
        gen_mod = _load_module(program_path, "evolved_generator")
    except Exception as e:
        sys.stderr.write(f"[ppol fitness] failed to load generator: {e}\n{traceback.format_exc()}\n")
        return _wrap_result({"combined_score": 0.0, "human_likeness": 0.0, "intra_set_diversity": 0.0}, "")

    # Determine train batch (sliding window) + epoch + curriculum N
    runner = _runner()
    task_ids = list(cfg["train_task_ids"])
    all_tasks_by_id = {t.task_id: t for t in runner.get_tasks()}
    train_tasks = [all_tasks_by_id[i] for i in task_ids if i in all_tasks_by_id]
    if not train_tasks:
        sys.stderr.write("[ppol fitness] no train tasks matched config.train_task_ids\n")
        return _wrap_result({"combined_score": 0.0, "human_likeness": 0.0, "intra_set_diversity": 0.0}, "")

    batch_size = max(1, int(cfg.get("batch_size", d.eval_batch_size)))
    batch, epoch, batch_in_epoch, steps_per_epoch = _sliding_batch(train_tasks, iteration, batch_size)
    n_personas = _n_personas_for(epoch, iteration, len(train_tasks))
    n_workers  = max(1, int(cfg.get("n_workers", d.parallel_episode_workers)))
    lambda_h, lambda_b = _scoring_weights(n_personas)

    print(
        f"\n[ppol fitness] iter={iteration} epoch={epoch} batch={batch_in_epoch}/{steps_per_epoch} "
        f"n_personas={n_personas} weights=(h={lambda_h:.2f}, b={lambda_b:.2f})",
        flush=True,
    )

    # Run the batch
    per_task_fps, per_task_personas, per_task_results = _run_batch(
        gen_mod, batch, "train", n_personas, n_workers
    )
    all_fps = [fp for grp in per_task_fps for fp in grp]
    if not all_fps:
        return _wrap_result({"combined_score": 0.0, "human_likeness": 0.0, "intra_set_diversity": 0.0}, "")

    # Score
    disc = _discriminator()
    p_humans = [disc.predict_human_probability(fp) for fp in all_fps]
    human_likeness = float(np.mean(p_humans))

    human_dist, H, d_ref = _human_reference()
    feature_names = human_dist.feature_names
    chamfers = [_chamfer_score(grp, H, d_ref, feature_names) for grp in per_task_fps if grp]
    intra_set_diversity = float(np.mean(chamfers)) if chamfers else 0.0

    combined = lambda_h * human_likeness + lambda_b * intra_set_diversity

    metrics: Dict[str, Any] = {
        "combined_score":            float(combined),
        "human_likeness":            float(human_likeness),
        "intra_set_diversity":       float(intra_set_diversity),
        "n_episodes":                float(len(all_fps)),
        "n_tasks_in_batch":          float(len(batch)),
        "n_personas_per_task":       float(n_personas),
        "epoch":                     float(epoch),
        "batch_in_epoch":            float(batch_in_epoch),
        "steps_per_epoch":           float(steps_per_epoch),
        "score_weight_human_likeness": float(lambda_h),
        "score_weight_intra_diversity": float(lambda_b),
    }

    # Per-episode detail + p_human + persona context for reflection / logging
    persona_episodes: List[Dict[str, Any]] = []
    task_persona_contexts: List[Dict[str, Any]] = []
    flat_p_idx = 0
    for ti, task in enumerate(batch):
        ctx = task.description
        if getattr(task, "context", None):
            ctx = f"{task.description}\n\nContext: {task.context}"
        per_persona = []
        for j, (result, fp, persona_meta) in enumerate(per_task_results[ti]):
            p_h = p_humans[flat_p_idx] if flat_p_idx < len(p_humans) else None
            flat_p_idx += 1
            persona_episodes.append({
                "persona":     persona_meta.get("expanded_instruction"),
                "trajectory":  result.trajectory,
                "fingerprint": fp,
                "p_human":     p_h,
            })
            per_persona.append({
                **persona_meta,
                "human_likeness": p_h,
            })
        task_persona_contexts.append({"task_context": ctx, "personas": per_persona})

    # Reflection (LLM call → mutator artifact)
    reflection = _generate_reflection(metrics, persona_episodes, task_persona_contexts)

    # Train curve + per-iter log + plot refresh
    _append_train_curve(iteration, epoch, batch_in_epoch, steps_per_epoch,
                        program_path, metrics, combined)
    _save_iteration_log(
        "train", iteration, epoch, batch_in_epoch, steps_per_epoch,
        program_path, metrics, per_task_personas,
        [t.description for t in batch],
        [
            [{**pm, "human_likeness": pe.get("human_likeness")}
             for pm, pe in zip(per_task_personas[ti], task_persona_contexts[ti].get("personas", []))]
            for ti in range(len(batch))
        ],
        reflection_text=reflection,
    )
    _refresh_plots()

    # Val the current candidate whenever its combined score clears the bar (synchronous)
    _maybe_run_val(iteration, epoch, steps_per_epoch, program_path, combined)
    _check_early_stop(iteration)

    return _wrap_result(metrics, reflection)


def _wrap_result(metrics: Dict[str, Any], reflection: str):
    """Wrap metrics + reflection into an EvaluationResult (or plain dict fallback)."""
    if EvaluationResult is None:
        return metrics
    cfg = _config() if _CONFIG is not None else {}
    d = _defaults()
    max_chars = int(cfg.get("max_evolution_feedback_chars", d.max_evolution_feedback_chars))
    artifacts: Dict[str, str] = {}
    refl = (reflection or "").strip()
    if refl:
        artifacts["evaluator_reflection"] = refl[:max_chars]
    return EvaluationResult(metrics=metrics, artifacts=artifacts)
