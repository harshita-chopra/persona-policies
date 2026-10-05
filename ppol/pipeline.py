"""
Python API for the ppol pipeline.

Typical order:
  1. compute_human_reference()  — fingerprint your human traces
  2. collect_baseline()         — run episodes without persona
  3. train_discriminator()      — train RF on human vs baseline fingerprints
  4. PPol(output_dir=...).evolve(...)   — evolve the persona generator G(c, D, N)
  5. PPol.generate() / benchmark_policy() / ppol.evaluation.evaluate_simulators()
     — generate personas and score on held-out tasks
"""

from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from ppol.config import DEFAULT_EVOLVE_ITERATIONS, default_config
from ppol.core.runner import EpisodeRunner
from ppol.core.types import EpisodeResult, Task


def split_tasks(
    tasks: Sequence[Task],
    *,
    test_fraction: Optional[float] = 0.2,
    val_fraction: Optional[float] = None,
    seed: Optional[int] = None,
    # Back-compat aliases (deprecated; prefer test_fraction / val_fraction).
    test_size: Optional[float] = None,
    val_size: Optional[float] = None,
) -> Tuple[List[Task], List[Task], List[Task]]:
    """Two-step split: first hold out test, then carve val out of train.

    1. ``test_fraction`` is held out of the full set for test.
    2. ``val_fraction`` is carved out of the *remaining* train pool for validation.

    Defaults mirror ``PPolConfig`` (``val_fraction=0.2``, ``seed=42``).
    With test_fraction=0.2 and val_fraction=0.2 the final ratios are
    ``0.64 train / 0.16 val / 0.20 test``.

    Args:
        tasks:          Any sequence of Task objects.
        test_fraction:  Fraction of full set held out for test (default 0.20).
        val_fraction:   Fraction of the train pool carved out for val
                        (default ``PPolConfig.val_fraction``).
        seed:           Random seed (default ``PPolConfig.seed``).

    Returns:
        (train_tasks, val_tasks, test_tasks)
    """
    cfg = default_config()
    if test_size is not None:
        test_fraction = test_size
    if val_size is not None:
        val_fraction = val_size
    if test_fraction is None:
        test_fraction = 0.2
    if val_fraction is None:
        val_fraction = cfg.val_fraction
    if seed is None:
        seed = cfg.seed

    shuffled = list(tasks)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_test = int(n * test_fraction)
    test_tasks = shuffled[:n_test]
    rest       = shuffled[n_test:]
    n_val = int(len(rest) * val_fraction)
    val_tasks   = rest[:n_val]
    train_tasks = rest[n_val:]
    return train_tasks, val_tasks, test_tasks


def _row_to_persona(row: Any) -> Tuple[str, Dict]:
    """Normalize a generator output row to (text, meta)."""
    if not isinstance(row, dict):
        t = str(row).strip()
        return t, {}
    exp = str(row.get("expanded_instruction") or row.get("stage2_expanded_instruction") or "").strip()
    ap = row.get("axis_placement")
    return exp, {
        "persona_id": row.get("persona_id", ""),
        "description": row.get("description", ""),
        "axis_placement": dict(ap) if isinstance(ap, dict) else (ap or {}),
        "reasoning": row.get("reasoning") or row.get("expanded_reasoning", ""),
    }


def _call_generator(module: Any, ctx: str, n: int, axes: Any) -> list:
    """Call G(c, axes, n) without importing fitness.py (which pulls in openevolve)."""
    gd = getattr(module, "generate_personas_detailed", None)
    if callable(gd) and axes is not None:
        rows = list(gd(ctx, axes, n))
    else:
        s1 = getattr(module, "stage1_archetypes", None)
        s2 = getattr(module, "stage2_expand", None)
        if not (callable(s1) and callable(s2) and axes is not None):
            raise RuntimeError(
                "Evolved module must expose generate_personas_detailed(c, axes, n) "
                "or stage1_archetypes + stage2_expand."
            )
        rows = [
            {**arch, "expanded_instruction": s2(arch, ctx, axes)}
            for arch in s1(ctx, axes, n)
            if isinstance(arch, dict)
        ]
    result = []
    for row in rows:
        text, meta = _row_to_persona(row)
        result.append({"text": text, **meta})
    return result


def compute_human_reference(
    dialogs: List[dict],
    output_path: str,
    *,
    domain: str = "custom",
    source: str = "custom",
) -> str:
    """
    Extract behavioral fingerprints from human reference dialogs and save the
    distribution to a JSON file.

    The output file is what the discriminator and evaluator read as the
    human behavioral reference. For τ²-bench specifically, use
    ``examples/tau2bench/tau_reference.py:compute_tau2bench_human_reference``
    which handles dialogue loading + domain filtering.

    Args:
        dialogs:     List of dialogs in ppol format — each ``{"conversation": [{role, content}, ...], ...}``.
        output_path: Where to save the distribution (e.g. ``"outputs/reference_data/human_fingerprints_custom.json"``).
        domain:      Label stored in the distribution (used for mismatch checks).
        source:      Label for the data source.

    Returns:
        The resolved output path.

    Example::

        from ppol import DataLoader
        from ppol.pipeline import compute_human_reference

        dialogs = DataLoader.load_dialogs("my_human_dialogs.json")
        compute_human_reference(dialogs, "outputs/reference_data/human_fingerprints_custom.json")
    """
    from ppol.fingerprinting import (
        BehavioralFingerprintExtractor,
        compute_human_distribution,
    )

    if not dialogs:
        raise ValueError("dialogs list is empty")

    extractor = BehavioralFingerprintExtractor()
    human_dist = compute_human_distribution(
        dialogs, extractor, source=source, domain=domain
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    human_dist.save(output_path)

    # Also save individual per-dialog fingerprints as a sidecar — required by
    # the evolution fitness for Chamfer behavioral coverage.
    individual = []
    for d in dialogs:
        trace = d.get("conversation") or d.get("turns") or []
        if len(trace) < 2:
            continue
        fp = extractor.compute_fingerprint(trace)
        individual.append(fp.features)
    sidecar = Path(output_path).with_suffix(".individual.json")
    sidecar.write_text(json.dumps({"fingerprints": individual}))

    print(f"Saved human reference → {output_path}")
    print(f"                       {sidecar} ({len(individual)} per-dialog fingerprints)")
    print(f"  Dialogues:   {human_dist.n_dialogues}")
    print(f"  Features:    {len(human_dist.feature_names)}")
    return output_path


def collect_baseline(
    runner: EpisodeRunner,
    tasks: List[Task],
    output_path: str,
    *,
    n_workers: Optional[int] = None,
    verbose: bool = True,
) -> List[dict]:
    """
    Run episodes WITHOUT a persona for each task and save the trajectories and
    fingerprints to disk.

    The output is a single JSON file with ``meta`` and ``episodes`` (each episode
    embeds its trajectory and fingerprint). This file feeds ``train_discriminator()``.

    Args:
        runner:      Any ``EpisodeRunner`` — your custom runner or ``Tau2BenchRunner``.
        tasks:       List of ``Task`` objects from ``runner.get_tasks()`` (or your own).
        output_path: Where to save the baseline file (e.g. ``"outputs/reference_data/baseline.json"``).
        n_workers:   Parallel episode workers (default
                     ``PPolConfig.parallel_episode_workers``).

    Returns:
        The list of fingerprint row dicts (also written to disk).

    Example::

        # See examples/tau2bench/runner.py for Tau2BenchRunner
        # (examples/ is not a Python package; add it to sys.path or run from there).
        from ppol.pipeline import collect_baseline

        runner = Tau2BenchRunner()
        tasks  = runner.get_tasks()
        collect_baseline(runner, tasks, "outputs/reference_data/baseline.json")
    """
    if n_workers is None:
        n_workers = default_config().parallel_episode_workers

    from ppol.fingerprinting import BehavioralFingerprintExtractor

    extractor = BehavioralFingerprintExtractor()
    fp_rows: List[dict] = []
    traj_rows: List[dict] = []

    def _run_one(task: Task):
        result = runner.run_episode(task, persona_policy="")
        fp = extractor.compute_fingerprint(result.trajectory)
        return {
            "task_id": task.task_id,
            "trajectory": result.trajectory,
            "fingerprint": fp.to_dict(),
            "success": result.success,
            "reward": result.reward,
            "split": task.metadata.get("split", "train"),
        }

    it = tasks
    if verbose:
        it = tqdm(tasks, desc="baseline episodes", unit="ep")

    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_one, t): t for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), disable=not verbose, desc="baseline"):
                try:
                    row = fut.result()
                    fp_rows.append({k: v for k, v in row.items() if k != "trajectory"})
                    traj_rows.append({"task_id": row["task_id"], "trajectory": row["trajectory"]})
                except Exception as e:
                    print(f"Warning: episode failed: {e}")
    else:
        for task in it:
            try:
                row = _run_one(task)
                fp_rows.append({k: v for k, v in row.items() if k != "trajectory"})
                traj_rows.append({"task_id": row["task_id"], "trajectory": row["trajectory"]})
            except Exception as e:
                print(f"Warning: episode failed for task {task.task_id}: {e}")

    import datetime
    import numpy as _np
    success_rate = float(_np.mean([r.get("success", False) for r in fp_rows])) if fp_rows else 0.0
    data = {
        "meta": {
            "domain": "custom",
            "split": "custom",
            "n_episodes": len(fp_rows),
            "success_rate": success_rate,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        },
        "episodes": [
            {
                "task_id": fp_row["task_id"],
                "split": fp_row.get("split", "train"),
                "success": fp_row.get("success", False),
                "reward": fp_row.get("reward", 0.0),
                "fingerprint": fp_row["fingerprint"],
                "trajectory": traj_row["trajectory"],
            }
            for fp_row, traj_row in zip(fp_rows, traj_rows)
        ],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    if verbose:
        print(f"Saved {len(fp_rows)} baseline episodes → {output_path}")

    return fp_rows


def train_discriminator(
    human_reference_path: str,
    baseline_fingerprints_path: str,
    output_path: str,
    *,
    scatter_plot_path: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """
    Train a Random Forest discriminator to distinguish human from baseline
    simulator behavior, then save it to disk.

    Args:
        human_reference_path:        Path to ``human_fingerprints_*.json`` from
                                     ``compute_human_reference()``.
        baseline_fingerprints_path:  Path to ``baseline_*_fingerprints.json`` from
                                     ``collect_baseline()``.
        output_path:                 Where to save the trained discriminator (``.pkl``).

    Returns:
        The resolved output path.

    Example::

        from ppol.pipeline import train_discriminator

        train_discriminator(
            human_reference_path="outputs/reference_data/human_fingerprints_custom.json",
            baseline_fingerprints_path="outputs/reference_data/baseline_fingerprints.json",
            output_path="outputs/reference_data/discriminator_custom.pkl",
        )
    """
    from ppol.discriminator import BehavioralDiscriminator
    from ppol.fingerprinting import (
        BehavioralFingerprint,
        BehavioralFingerprintExtractor,
        HumanBehavioralDistribution,
    )

    # Load human reference fingerprints
    human_dist = HumanBehavioralDistribution.load(human_reference_path)

    extractor = BehavioralFingerprintExtractor()
    human_fps: List[BehavioralFingerprint] = []

    # Load per-dialogue fingerprints for training. Priority:
    # 1. Sidecar ``<human_reference_path>.individual.json`` from compute_human_reference()
    # 2. ``episodes`` key in the reference JSON itself (collect_baseline-produced format)
    # 3. Fall back to distribution mean (lower accuracy, warns)
    sidecar = Path(human_reference_path).with_suffix(".individual.json")
    if sidecar.is_file():
        for fp_dict in json.loads(sidecar.read_text())["fingerprints"]:
            human_fps.append(BehavioralFingerprint(features=fp_dict))
    else:
        with open(human_reference_path) as f:
            ref_raw = json.load(f)
        if isinstance(ref_raw, dict) and "episodes" in ref_raw:
            for ep in ref_raw["episodes"]:
                traj = ep.get("trajectory") or ep.get("conversation", [])
                if traj:
                    human_fps.append(extractor.compute_fingerprint(traj))

    if not human_fps:
        print(
            "Warning: no per-dialogue trajectories found for the human reference. "
            "Using distribution mean as a single reference point — discriminator accuracy will be lower. "
            "For best results, pass dialogs through collect_baseline() to embed trajectories."
        )
        human_fps = [BehavioralFingerprint(features=dict(human_dist.mean))]

    # Load baseline fingerprints
    with open(baseline_fingerprints_path) as f:
        raw = json.load(f)
    episodes = raw.get("episodes", raw) if isinstance(raw, dict) else raw
    baseline_fps = [
        BehavioralFingerprint(features=row["fingerprint"])
        for row in episodes
        if isinstance(row, dict) and "fingerprint" in row
    ]

    if not baseline_fps:
        raise ValueError(f"No fingerprints found in {baseline_fingerprints_path}")

    disc = BehavioralDiscriminator()
    disc.train(human_fps, baseline_fps, verbose=verbose)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    disc.save(output_path)
    if verbose:
        print(f"Saved discriminator → {output_path}")

    from ppol.analysis.plot_discriminator_scatter import (
        plot_human_vs_baseline_scatter,
    )

    plot_out = scatter_plot_path or str(
        Path(output_path).with_name("human_baseline_scatter.png")
    )
    plot_human_vs_baseline_scatter(
        human_fps=human_fps,
        baseline_fps=baseline_fps,
        out=plot_out,
        title="Human vs baseline fingerprints (discriminator train)",
        verbose=verbose,
    )

    return output_path


def benchmark_policy(
    runner: EpisodeRunner,
    tasks: List[Task],
    persona_policy_text: str,
    human_reference_path: str,
    discriminator_path: str,
    *,
    output_dir: Optional[str] = None,
    n_workers: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Score a persona policy on a set of tasks.

    Runs one episode per task (with persona injected), extracts behavioral
    fingerprints, and returns human-likeness (mean P(human)) and behavioral
    coverage scores. Optionally saves the full results to ``output_dir``.

    Args:
        runner:               Any ``EpisodeRunner``.
        tasks:                Tasks to evaluate on.
        persona_policy_text:  The persona policy string to evaluate.
        human_reference_path: Path to ``human_fingerprints_*.json``.
        discriminator_path:   Path to ``discriminator_*.pkl``.
        output_dir:           If set, saves results/trajectories/fingerprints JSON files here.
        n_workers:            Parallel episode workers.

    Returns:
        Metrics dict with at minimum ``human_likeness``, ``combined_score``,
        ``n_episodes``, ``persona_success_rate``.

    Example::

        # See examples/tau2bench/runner.py for Tau2BenchRunner
        # (examples/ is not a Python package; add it to sys.path or run from there).
        from ppol.pipeline import benchmark_policy

        runner = Tau2BenchRunner()
        tasks  = runner.get_tasks()

        policy = open("outputs/training_v1/openevolve/best/best_program.py").read()
        # or just a plain text persona string:
        policy = "Be terse and skeptical. Give very short replies."

        metrics = benchmark_policy(
            runner, tasks[:20], policy,
            human_reference_path="outputs/reference_data/human_fingerprints_retail.json",
            discriminator_path="outputs/reference_data/discriminator_retail.pkl",
        )
        print(metrics["human_likeness"], metrics["combined_score"])
    """
    if n_workers is None:
        n_workers = default_config().parallel_episode_workers

    from ppol.discriminator import BehavioralDiscriminator
    from ppol.fingerprinting import (
        BehavioralFingerprintExtractor,
        HumanBehavioralDistribution,
        compute_aggregate_dice_alignment,
    )

    extractor = BehavioralFingerprintExtractor()
    disc = BehavioralDiscriminator.load(discriminator_path)
    human_dist = HumanBehavioralDistribution.load(human_reference_path)

    results: List[EpisodeResult] = []

    def _run_one(task):
        return runner.run_episode(task, persona_policy=persona_policy_text)

    it = tasks
    if verbose:
        it = tqdm(tasks, desc="evaluating policy", unit="ep")

    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_one, t): t for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), disable=not verbose):
                try:
                    results.append(fut.result())
                except Exception as e:
                    print(f"Warning: episode failed: {e}")
    else:
        for task in it:
            try:
                results.append(_run_one(task))
            except Exception as e:
                print(f"Warning: episode failed for task {task.task_id}: {e}")

    if not results:
        return {"combined_score": 0.0, "human_likeness": 0.0, "n_episodes": 0}

    fps = [extractor.compute_fingerprint(r.trajectory) for r in results]
    p_human_scores = [disc.predict_human_probability(fp) for fp in fps]

    import numpy as np
    human_likeness = float(np.mean(p_human_scores))
    dice = compute_aggregate_dice_alignment(fps, human_dist)
    success_rate = float(np.mean([r.success for r in results]))

    metrics = {
        "human_likeness": human_likeness,
        "combined_score": human_likeness,
        "n_episodes": len(results),
        "persona_success_rate": success_rate,
        "dice_d1": dice["D1"],
        "dice_d2": dice["D2"],
        "dice_d3": dice["D3"],
        "dice_d4": dice["D4"],
    }

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(output_dir, "trajectories.json"), "w") as f:
            json.dump([{"task_id": r.task_id, "trajectory": r.trajectory, "success": r.success, "reward": r.reward} for r in results], f, indent=2)
        with open(os.path.join(output_dir, "fingerprints.json"), "w") as f:
            json.dump([fp.to_dict() for fp in fps], f, indent=2)
        if verbose:
            print(f"Results saved → {output_dir}")

    if verbose:
        print(f"human_likeness:    {human_likeness:.4f}")
        print(f"success_rate:      {success_rate:.4f}")
        print(f"n_episodes:        {len(results)}")

    return metrics


class PPol:
    """
    Persona policy session: evolve a generator then generate personas for your tasks.

    Steps 1–3 (baseline, human reference, discriminator) use the standalone
    functions ``compute_human_reference``, ``collect_baseline``, ``train_discriminator``.
    ``PPol`` wraps only the evolution and generation steps.

    Example::

        from ppol import PPol, DataLoader
        from ppol.pipeline import compute_human_reference, collect_baseline, train_discriminator

        # 1–3: data prep (standalone functions)
        dialogs = DataLoader.load_dialogs("human_dialogs.json")
        compute_human_reference(dialogs, "outputs/reference_data/human_fingerprints.json")
        runner = MyRunner(agent=my_agent)
        tasks  = runner.get_tasks()
        collect_baseline(runner, tasks, "outputs/reference_data/baseline.json")
        train_discriminator(
            human_reference_path="outputs/reference_data/human_fingerprints.json",
            baseline_fingerprints_path="outputs/reference_data/baseline.json",
            output_path="outputs/reference_data/discriminator.pkl",
        )

        # 4: evolve
        p = PPol(output_dir="outputs/my_dataset/")
        p.evolve(
            runner=runner, train_tasks=train, val_tasks=val,
            human_reference_path="outputs/reference_data/human_fingerprints.json",
            baseline_path="outputs/reference_data/baseline.json",
            discriminator_path="outputs/reference_data/discriminator.pkl",
            iterations=200,
        )

        # 5: generate personas and run episodes
        personas = p.generate(task_context=tasks[0].description, n=5)
        for persona in personas:
            result = runner.run_episode(tasks[0], persona_policy=persona["text"])
            print(persona["persona_id"], result.reward)
    """

    def __init__(self, *, output_dir: str = "outputs/") -> None:
        self._output_dir = Path(output_dir)

    def evolve(
        self,
        runner,
        train_tasks: List[Task],
        human_reference_path: str,
        baseline_path: str,
        *,
        val_tasks: Optional[List[Task]] = None,
        runner_kwargs: Optional[Dict[str, Any]] = None,
        discriminator_path: Optional[str] = None,
        iterations: Optional[int] = None,
        batch_size: Optional[int] = None,
        n_personas: Optional[int] = None,
        n_workers: Optional[int] = None,
        seed: Optional[int] = None,
        lambda_human_likeness: Optional[float] = None,
        lambda_intra_diversity: Optional[float] = None,
        curriculum: Optional[bool] = None,
        n_personas_schedule: Optional[List[Tuple[int, int]]] = None,
        reflection_model: Optional[str] = None,
        evolution_feedback_max_tokens: Optional[int] = None,
        max_evolution_feedback_chars: Optional[int] = None,
        early_stop_iters_without_new_best: Optional[int] = None,
        standardize_coverage: bool = True,
        train_max_turns: Optional[int] = None,
        resume: bool = False,
        log_level: str = "INFO",
    ) -> bool:
        """Evolve a persona generator with OpenEvolve. Domain-agnostic.

        Args:
            runner:               Any EpisodeRunner instance; its module/class
                                  is auto-detected and re-instantiated inside the
                                  evolution subprocess.
            train_tasks:          Tasks used for training (sampled per iteration).
            human_reference_path: Path to ``human_fingerprints.json``.
            baseline_path:        Path to ``baseline.json``.
            val_tasks:            (Optional) held-out tasks for validation monitoring.
            runner_kwargs:        Init kwargs for the runner class. Required if the
                                  runner's ``__init__`` takes arguments.
            discriminator_path:   (Optional) path to ``discriminator.pkl``.
            iterations:           OpenEvolve iterations (default 70, same as
                                  ``run_evolution.py``; 200+ to converge).
            batch_size:           Train tasks sampled per iteration (default
                                  ``PPolConfig.eval_batch_size``).
            n_personas:           Personas generated per task per iteration
                                  (default ``PPolConfig.n_personas``).
            n_workers:            Parallel episode workers inside fitness
                                  (default ``PPolConfig.parallel_episode_workers``).
            seed:                 Random seed for batch sampling (default
                                  ``PPolConfig.seed``).
            standardize_coverage: z-score the Chamfer coverage space by the human
                                  reference (default True). False = raw 19-D
                                  space (used by the τ²-bench example).
            train_max_turns:      (Optional) per-episode turn cap applied only to
                                  TRAIN episodes, if the runner's ``run_episode``
                                  accepts ``max_turns``. Val episodes use the
                                  runner's own default cap.
            resume:               Resume from latest checkpoint in output_dir.
            log_level:            OpenEvolve log level.

        Returns:
            True on completion / early-stop.

        Best program saved to ``<output_dir>/openevolve/best/best_program.py``.
        """
        import inspect as _inspect

        from ppol.evolution.run_evolution import (
            _checkpoint_sort_key,
            run_openevolve,
        )

        cfg = default_config()
        iterations = DEFAULT_EVOLVE_ITERATIONS if iterations is None else iterations
        batch_size = cfg.eval_batch_size if batch_size is None else batch_size
        n_personas = cfg.n_personas if n_personas is None else n_personas
        n_workers = cfg.parallel_episode_workers if n_workers is None else n_workers
        seed = cfg.seed if seed is None else seed
        lambda_human_likeness = (
            cfg.lambda_human_likeness if lambda_human_likeness is None else lambda_human_likeness
        )
        lambda_intra_diversity = (
            cfg.lambda_intra_diversity if lambda_intra_diversity is None else lambda_intra_diversity
        )
        curriculum = cfg.curriculum if curriculum is None else curriculum
        reflection_model = (
            cfg.evolution_feedback_model if reflection_model is None else reflection_model
        )
        evolution_feedback_max_tokens = (
            cfg.evolution_feedback_max_tokens
            if evolution_feedback_max_tokens is None
            else evolution_feedback_max_tokens
        )
        max_evolution_feedback_chars = (
            cfg.max_evolution_feedback_chars
            if max_evolution_feedback_chars is None
            else max_evolution_feedback_chars
        )
        early_stop_iters_without_new_best = (
            cfg.early_stop_iters_without_new_best
            if early_stop_iters_without_new_best is None
            else early_stop_iters_without_new_best
        )

        _REPO_ROOT = Path(__file__).resolve().parents[1]
        out_dir = self._output_dir / "openevolve"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect runner module + class from the instance
        runner_cls    = type(runner)
        runner_module = _inspect.getsourcefile(runner_cls)
        if runner_module is None:
            raise ValueError(
                f"Could not locate source file for runner class {runner_cls.__name__}; "
                "define it in a .py file (not __main__/REPL)."
            )

        # Write the run config that fitness.py reads via PPOL_RUN_CONFIG
        default_schedule = n_personas_schedule or list(cfg.n_personas_schedule)
        run_config = {
            "runner_module":         str(Path(runner_module).resolve()),
            "runner_class":          runner_cls.__name__,
            "runner_kwargs":         runner_kwargs or {},
            "train_task_ids":        [t.task_id for t in train_tasks],
            "val_task_ids":          [t.task_id for t in (val_tasks or [])],
            "human_reference_path":  str(Path(human_reference_path).resolve()),
            "baseline_path":         str(Path(baseline_path).resolve()),
            "discriminator_path":    str(Path(discriminator_path).resolve()) if discriminator_path else None,
            "openevolve_output_dir": str(out_dir.resolve()),
            "n_personas":            int(n_personas),
            "batch_size":            int(batch_size),
            "n_workers":             int(n_workers),
            "seed":                  int(seed),
            "lambda_human_likeness":  float(lambda_human_likeness),
            "lambda_intra_diversity": float(lambda_intra_diversity),
            "curriculum":             bool(curriculum),
            "n_personas_schedule":    [list(t) for t in default_schedule],
            "reflection_model":       reflection_model,
            "evolution_feedback_max_tokens": int(evolution_feedback_max_tokens),
            "max_evolution_feedback_chars":  int(max_evolution_feedback_chars),
            "early_stop_iters_without_new_best": int(early_stop_iters_without_new_best),
            "standardize_coverage":   bool(standardize_coverage),
            "train_max_turns":        int(train_max_turns) if train_max_turns else None,
        }
        config_path = self._output_dir / "ppol_run_config.json"
        config_path.write_text(json.dumps(run_config, indent=2))
        os.environ["PPOL_RUN_CONFIG"] = str(config_path.resolve())

        resume_checkpoint = None
        if resume:
            ck = out_dir / "checkpoints"
            if ck.is_dir():
                subs = sorted(
                    [p for p in ck.iterdir() if p.is_dir()], key=_checkpoint_sort_key
                )
                if subs:
                    resume_checkpoint = str(subs[-1])

        return run_openevolve(
            initial_program_path=str(
                _REPO_ROOT / "ppol/evolution/initial_generator.py"
            ),
            evaluator_path=str(_REPO_ROOT / "ppol/evolution/fitness.py"),
            config_path=str(
                _REPO_ROOT / "ppol/evolution/openevolve_config.yaml"
            ),
            output_dir=str(out_dir),
            n_iterations=iterations,
            resume_checkpoint=resume_checkpoint,
            log_level=log_level,
        )

    def generate(
        self,
        task_context: Optional[str] = None,
        n: Optional[int] = None,
        *,
        tasks=None,
        best_program: Optional[str] = None,
    ) -> "list[dict] | list[dict[str, object]]":
        """
        Generate persona policies using the evolved generator G(c, D, N).

        Each returned persona dict has:

        - ``"text"``            — the persona string; pass this to ``runner.run_episode(persona_policy=...)``
        - ``"persona_id"``      — short snake_case identifier
        - ``"description"``     — 2-3 sentence human description
        - ``"axis_placement"``  — ``{behavior: bool}`` for each behavioral axis
        - ``"reasoning"``       — why these placements fit together

        Args:
            task_context: The user-simulator system prompt (task scenario). When provided,
                          returns a flat list of N persona dicts for this single context.
            n:            Number of persona policies to generate per task context.
            tasks:        List of ``Task`` objects. When provided, generates N personas
                          per task and returns a list of ``{"task": Task, "personas": [...]}``
                          dicts. Mutually exclusive with ``task_context``.
            best_program: Path to ``best_program.py``. Auto-detected from
                          ``<output_dir>/openevolve/best/best_program.py`` if omitted.

        Returns:
            - If ``task_context`` is given: ``[{"text": ..., "persona_id": ..., ...}, ...]``
            - If ``tasks`` is given: ``[{"task": Task, "personas": [...]}, ...]``

        Example — single task::

            from ppol import PPol
            # See examples/tau2bench/runner.py for Tau2BenchRunner
        # (examples/ is not a Python package; add it to sys.path or run from there).

            p = PPol(output_dir="outputs/my_run/")
            runner = Tau2BenchRunner()
            task = runner.get_tasks()[0]

            personas = p.generate(task_context=task.description, n=5)
            for persona in personas:
                result = runner.run_episode(task, persona_policy=persona["text"])
                print(persona["persona_id"], result.reward)

        Example — all tasks::

            all_results = p.generate(tasks=runner.get_tasks(), n=3)
            for item in all_results:
                task = item["task"]
                for persona in item["personas"]:
                    result = runner.run_episode(task, persona_policy=persona["text"])
        """
        if task_context is not None and tasks is not None:
            raise ValueError("Provide either task_context or tasks, not both.")
        if task_context is None and tasks is None:
            raise ValueError("Provide either task_context or tasks.")

        if n is None:
            n = default_config().n_personas

        module = self._load_generator(best_program=best_program)
        axes = getattr(module, "DIVERSITY_AXES", None)

        def _gen_one(ctx: str) -> list:
            return _call_generator(module, ctx, n, axes)

        if task_context is not None:
            return _gen_one(task_context)

        # tasks mode — parallelize per-task generation (each _gen_one is an
        # independent LLM call; serial over 100 tasks is the benchmark bottleneck).
        def _ctx(task):
            ctx = task.description
            if hasattr(task, "context") and task.context:
                ctx = f"{task.description}\n\nContext: {task.context}"
            return ctx

        output: list = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=min(12, max(1, len(tasks)))) as pool:
            futs = {pool.submit(_gen_one, _ctx(t)): i for i, t in enumerate(tasks)}
            for f in as_completed(futs):
                i = futs[f]
                output[i] = {"task": tasks[i], "personas": f.result()}
        return output

    def _load_generator(self, *, best_program: Optional[str] = None):
        """Load best_program.py as a module. Resolves path from output_dir if not given."""
        import importlib.util

        if best_program is None:
            candidate = self._output_dir / "openevolve" / "best" / "best_program.py"
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"best_program.py not found at {candidate}. "
                    "Run evolve() first or pass best_program= explicitly."
                )
            best_program = str(candidate)

        spec = importlib.util.spec_from_file_location("evolved_persona", best_program)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

