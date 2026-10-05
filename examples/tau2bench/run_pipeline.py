"""
Full Pipeline Runner
====================
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Keep all tau2bench artifacts under examples/tau2bench/outputs/ (never repo-root outputs/).
os.environ.setdefault(
    "PERSONA_POLICIES_OUTPUTS_ROOT", str(Path(__file__).resolve().parent / "outputs")
)


def step_data():
    """Human references in the standard ppol filenames (train / held-out test /
    full). Train = evolution coverage cloud; train+test = held-out eval cloud;
    full = discriminator human class."""
    print("\n[STEP 1] Human behavioral references from τ²-bench human data...")
    from ppol.config import PPolConfig
    from tau_reference import compute_tau2bench_human_reference

    cfg = PPolConfig()
    cfg.ensure_output_dirs()
    rd = Path(cfg.reference_data_dir)
    for split, name in (("train", "human_fingerprints.json"),
                        ("test", "human_fingerprints.test.json"),
                        ("all", "human_fingerprints_full.json")):
        compute_tau2bench_human_reference(str(rd / name), domain=cfg.taubench_domain, split=split)
    print(f"✓ Data step complete: references → {rd}")


def step_baseline():
    """Persona-free τ² rollouts over the official train∪test pool (the
    discriminator's negative class) via the generic ``ppol.pipeline``."""
    print("\n[STEP 2] Collecting baseline τ² trajectories (train∪test)...")
    from ppol.config import PPolConfig
    from ppol.pipeline import collect_baseline
    from runner import Tau2BenchRunner

    cfg = PPolConfig()
    out = Path(cfg.reference_data_dir) / "baseline.json"
    if out.is_file():
        print(f"✓ Baseline step skipped ({out} already present)")
        return
    runner = Tau2BenchRunner(cfg)
    runner.use_train_and_test_splits()
    collect_baseline(runner, runner.get_tasks(), str(out),
                     n_workers=cfg.parallel_episode_workers)
    print("✓ Baseline step complete")


def step_discriminator():
    print("\n[STEP 3] Training behavioral discriminator (full humans vs baseline)...")
    from ppol.config import PPolConfig
    from ppol.pipeline import train_discriminator

    cfg = PPolConfig()
    rd = Path(cfg.reference_data_dir)
    train_discriminator(
        human_reference_path=str(rd / "human_fingerprints_full.json"),
        baseline_fingerprints_path=str(rd / "baseline.json"),
        output_path=cfg.discriminator_model_path,
    )
    print("✓ Discriminator step complete")


def step_evolve(iterations: int, domain: str | None = None, val_fraction: float | None = None):
    """Evolve via the generic ``PPol.evolve`` (same path as ColBench/WildChat).

    The coverage cloud is the TRAIN-split human reference in raw 19-D space
    (``standardize_coverage=False``) and train episodes cap at
    ``config.train_max_steps_per_episode`` turns.
    """
    print("\n[STEP 4] Evolving persona generator G(c, D, N) via PPol.evolve...")
    from ppol import PPol
    from ppol.config import PPolConfig
    from runner import Tau2BenchRunner
    from tau_reference import compute_tau2bench_human_reference
    from tau_train_context import split_train_val

    cfg = PPolConfig()
    rd = Path(cfg.reference_data_dir)

    # Train-split human reference (coverage cloud), built once per run dir.
    train_ref = str(rd / "human_fingerprints.json")
    if not Path(train_ref).is_file():
        compute_tau2bench_human_reference(train_ref, domain=cfg.taubench_domain, split="train")

    runner = Tau2BenchRunner(cfg)
    tasks_by_id = {t.task_id: t for t in runner.get_tasks()}
    train_ids, val_ids = split_train_val(
        seed=cfg.seed, val_fraction=cfg.val_fraction,
        domain=cfg.taubench_domain, taubench_root=Path(cfg.taubench_root),
    )
    train_tasks = [tasks_by_id[i] for i in train_ids if i in tasks_by_id]
    val_tasks = [tasks_by_id[i] for i in val_ids if i in tasks_by_id]
    print(f"  train pool: {len(train_tasks)} tasks | val: {len(val_tasks)} tasks")

    p = PPol(output_dir=str(Path(cfg.openevolve_output_dir).parent))
    p.evolve(
        runner=runner,
        train_tasks=train_tasks,
        val_tasks=val_tasks,
        human_reference_path=train_ref,
        baseline_path=cfg.baseline_path,
        discriminator_path=cfg.discriminator_model_path,
        iterations=iterations,
        standardize_coverage=False,                       # raw 19-D Chamfer space
        train_max_turns=cfg.train_max_steps_per_episode,  # shorter train episodes
        resume=False,
    )
    print("✓ Evolution step complete")


def step_benchmark():
    print("\n[STEP 5] Benchmark (test split) via ppol.evaluation.evaluate_simulators...")
    # benchmark.py lives alongside this file (examples/ is not a package).
    import importlib.util as _ilu

    _bench_path = Path(__file__).resolve().parent / "benchmark.py"
    _spec = _ilu.spec_from_file_location("_tau2_benchmark", _bench_path)
    _bench = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_bench)
    _bench.main([])
    print("✓ Benchmark step complete")


def main():
    from ppol.config import PPolConfig

    PPolConfig().ensure_output_dirs()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help="Comma-separated: data,baseline,discriminator,evolve,benchmark or 'all'",
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--domain",
        default=None,
        help="tau2 domain selector, e.g. retail, airline, or retail_airline",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Validation fraction inside the official train split, e.g. 0.1",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Run label; output folder is outputs/training_<version>/",
    )
    args = parser.parse_args()

    if args.domain:
        os.environ["PERSONA_POLICIES_DOMAIN"] = args.domain
    if args.version:
        os.environ["PERSONA_POLICIES_VERSION"] = args.version
    elif not os.environ.get("PERSONA_POLICIES_VERSION"):
        # Interpretable default suffix: <domain>_<user-sim-model>, e.g. retail_qwen.
        from ppol.config import PPolConfig, canonical_domain_name, short_model_tag
        _cfg = PPolConfig()
        os.environ["PERSONA_POLICIES_VERSION"] = (
            f"{canonical_domain_name(_cfg.taubench_domain)}_{short_model_tag(_cfg.taubench_user_model)}"
        )
    if args.val_fraction is not None:
        os.environ["PERSONA_POLICIES_VAL_FRACTION"] = str(args.val_fraction)

    if args.steps == "all":
        steps = ["data", "baseline", "discriminator", "evolve", "benchmark"]
    else:
        steps = [s.strip() for s in args.steps.split(",")]

    for step in steps:
        if step == "data":
            step_data()
        elif step == "baseline":
            step_baseline()
        elif step == "discriminator":
            step_discriminator()
        elif step == "evolve":
            step_evolve(args.iterations, args.domain, args.val_fraction)
        elif step == "benchmark":
            step_benchmark()
        else:
            print(f"Unknown step: {step}")


if __name__ == "__main__":
    main()
