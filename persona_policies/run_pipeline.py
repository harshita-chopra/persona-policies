"""
Full Pipeline Runner
====================
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def step_data():
    print("\n[STEP 1] Human behavioral reference from τ²-bench human data (domain = taubench_domain)...")
    from persona_policies.config import PersonaPoliciesConfig
    from persona_policies.scripts.compute_human_reference import compute_and_save

    cfg = PersonaPoliciesConfig()
    cfg.ensure_output_dirs()
    out = compute_and_save(cfg)
    print(f"✓ Data step complete: human reference → {out}")


def step_baseline():
    print("\n[STEP 2] Collecting baseline τ² trajectories (config.baseline_collect_split; default all train∪test)...")
    from persona_policies.config import PersonaPoliciesConfig
    from persona_policies.scripts.collect_baseline import collect_baseline

    traj, _ = collect_baseline(PersonaPoliciesConfig(), n_episodes=None)
    if traj is None:
        print("✓ Baseline step skipped (artifacts already present; use --force in collect_baseline to regenerate)")
    else:
        print("✓ Baseline step complete")


def step_discriminator():
    print("\n[STEP 3] Training behavioral discriminator...")
    from persona_policies.scripts.train_discriminator import main as train_main

    train_main()
    print("✓ Discriminator step complete")


def step_evolve(iterations: int, domain: str | None = None, val_fraction: float | None = None):
    print("\n[STEP 4] Running OpenEvolve (G(c, D, N) via initial_generator.py)...")
    cmd = [
        sys.executable,
        str(_REPO / "persona_policies/evolution/run_evolution.py"),
        "--iterations",
        str(iterations),
    ]
    if domain:
        cmd.extend(["--domain", domain])
    if val_fraction is not None:
        cmd.extend(["--val-fraction", str(val_fraction)])
    subprocess.run(cmd, check=True)
    print("✓ Evolution step complete")


def step_benchmark():
    print("\n[STEP 5] Benchmark (test split)...")
    from persona_policies.benchmark import run
    from persona_policies.config import PersonaPoliciesConfig

    cfg = PersonaPoliciesConfig()
    best = Path(cfg.openevolve_output_dir) / "best" / "best_program.py"
    if not best.is_absolute():
        best = _REPO / best
    if not best.is_file():
        raise SystemExit(f"No best_program at {best}; run evolution first.")
    run(cfg, best)
    print("✓ Benchmark step complete")


def main():
    from persona_policies.config import PersonaPoliciesConfig

    PersonaPoliciesConfig().ensure_output_dirs()

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
    elif args.domain:
        os.environ["PERSONA_POLICIES_VERSION"] = args.domain
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
