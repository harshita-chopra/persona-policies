"""
Evolution Launcher
==================
Runs OpenEvolve on the persona generator G(c, D, N). The evolved program contains
LLM prompts for population (joint generation) and roleplay expansion
(task-conditioned). OpenEvolve mutates these prompts over iterations.

Each fitness evaluation samples a fresh batch of τ² val tasks, runs G to produce
personas, simulates τ² episodes, and scores quality + diversity.

Usage:

  # Default run (recommended)
  python persona_policies/evolution/run_evolution.py --iterations 200

  # Resume from checkpoint
  python persona_policies/evolution/run_evolution.py --iterations 200 --resume

Requires ``openevolve-run`` on PATH (``pip install openevolve``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from persona_policies.config import PersonaPoliciesConfig


def _require_runtime_packages() -> None:
    missing = []
    for package, import_name in (
        ("litellm", "litellm"),
        ("openevolve", "openevolve"),
    ):
        try:
            __import__(import_name)
        except ModuleNotFoundError:
            missing.append(package)
    if not missing:
        return
    cmd = f"{sys.executable} -m pip install " + " ".join(missing)
    print(
        "error: missing Python package(s) in this environment: "
        f"{', '.join(missing)}\nInstall them with:\n  {cmd}",
        file=sys.stderr,
    )
    sys.exit(1)


def _read_openevolve_checkpoint_interval(config_path: Path) -> int:
    """Match ``checkpoint_interval`` in ``openevolve_config.yaml`` (banner only)."""
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        return 1
    m = re.search(r"^checkpoint_interval:\s*(\d+)", text, re.MULTILINE)
    return int(m.group(1)) if m else 1


def _checkpoint_interval_banner_line(n: int) -> str:
    if n <= 1:
        return "each iteration"
    return f"~every {n} iterations"


def _checkpoint_sort_key(p: Path) -> tuple[int, int | float]:
    """Prefer numeric ``checkpoint_<N>`` ordering; fallback to mtime for others."""
    m = re.fullmatch(r"checkpoint_(\d+)", p.name)
    if m:
        return (1, int(m.group(1)))
    try:
        return (0, p.stat().st_mtime_ns)
    except OSError:
        return (0, -1)


def _print_evolution_outputs_help(output_dir: str, checkpoint_interval: int) -> None:
    root = Path(output_dir).resolve()
    cadence = _checkpoint_interval_banner_line(checkpoint_interval)
    print(
        f"""
  OpenEvolve --output = {root}:
    checkpoints/     OpenEvolve DB (resume); programs/ under checkpoint_* — snapshot cadence: {cadence}
                     (set checkpoint_interval in persona_policies/evolution/openevolve_config.yaml)
    best/            best_program.py (final best code)
    logs/            controller logs

  Sibling folders (same training/ parent):
    ../simulations/iter_NNNN/  τ² fitness logs, trajectories, scores (not used for resume)
    ../results/evolution_best.json  best metrics JSON when run finishes

  Resume: --checkpoint <path/to/checkpoint_N>

  Bedrock mutation LLM: run once per venv:
    python persona_policies/evolution/install_openevolve_litellm_pth.py
"""
    )


def run_openevolve(
    initial_program_path: str,
    evaluator_path: str,
    config_path: str,
    output_dir: str,
    n_iterations: int,
    resume_checkpoint: str | None,
    log_level: str | None,
) -> bool:
    cmd = [
        sys.executable,
        "-m",
        "persona_policies.evolution.openevolve_entry",
        initial_program_path,
        evaluator_path,
        "--config",
        config_path,
        "--output",
        output_dir,
        "--iterations",
        str(n_iterations),
    ]
    if resume_checkpoint and os.path.isdir(resume_checkpoint):
        cmd.extend(["--checkpoint", resume_checkpoint])
    if log_level:
        cmd.extend(["--log-level", log_level])

    print(f"\n{'='*70}\nEVOLVING PERSONA GENERATOR\nCommand: {' '.join(cmd)}\n{'='*70}")

    (Path(output_dir) / "EARLY_STOP").unlink(missing_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    r = str(_REPO_ROOT)
    p = env.get("PYTHONPATH", "")
    if r not in p.split(os.pathsep):
        env["PYTHONPATH"] = f"{r}{os.pathsep}{p}" if p else r

    rc = subprocess.run(cmd, env=env).returncode
    return rc == 0 or (Path(output_dir) / "EARLY_STOP").is_file()


def collect_best_from_database(output_dir: str) -> dict | None:
    db_path = Path(output_dir) / "database.json"
    if not db_path.is_file():
        return None
    try:
        data = json.loads(db_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    programs = data.get("programs") or data.get("archive") or []
    best = None
    best_score = -1.0
    for pid, prog in programs.items() if isinstance(programs, dict) else []:
        if isinstance(prog, dict) and "metrics" in prog:
            m = prog["metrics"]
            s = float(m.get("combined_score", -1))
            if s > best_score:
                best_score = s
                best = {"id": pid, "metrics": m}
    if best:
        return {"combined_score": best_score, "metrics": best.get("metrics", {})}
    return None


def main():
    _require_runtime_packages()

    parser = argparse.ArgumentParser(
        description="Evolve persona generator G(c, D, N) via OpenEvolve"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--iterations",
        type=int,
        default=70,
        help="OpenEvolve iterations (default 70)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Pass --log-level DEBUG to openevolve-run.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
    )
    parser.add_argument(
        "--version",
        default="",
        help="Optional run label; training folder becomes training_<version>/ when set.",
    )
    parser.add_argument(
        "--domain",
        default="",
        help="Optional tau2 domain override, e.g. airline or retail_airline. Also propagated to evaluator subprocesses.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Fraction of official train tasks held out for validation (default: config.val_fraction).",
    )
    args = parser.parse_args()

    if args.domain:
        os.environ["PERSONA_POLICIES_DOMAIN"] = args.domain
    if args.val_fraction is not None:
        os.environ["PERSONA_POLICIES_VAL_FRACTION"] = str(args.val_fraction)
    if args.version:
        os.environ["PERSONA_POLICIES_VERSION"] = args.version
    elif args.domain and not os.environ.get("PERSONA_POLICIES_VERSION"):
        os.environ["PERSONA_POLICIES_VERSION"] = args.domain

    log_level = args.log_level
    if log_level is None and args.verbose:
        log_level = "DEBUG"
    elif log_level is None:
        log_level = "INFO"

    config = PersonaPoliciesConfig()
    config.ensure_output_dirs()
    out_dir = str(_REPO_ROOT / config.openevolve_output_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    initial_path = str(_REPO_ROOT / "persona_policies/evolution/initial_generator.py")
    evaluator_path = str(_REPO_ROOT / "persona_policies/evolution/fitness.py")
    config_path = str(_REPO_ROOT / "persona_policies/evolution/openevolve_config.yaml")
    _checkpoint_interval = _read_openevolve_checkpoint_interval(Path(config_path))
    _print_evolution_outputs_help(out_dir, _checkpoint_interval)

    # Print train/val split info
    import math as _math
    from persona_policies.tau_train_context import split_train_val
    train_ids, val_ids = split_train_val(
        seed=config.seed,
        val_fraction=config.val_fraction,
        domain=config.taubench_domain,
        taubench_root=Path(config.taubench_root),
    )
    _steps_per_epoch = max(1, _math.ceil(len(train_ids) / max(1, config.eval_batch_size)))
    print(
        f"\n  Train/val split ({config.taubench_domain}): "
        f"{len(train_ids)} train, {len(val_ids)} val tasks"
    )
    print(
        f"  Each evaluate(): {config.eval_batch_size} tasks × {config.n_personas} personas/task "
        f"(parallel workers={config.parallel_episode_workers})"
    )
    print(
        f"  Flow: 1 iter == 1 random training minibatch; {_steps_per_epoch} batches ~= 1 epoch; "
        f"FULL val runs when OpenEvolve's elite changes (checkpoint or final best/; monitor only)"
    )
    print(f"  Episodes per (persona, task): {config.n_episodes_per_persona_task}\n")

    resume = None
    if args.resume:
        ck = Path(out_dir) / "checkpoints"
        if ck.is_dir():
            subs = sorted([p for p in ck.iterdir() if p.is_dir()], key=_checkpoint_sort_key)
            if subs:
                resume = str(subs[-1])

    ok = run_openevolve(
        initial_path,
        evaluator_path,
        config_path,
        out_dir,
        args.iterations,
        resume,
        log_level,
    )

    if ok:
        best = collect_best_from_database(out_dir)
        if best:
            results_dir = _REPO_ROOT / config.training_results_dir
            results_dir.mkdir(parents=True, exist_ok=True)
            out_json = results_dir / "evolution_best.json"
            out_json.write_text(json.dumps(best, indent=2), encoding="utf-8")
            print(f"Saved best metrics to {out_json}")
        bp = Path(out_dir) / "best" / "best_program.py"
        if bp.is_file():
            print(f"Best generator code: {bp}")


if __name__ == "__main__":
    main()
