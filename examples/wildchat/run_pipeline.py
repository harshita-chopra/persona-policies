"""RealUserSim (WildChat) × ppol pipeline (open-domain chat track).

RealUserSim is one HF split (1200 conversations); this pipeline partitions it 
user-disjointly so no real human appears in both the train side and the held-out side.

Steps (--steps, comma-separated or 'all'):
  reference     RealUserSim human reference (real user turns), split disjointly
                by user_ip into a train half (disc + coverage) and a held-out half.
  baseline      persona-free user-sim episodes on train tasks (discriminator neg)
  discriminator RF: real WildChat user vs. persona-free sim
  evolve        evolve the persona generator G(c, D, N) on the train pool.

Held-out evaluation is separate: `python examples/wildchat/benchmark.py`.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from runner import WildChatRunner  # noqa: E402


def _out_dir(args) -> Path:
    d = Path(args.output_dir)
    (d / "reference_data").mkdir(parents=True, exist_ok=True)
    return d


def _runner_kwargs(args) -> dict:
    return {"split": "train", "num_tasks": args.train_pool, "seed": args.train_seed,
            "train_frac": args.train_frac, "agent_model": args.agent_model,
            "human_sim_model": args.human_sim_model, "judge_model": args.judge_model,
            "max_steps": args.max_steps}


def _train_val_split(args):
    """Return (runner, train_tasks, val_tasks): fixed val_size held out of the
    seed-shuffled train pool; the rest is the train pool."""
    runner = WildChatRunner(**_runner_kwargs(args))
    tasks = runner.get_tasks()
    random.Random(args.train_seed).shuffle(tasks)
    return runner, tasks[args.val_size:], tasks[: args.val_size]


def step_reference(args):
    print("\n[STEP 1] Building RealUserSim human reference...")
    from wildchat_reference import compute_wildchat_human_reference

    out = compute_wildchat_human_reference(
        str(_out_dir(args) / "reference_data" / "human_fingerprints.json"),
        max_conversations=args.max_conversations, train_frac=args.train_frac,
        split_seed=args.train_seed,
    )
    print(f"✓ Human reference → {out}")


def step_baseline(args):
    print("\n[STEP 2] Collecting persona-free baseline episodes (train tasks)...")
    from ppol.pipeline import collect_baseline

    runner, train, _ = _train_val_split(args)
    collect_baseline(
        runner, train[: args.baseline_tasks],
        str(_out_dir(args) / "reference_data" / "baseline.json"),
        n_workers=args.n_workers,
    )
    print("✓ Baseline complete")


def step_discriminator(args):
    print("\n[STEP 3] Training discriminator (real user vs. persona-free sim)...")
    from ppol.pipeline import train_discriminator

    rd = _out_dir(args) / "reference_data"
    train_discriminator(
        human_reference_path=str(rd / "human_fingerprints.json"),
        baseline_fingerprints_path=str(rd / "baseline.json"),
        output_path=str(rd / "discriminator.pkl"),
    )
    print("✓ Discriminator complete")


def step_evolve(args):
    print("\n[STEP 4] Evolving persona generator (train pool)...")
    from ppol import PPol

    runner, train, val = _train_val_split(args)
    print(f"  train pool: {len(train)} tasks | fixed val: {len(val)} tasks")
    rd = _out_dir(args) / "reference_data"
    p = PPol(output_dir=str(_out_dir(args)))
    p.evolve(
        runner=runner,
        runner_kwargs=_runner_kwargs(args),
        train_tasks=train,
        val_tasks=val,
        human_reference_path=str(rd / "human_fingerprints.json"),
        baseline_path=str(rd / "baseline.json"),
        discriminator_path=str(rd / "discriminator.pkl"),
        iterations=args.iterations,
        n_workers=args.n_workers,
        resume=args.resume,
        lambda_intra_diversity=args.lambda_b,
        batch_size=args.batch_size,
    )
    print("✓ Evolution complete")


_STEPS = {
    "reference": step_reference,
    "baseline": step_baseline,
    "discriminator": step_discriminator,
    "evolve": step_evolve,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="all",
                    help="Comma-separated subset of: " + ",".join(_STEPS) + " (or 'all')")
    ap.add_argument("--output-dir", default=str(_HERE / "outputs"))
    ap.add_argument("--train-pool", type=int, default=400, help="tasks sampled from the train users")
    ap.add_argument("--val-size", type=int, default=20, help="fixed val tasks held out of the pool")
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.75, help="fraction of users in the train split")
    ap.add_argument("--baseline-tasks", type=int, default=100, help="train tasks for the persona-free baseline (disc negative class)")
    ap.add_argument("--max-conversations", type=int, default=None, help="optional cap on RealUserSim dialogs (default: all eligible)")
    ap.add_argument("--iterations", type=int, default=70)
    ap.add_argument("--lambda-b", type=float, default=0.5,
                    help="coverage weight (lambda_intra_diversity); λ_b at N=10, λ_h=1−λ_b")
    ap.add_argument("--n-workers", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--resume", action="store_true",
                    help="resume evolution from the latest checkpoint (runs --iterations MORE)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="train tasks per evolution eval (default 5; raise to cut fitness noise)")
    ap.add_argument("--agent-model", default=None)
    ap.add_argument("--human-sim-model", default=None)
    ap.add_argument("--judge-model", default=None)
    args = ap.parse_args()

    if args.agent_model is None or args.human_sim_model is None:
        from ppol.config import default_config
        cfg = default_config()
        args.agent_model = args.agent_model or cfg.taubench_agent_model
        args.human_sim_model = args.human_sim_model or cfg.taubench_user_model

    steps = list(_STEPS) if args.steps == "all" else [s.strip() for s in args.steps.split(",")]
    for s in steps:
        if s not in _STEPS:
            raise SystemExit(f"Unknown step: {s}")
        _STEPS[s](args)


if __name__ == "__main__":
    main()
