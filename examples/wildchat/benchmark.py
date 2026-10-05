"""Using RealUserSim (WildChat, open-domain chat).

Wrapper over :func:`ppol.evaluation.evaluate_simulators`. Runs the base-simulator 
/ DP / initial / evolved-PPol conditions on the held-out (user-disjoint) test split 
and scores each with P(human) + Chamfer coverage + Dice. Per-episode fingerprints 
are cached under ``outputs/benchmark/episodes_<cond>.json`` (use ``--from-cache``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from runner import WildChatRunner  # noqa: E402

from ppol.evaluation import evaluate_simulators  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(_HERE / "outputs"))
    ap.add_argument("--test-size", type=int, default=100, help="tasks sampled from the held-out test users")
    ap.add_argument("--test-seed", type=int, default=0)
    ap.add_argument("--n-personas", type=int, default=10)
    ap.add_argument("--n-workers", type=int, default=12)
    ap.add_argument("--from-cache", action="store_true",
                    help="recompute metrics from cached fingerprints; run no episodes")
    ap.add_argument("--ppol-program", default=None,
                    help="generator .py to evaluate as the 'ppol' condition "
                         "(default: openevolve/best/best_program.py, the train-best)")
    ap.add_argument("--conditions", default="default,dp,initial,ppol",
                    help="comma-separated subset of: default,dp,initial,ppol")
    ap.add_argument("--human-sim-temp", type=float, default=None,
                    help="override user-simulator temperature (default: runner's 0.0). "
                         "Raise to inject human-like diversity.")
    ap.add_argument("--bench-subdir", default="benchmark",
                    help="subdir under output-dir for results")
    ap.add_argument("--human-sim-model", default=None,
                    help="override the user-simulator model (must match the model the "
                         "discriminator/evolution were run with)")
    ap.add_argument("--judge-model", default=None,
                    help="override the reward judge model (default: agent model)")
    ap.add_argument("--scatter-conditions", default=None,
                    help="conditions to draw in scatter.png (default: all run)")
    args = ap.parse_args()
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    runner_kw = {"split": "test", "num_tasks": args.test_size, "seed": args.test_seed}
    if args.human_sim_temp is not None:
        runner_kw["human_sim_temperature"] = args.human_sim_temp
    if args.human_sim_model:
        runner_kw["human_sim_model"] = args.human_sim_model
    if args.judge_model:
        runner_kw["judge_model"] = args.judge_model
    runner = WildChatRunner(**runner_kw)

    evaluate_simulators(
        runner,
        output_dir=args.output_dir,
        title="WildChat (RealUserSim)",
        name="WildChat",
        n_personas=args.n_personas,
        n_workers=args.n_workers,
        conditions=conditions,
        from_cache=args.from_cache,
        ppol_program=args.ppol_program,
        bench_subdir=args.bench_subdir,
        scatter_conditions=args.scatter_conditions,
        sim_model_label=args.human_sim_model,
    )


if __name__ == "__main__":
    main()
