"""Held-out benchmark for τ²-bench (retail / airline).

Wrapper over :func:`ppol.evaluation.evaluate_simulators` (the shared
harness also used by ColBench/WildChat). Runs the base-simulator / DP /
initial / evolved-PPol conditions on the official τ² **test** split and scores
each with P(human) + Chamfer coverage + Dice.

Coverage is computed in **raw** 19-D fingerprint space (``standardize=False``);
the method rows score against the full train+test human cloud (the harness
default), with the Humans ceiling row vs the train cloud.

    python examples/tau2bench/benchmark.py --n-personas 10
    python examples/tau2bench/benchmark.py --from-cache   # recompute metrics only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
os.environ.setdefault("PERSONA_POLICIES_OUTPUTS_ROOT", str(_HERE / "outputs"))

from runner import Tau2BenchRunner  # noqa: E402
from tau_reference import compute_tau2bench_human_reference  # noqa: E402

from ppol.config import PPolConfig, canonical_domain_name  # noqa: E402
from ppol.evaluation import evaluate_simulators  # noqa: E402


def _ensure_split_references(cfg: PPolConfig) -> None:
    """Build the train / held-out-test human references in the harness's
    standard filenames if absent (analytic — no episodes)."""
    rd = Path(cfg.reference_data_dir)
    train = rd / "human_fingerprints.json"
    test = rd / "human_fingerprints.test.json"
    if not train.is_file():
        compute_tau2bench_human_reference(str(train), domain=cfg.taubench_domain, split="train")
    if not test.is_file():
        compute_tau2bench_human_reference(str(test), domain=cfg.taubench_domain, split="test")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default=None, help="retail, airline, or retail_airline")
    ap.add_argument("--n-personas", type=int, default=10)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--from-cache", action="store_true",
                    help="recompute metrics from cached fingerprints; run no episodes")
    ap.add_argument("--ppol-program", default=None,
                    help="generator .py for the 'ppol' condition "
                         "(default: <openevolve_output_dir>/best/best_program.py)")
    ap.add_argument("--conditions", default="default,dp,initial,ppol",
                    help="comma-separated subset of: default,dp,initial,ppol")
    ap.add_argument("--bench-subdir", default="benchmark",
                    help="subdir under outputs/ for results")
    ap.add_argument("--scatter-conditions", default=None,
                    help="conditions to draw in scatter.png (default: all run)")
    args = ap.parse_args(argv)

    if args.domain:
        os.environ["PERSONA_POLICIES_DOMAIN"] = args.domain
    cfg = PPolConfig()
    cfg.ensure_output_dirs()
    _ensure_split_references(cfg)

    runner = Tau2BenchRunner(cfg)
    runner.use_split(cfg.task_split_benchmark)   # official test split

    ppol_program = args.ppol_program or str(
        Path(cfg.openevolve_output_dir) / "best" / "best_program.py")

    evaluate_simulators(
        runner,
        output_dir=cfg.outputs_root,
        title=f"τ²-bench ({canonical_domain_name(cfg.taubench_domain)})",
        name="tau2bench",
        n_personas=args.n_personas,
        n_workers=args.n_workers,
        conditions=[c.strip() for c in args.conditions.split(",") if c.strip()],
        from_cache=args.from_cache,
        ppol_program=ppol_program,
        bench_subdir=args.bench_subdir,
        scatter_conditions=args.scatter_conditions,
        sim_model_label=cfg.taubench_user_model,
        standardize=False,                              # raw 19-D fingerprint space
        discriminator_path=cfg.discriminator_model_path,
    )


if __name__ == "__main__":
    main()
