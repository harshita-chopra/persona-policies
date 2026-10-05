# τ²-bench example

End-to-end pipeline for τ²-bench (retail / airline domains) — human reference, baseline rollouts, discriminator, evolution, benchmark.

## Contents

| File | Purpose |
|---|---|
| `run_pipeline.py` | One-shot Python driver: data → baseline → discriminator → evolve → benchmark. |
| `runner.py` | `Tau2BenchRunner` (ppol `EpisodeRunner`) + τ² orchestrator glue. |
| `tau_reference.py` | Human-reference builder (train/test/all splits) from the real τ² human dialogues — [`cmu-lti/tau-usi`](https://huggingface.co/datasets/cmu-lti/tau-usi), **auto-downloaded** on first use. |
| `benchmark.py` | Wrapper over `ppol.evaluation.evaluate_simulators` (held-out eval on the test split). |
| `tau_train_context.py` | Official train/test split + evolution train/val slicing helpers. |

### τ²-bench data layout (splits)

Task ids and the official train/test split come from the τ²-bench checkout
(`<taubench_root>/data/tau2/domains/<domain>/split_tasks.json`, with
`tasks.json` beside it). `taubench_root` is auto-derived from the installed
`tau2` package's data dir when no local `tau2-bench/` checkout exists.
Evolution trains on the official **train** split (a `val_fraction` slice is
held out for monitoring); `benchmark.py` evaluates on the official **test**
split. The human dialogues (auto-downloaded from [`cmu-lti/tau-usi`](https://huggingface.co/datasets/cmu-lti/tau-usi); or point `PPolConfig.tau_bench_human_path` at a local copy) carry `instance_id`s that map back to task ids, which is how train-split human references are built.

## Prerequisites

```bash
pip install ppol
pip install git+https://github.com/sierra-research/tau2-bench   # τ² itself, not on PyPI
export OPENROUTER_API_KEY=...    # or OPENAI_API_KEY / GEMINI_API_KEY / AWS creds

# once per venv: route OpenEvolve's mutation LLM through LiteLLM
python ppol/evolution/install_openevolve_litellm_pth.py
```

## Run

```bash
python examples/tau2bench/run_pipeline.py --iterations 200 --domain retail
```

This will, in order:

1. **Human reference** — fingerprint the τ² human dialogues (auto-downloaded from `cmu-lti/tau-usi`) filtered to the configured domain.
2. **Baseline** — run τ² rollouts *without* any persona to anchor the "not human-like" class.
3. **Discriminator** — train an RF classifier on human vs. baseline fingerprints.
4. **Evolution** — `PPol.evolve()` (the same generic path as ColBench/WildChat), with a train-split human coverage cloud in raw 19-D space and a capped train-episode length.
5. **Benchmark** — score the best evolved program on the test split.

All artifacts land under `outputs/` (configurable via `PPolConfig.outputs_root`).

To resume a stopped evolution run without re-running earlier steps, call
`PPol.evolve(..., resume=True)` (see `run_pipeline.py:step_evolve`).

## Selecting the LLMs

τ² involves several LLM roles; edit `ppol/config.py` (or pass a `PPolConfig`) to choose:

| Role | Field | Typical choice |
|---|---|---|
| τ² agent (task performer) | `taubench_agent_model` | strong model — drives task success |
| τ² user simulator | `taubench_user_model` | small/cheap is fine |
| Reflection (per-iter feedback) | `evolution_feedback_model` | mid-tier |
| OpenEvolve mutation | (in `openevolve_config.yaml`) | strong model, must route through LiteLLM |

## Programmatic version

Run from the repo root so `examples/tau2bench/` is on `sys.path`:

```python
import sys; sys.path.insert(0, "examples/tau2bench")
from runner import Tau2BenchRunner
from tau_reference import compute_tau2bench_human_reference

from ppol import PPol, split_tasks
from ppol.pipeline import collect_baseline, train_discriminator

runner = Tau2BenchRunner()
train, val, _ = split_tasks(runner.get_tasks())

compute_tau2bench_human_reference("outputs/ref/human_retail.json", domain="retail")
collect_baseline(runner, runner.get_tasks(), "outputs/ref/baseline_retail.json", n_workers=4)
train_discriminator(
    human_reference_path="outputs/ref/human_retail.json",
    baseline_fingerprints_path="outputs/ref/baseline_retail.json",
    output_path="outputs/ref/discriminator_retail.pkl",
)

p = PPol(output_dir="outputs/retail")
p.evolve(
    runner=runner, train_tasks=train, val_tasks=val,
    human_reference_path="outputs/ref/human_retail.json",
    baseline_path="outputs/ref/baseline_retail.json",
    discriminator_path="outputs/ref/discriminator_retail.pkl",
    iterations=200,
)
```

See [`ppol/README.md`](../../ppol/README.md) for the full API reference.
