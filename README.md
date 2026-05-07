# Persona Policies

**Persona Policies (PPol)** is a framework for evolving realistic, diverse user-simulator personas for LLM agent benchmarks. It is a plug-and-play control layer that injects behavioral variation into existing user simulators without changing task goals, rewards, or environment state.

## Overview

Default LLM-based user simulators (e.g., in τ²-bench) are overly cooperative and homogeneous, producing a behavioral gap relative to real users. PPol narrows this gap by:

1. Generating a population of **persona policies** — short natural-language instructions appended to the user simulator's system prompt that control *how* the user communicates while keeping task facts fixed.
2. Optimizing the persona generator via **evolutionary program search** (OpenEvolve), guided by two objectives: **human-likeness** (a trained Random Forest discriminator on behavioral fingerprints) and **behavioral coverage** (Chamfer distance to a human reference distribution).

## Quick Start

### Prerequisites

1. **Clone this repository**
2. **Install τ²-bench** from its official repository (not included here):
   ```bash
   git clone <tau2-bench-repo-url>
   pip install -e ./tau2-bench
   ```
3. **Create the conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate persona-policies
   ```
4. **Set API credentials** for your LLM provider(s):
   ```bash
   export OPENROUTER_API_KEY=...   # for OpenRouter models
   export AWS_REGION_NAME=us-west-2  # if using Bedrock models
   ```
5. **Wire OpenEvolve through LiteLLM** (run once per environment):
   ```bash
   python persona_policies/evolution/install_openevolve_litellm_pth.py
   ```

### Running the Full Pipeline

```bash
# 1. Collect baseline τ²-bench trajectories (no persona injection)
python persona_policies/scripts/collect_baseline.py --n 100

# 2. Train the behavioral discriminator (human vs. simulator)
python persona_policies/scripts/train_discriminator.py

# 3. Run OpenEvolve to evolve the persona generator
python persona_policies/evolution/run_evolution.py --iterations 70

# 4. Benchmark the evolved program on the test split
python -m persona_policies.benchmark \
    --best-program persona_policies/outputs/training_v1/openevolve/best/best_program.py
```

Or use the convenience runner:
```bash
python persona_policies/run_pipeline.py --iterations 70 --domain retail
```

Full documentation, argument reference, and output layout: **[persona_policies/README.md](persona_policies/README.md)**.

## Agent Training

`sft_experiments/` contains the LoRA fine-tuning code for training Gemma-4-31B on default-only vs. PPol-augmented τ²-bench rollouts. See **[sft_experiments/README.md](sft_experiments/README.md)** for the full pipeline (trajectory collection → ShareGPT conversion → LLaMA-Factory training → evaluation).

## Repository Structure

```
persona_policies/          # Core Python package
├── config.py              # Central configuration (PersonaPoliciesConfig)
├── benchmark.py           # Test-split evaluation and plots
├── evaluator.py           # Multi-objective scorer (human-likeness + coverage)
├── discriminator.py       # Random Forest behavioral classifier
├── fingerprinting.py      # 19-feature behavioral fingerprint extractor
├── injector.py            # Persona injection into τ²-bench at runtime
├── evolution/             # OpenEvolve integration
│   ├── initial_generator.py   # The evolved program (G(c, D, N))
│   ├── fitness.py             # OpenEvolve fitness callback
│   ├── run_evolution.py       # Evolution launcher
│   └── openevolve_config.yaml # Evolution hyperparameters
├── scripts/               # Setup scripts (baselines, discriminator, etc.)
├── data/                  # Human dialogue reference data
└── outputs/               # Generated artifacts (gitignored; created at runtime)
    ├── reference_data/    # Baselines, discriminators, human fingerprints
    ├── training_<name>/   # OpenEvolve checkpoints and logs
    └── testing/           # Benchmark results

sft_experiments/           # Agent LoRA fine-tuning
├── configs/               # LLaMA-Factory LoRA YAMLs (one per domain × regime)
├── data/                  # Trajectory collection and ShareGPT data builder
└── train/                 # Thin LLaMA-Factory launcher
```

