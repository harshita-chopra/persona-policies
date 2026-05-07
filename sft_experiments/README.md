# SFT Experiments

Code for the agent-training case study. We fine-tune Gemma-4-31B with LoRA on two regimes:

- **Default-only SFT** — successful τ²-bench rollouts using the default user simulator.
- **Default + PPol SFT** — same recipe, training mix includes successful PPol-persona rollouts.

Three domains: `retail`, `airline`, `retail_airline`.

## Layout

```
sft_experiments/
  configs/                   LoRA SFT YAMLs (LLaMA-Factory) + dataset_info fragment
  data/
    collect_baseline_trajectories.py   τ² rollouts, default user, no persona
    collect_persona_trajectories.py    τ² rollouts, PPol persona-conditioned user
    build_sharegpt_sft_data.py         raw τ² traces → ShareGPT (tool-aware) for LLaMA-Factory
  train/
    train_lora.py            thin wrapper around `llamafactory-cli train <yaml>`
```

## Dependencies

External — install separately and point env vars at them:

- `tau2-bench` — task definitions and rollout runner: `<tau2-bench-repo-url>`
- `tau-trait` — out-of-distribution user simulator suite (Skeptical / Incoherent / Impatient / Confusion): `<tau-trait-repo-url>`
- `persona_policies` — PPol persona generator and τ² episode runner: see [`../persona_policies/`](../persona_policies/) in this repo
- `LLaMA-Factory` — trainer: https://github.com/hiyouga/LLaMA-Factory

Python deps for the scripts in this folder:

```bash
pip install -r requirements.txt
```

Set in your environment:

- `TAU2_BENCH_ROOT` — path to the cloned `tau2-bench` repo
- `LLAMA_FACTORY_DIR` — path to your LLaMA-Factory checkout
- `OPENROUTER_API_KEY` (and/or AWS Bedrock keys) for hosted user/agent LLMs
- `HF_TOKEN` for Gemma weights

## Pipeline

### 1. Collect raw trajectories

Default user (baseline SFT data):

```bash
python data/collect_baseline_trajectories.py \
  --domain retail --split train \
  --agent-model openrouter/google/gemma-4-31b-it \
  --user-model openrouter/google/gemma-4-31b-it \
  --output data/raw/baseline_retail_train.jsonl
```

PPol persona-conditioned user (requires an evolved persona generator program):

```bash
python data/collect_persona_trajectories.py \
  --domain retail --split train \
  --program-path /path/to/evolved/program.py \
  --n-personas 10 \
  --agent-model openrouter/google/gemma-4-31b-it \
  --user-model openrouter/google/gemma-4-31b-it \
  --output data/raw/ppol_retail_train.jsonl
```

Outputs are JSONL, one episode per line with the full `sim.get_messages()` trace (tool calls preserved) and reward.

### 2. Build ShareGPT SFT data

Filters to successful (`reward >= 1.0`) episodes, splits parallel tool calls, drops `<thinking>` blocks, writes `train.json` / `val.json` for LLaMA-Factory.

Default-only:

```bash
python data/build_sharegpt_sft_data.py \
  --inputs data/raw/baseline_retail_train.jsonl \
  --domain retail \
  --output-dir data/sft/retail_baseline
```

Default + PPol mixed:

```bash
python data/build_sharegpt_sft_data.py \
  --inputs data/raw/baseline_retail_train.jsonl data/raw/ppol_retail_train.jsonl \
  --domain retail \
  --output-dir data/sft/retail_mixed_baseline_ppol
```

### 3. Train

```bash
python train/train_lora.py configs/retail_baseline_lora.yaml
python train/train_lora.py configs/retail_mixed_persona_lora.yaml
```

`train_lora.py` merges `configs/dataset_info.json` into the LLaMA-Factory `data/dataset_info.json`, then invokes `llamafactory-cli train`. All SFT runs use the same hyperparameters and a fixed step budget (`max_steps`); only `dataset`, `eval_dataset`, `output_dir`, and `max_steps` differ across runs (see `configs/_base_lora.yaml`).

## Evaluation

Test-time scores are produced with the official benchmarks:

- τ²-bench (default cooperative user simulator): `<tau2-bench-repo-url>`
- τ-trait (Skeptical / Incoherent / Impatient / Confusion challenge suites): `<tau-trait-repo-url>`

Serve the trained LoRA with vLLM (`--lora-modules`) and point the official runners at the OpenAI-compatible endpoint as the agent backend.
