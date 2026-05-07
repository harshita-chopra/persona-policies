# Persona Policies

Persona Policies is a small framework on top of `tau2-bench` for evolving user-simulator behavior prompts. It does not change tau2 task goals, private facts, rewards, or tools. It only appends a persona policy block to the tau2 user simulator system prompt so simulated customers behave in more varied and human-like ways.

The main evolved file is:

```text
persona_policies/evolution/initial_generator.py
```

OpenEvolve mutates that file. Fitness is computed by running tau2 episodes and scoring the resulting dialogues with two objectives:

- **Human-likeness** (HL): mean `P(human)` from the model-specific discriminator under `outputs/reference_data/`
- **Behavioral coverage** (B_cover): per-task Chamfer distance between generated persona fingerprints and the train-split human reference cloud, normalized to [0, 1]
- **Combined score**: `score = lambda_h * HL + lambda_b * B_cover`, where training uses curriculum-adjusted weights and validation/benchmark use the configured weights directly

## How It Works

`initial_generator.py` implements `G(c, D, N)`:

- `c`: formatted tau2 user scenario for one task
- `D`: `DIVERSITY_AXES` — the list of behavioral axes
- `N`: number of personas to generate

Generation is two-stage:

1. `generate_population(...)` jointly proposes `N` population members with axis placements.
2. `expand_personas_parallel(...)` expands each member into full roleplay instructions using one single-person Stage-2 LLM call per member, parallelized within the task.

During OpenEvolve training, `fitness.py`:

1. Splits official tau2 `train` tasks into an internal train slice and held-out val slice using `val_fraction`.
2. Samples a random train minibatch each iteration.
3. Chooses the training persona count from curriculum.
4. Generates personas for each task in the minibatch.
5. Runs tau2 rollouts for every `(task, persona)` pair.
6. Applies curriculum-aware scoring weights.
7. Builds `evaluator_reflection` from scalar metrics, sampled high/low dialogue excerpts, persona text, compact fingerprints, task context, and per-persona human-likeness summaries.
8. Returns numeric metrics plus the reflection artifact to OpenEvolve.
9. Writes train logs under `outputs/training_<version>/simulations/iter_NNNN/`.

OpenEvolve then mutates `initial_generator.py` using:

- static guidance from `evolution/templates/system_message.txt`
- dynamic feedback from the `evaluator_reflection` artifact
- one sampled `{mutation_angle}` inserted into `evolution/templates/diff_user.txt`

Mutation angles are defined in `prompt.template_variations.mutation_angle` in `openevolve_config.yaml`. They nudge each mutation toward a lever like revising `DIVERSITY_AXES`, changing population prompts, changing roleplay-expansion prompts, adding conditional expansion logic, or trying a new generator idea.

Curriculum changes only training batches. With the current default schedule, epoch 1 uses `N=5`, epoch 2 uses `N=8`, and epoch 3+ uses `N=10`. Validation and benchmark always use full `n_personas`.

The behavioral coverage weight (λ_b) is ramped with the current training `N`:

```text
ratio = current_train_N / final_n_personas
lambda_b_train = lambda_intra_diversity * ratio   # lambda_intra_diversity = lambda_b (behavioral coverage weight)
lambda_h_train = 1 - lambda_b_train
```

So early curriculum batches put less pressure on coverage while there are fewer personas. Validation and benchmark use the configured weights directly.

Validation is monitor-only: when the on-disk elite changes, a full validation run is launched on the held-out val slice and logged under `outputs/training_<version>/validation/`. Validation does not produce reflection, mutation, or training feedback.

Reflection intentionally excludes diagnostic Dice dimensions and the per-feature human/persona comparison table so the mutator does not optimize toward brittle proxy features like word count, capitalization, or punctuation artifacts.

## Setup

Run commands from the repo root:

```bash
conda env create -f environment.yml
conda activate persona-policies
```

**τ²-bench is not included in this repository.** Install it from its official source before running any commands:

```bash
git clone <tau2-bench-repo-url>
pip install -e ./tau2-bench
```

If needed, install Persona Policies dependencies manually:

```bash
pip install -r persona_policies/requirements.txt
pip install pytest
```

Set API credentials for your LLM provider(s) before running:

```bash
export OPENROUTER_API_KEY=...
export AWS_REGION_NAME=us-west-2  # if using Bedrock models
```

Default models are set in `persona_policies/config.py` and `persona_policies/evolution/openevolve_config.yaml`:

- generator/reflection: `openrouter/google/gemini-3-flash-preview`
- tau2 user simulator: `openrouter/qwen/qwen3-next-80b-a3b-instruct`
- tau2 agent/env/NL assertions/fallback: `openrouter/google/gemma-4-31b-it`

Run once per environment if OpenEvolve is not routing through LiteLLM:

```bash
python persona_policies/evolution/install_openevolve_litellm_pth.py
```

## Current Important Defaults

From `PersonaPoliciesConfig`:

| Setting | Current default |
|---|---|
| `taubench_domain` | `retail` |
| `task_split_evolution` | `train` |
| `task_split_benchmark` | `test` |
| `baseline_collect_split` | `all` |
| `taubench_user_model` | `openrouter/qwen/qwen3-next-80b-a3b-instruct` |
| `val_fraction` | `0.2` |
| `n_personas` | `10` |
| `eval_batch_size` | `5` |
| `parallel_episode_workers` | `20` |
| `curriculum` | `True` |
| `n_personas_schedule` | `[(1, 5), (2, 8), (3, 10)]` |
| `lambda_human_likeness` | `0.5` (λ_h) |
| `lambda_intra_diversity` | `0.5` (λ_b, behavioral coverage weight) |
| `max_turns_per_episode` | `30` for validation/benchmark |
| `train_max_steps_per_episode` | `20` for training |
| `version` | `v1`, so outputs go under `outputs/training_v1/` |

`train_max_steps_per_episode` and `max_turns_per_episode` both cap tau2 orchestrator steps, not clean human/assistant turns. User, assistant, and tool/environment steps all count.

From `openevolve_config.yaml`:

| Setting | Current default |
|---|---|
| `checkpoint_interval` | `1` |
| `evaluator.parallel_evaluations` | `1` |
| `database.population_size` | `40` |
| `database.num_islands` | `5` |
| `database.embedding_model` | `null`, so novelty embedding checks are off |
| `database.log_prompts` | `false` |
| `evolution_trace.enabled` | `false` |

## Files To Run

### 1. Collect Baseline Tau2 Episodes

```bash
python persona_policies/scripts/collect_baseline.py --n 100
```

Useful args:

| Arg | Meaning |
|---|---|
| `--n` | number of baseline episodes; default is one per task in selected split(s) |
| `--domain DOMAIN` | tau2 domain; default `taubench_domain` |
| `--collect-split {train,test,all}` | overrides `baseline_collect_split` |
| `--workers N` | rollout workers; default `parallel_episode_workers` |
| `--save-every N` | periodically write snapshots; default `10` |
| `--force` | regenerate even if cached baseline files exist |

Does:

- runs tau2 with the default user simulator and no persona injection
- computes fingerprints for the resulting trajectories
- records baseline success rate

Outputs under `persona_policies/outputs/reference_data/`:

```text
baseline_<domain>_<split>_<agent_model>_user_<user_model>_results.json
baseline_<domain>_<split>_<agent_model>_user_<user_model>_fingerprints.json
baseline_<domain>_<split>_<agent_model>_user_<user_model>_trajectories.json
```


### 2. Train Human-Likeness Discriminator

```bash
python persona_policies/scripts/train_discriminator.py
```

Useful args:

| Arg | Meaning |
|---|---|
| `--domain DOMAIN` | τ² domain filter (default: `config.taubench_domain`) |

Does:

- computes human fingerprints
- trains a RandomForest discriminator on human vs baseline-simulator fingerprints
- saves the model used by evolution fitness

Inputs:

- baseline fingerprints from step 1
- `persona_policies/data/tau_bench_human.json` by default

Outputs under `persona_policies/outputs/reference_data/`:

```text
human_fingerprints_<domain>.json
discriminator_<domain>_<agent_model>_user_<user_model>.pkl
```

### 3. Run OpenEvolve

```bash
python persona_policies/evolution/run_evolution.py --iterations 70
```

Useful args:

| Arg | Meaning |
|---|---|
| `--iterations N` | OpenEvolve iterations (default `70`) |
| `--resume` | resume from latest checkpoint under the run output dir |
| `--version NAME` | output folder becomes `outputs/training_<NAME>/` |
| `--domain DOMAIN` | tau2 domain override propagated to evaluator subprocesses; if no `--version` is given, outputs go to `training_<domain>/` |
| `--verbose` | pass DEBUG logging to OpenEvolve |
| `--log-level LEVEL` | explicit OpenEvolve log level |

Does:

- launches OpenEvolve through `persona_policies.evolution.openevolve_entry`
- mutates `evolution/initial_generator.py`
- evaluates candidates with `evolution/fitness.py`
- uses random train minibatches for fitness
- runs full validation only when the on-disk elite changes

Inputs:

```text
persona_policies/evolution/initial_generator.py
persona_policies/evolution/fitness.py
persona_policies/evolution/openevolve_config.yaml
persona_policies/outputs/reference_data/discriminator_<domain>_<agent_model>_user_<user_model>.pkl
persona_policies/outputs/reference_data/human_fingerprints_<domain>.json
baseline JSONs from step 1
```

Outputs under `persona_policies/outputs/training_<version>/`:

```text
openevolve/checkpoints/
openevolve/best/best_program.py
openevolve/logs/
simulations/iter_NNNN/
validation/iter_NNNN/
results/train_curve.jsonl
results/val_curve.jsonl
results/evolution_best.json
```

OpenEvolve resume uses `openevolve/checkpoints/`. The `simulations/` and `validation/` folders are logs only.

For airline from scratch:

```bash
python persona_policies/scripts/compute_human_reference.py --domain airline
python persona_policies/scripts/collect_baseline.py --domain airline --collect-split all --force
python persona_policies/scripts/train_discriminator.py --domain airline
python persona_policies/evolution/run_evolution.py --domain airline --iterations 70
```

### 4. Benchmark Best Program On Test Split

```bash
python -m persona_policies.benchmark \
  --best-program persona_policies/outputs/training_v1/openevolve/best/best_program.py
```

For the evolved OpenEvolve run, prefer selecting the checkpoint with the best validation average over `N=5,8,10`:

```bash
python -m persona_policies.benchmark --best-by-val-avg --n-personas-list 5,8,10
```

This requires `val_curve.jsonl` plus `val_n_personas_sweep.jsonl`. Create the sweep first if needed:

```bash
python -m persona_policies.benchmark --val-sweep --n-personas-list 5,8,10
```

Built-in baselines:

```bash
python -m persona_policies.benchmark --baseline no_persona
python -m persona_policies.benchmark --baseline seed_initial_generator
python -m persona_policies.benchmark --baseline direct_llm_personas
```

These run on `task_split_benchmark`, which defaults to the official tau2 `test` split.

Baseline meanings:

- `no_persona`: tau2 default user simulator, no persona policy injection.
- `seed_initial_generator`: unevolved `persona_policies/evolution/initial_generator.py`.
- `direct_llm_personas`: one direct LLM call per task to generate `N` full persona instructions, without the axis/Stage-2 pipeline.

Useful args:

| Arg | Meaning |
|---|---|
| `--best-program PATH` | required path to `best_program.py` |
| `--best-by-val-avg` | benchmark the checkpoint whose mean val score over `--n-personas-list` is highest |
| `--baseline {no_persona,seed_initial_generator,direct_llm_personas}` | run a built-in baseline instead of `--best-program` |
| `--n-tasks N` | limit test tasks; default full test split |
| `--n-personas N` | override `config.n_personas` |
| `--domain DOMAIN` | override `taubench_domain` |
| `--out-name NAME` | output subdirectory under `outputs/testing/` |

Does:

- loads evolved `best_program.py` or one built-in baseline
- runs tau2 benchmark split tasks, defaulting to the official `test` split
- computes the same core metrics as training
- writes plots and logs for held-out reporting

Outputs under `persona_policies/outputs/testing/<stem>/`:

```text
log.json
trajectories.json
fingerprints.json
summary.txt
feature_bars.png
human_likeness_bars.png
fingerprint_scatter.png
```

## Output Layout

```text
persona_policies/outputs/
├── reference_data/
│   ├── baseline_*_results.json
│   ├── baseline_*_fingerprints.json
│   ├── baseline_*_trajectories.json
│   ├── human_fingerprints_<domain>.json
│   └── discriminator_<domain>_<agent_model>_user_<user_model>.pkl
├── training_<version>/
│   ├── openevolve/
│   ├── simulations/
│   ├── validation/
│   └── results/
└── testing/
```

## Troubleshooting

- `ModuleNotFoundError: persona_policies`: run commands from repo root.
- `ModuleNotFoundError: tau2`: install τ²-bench from its official repository (`pip install -e /path/to/tau2-bench`).
- OpenEvolve mutation not using LiteLLM: run `python persona_policies/evolution/install_openevolve_litellm_pth.py`.
- Missing discriminator: run baseline collection, then `train_discriminator.py`.
- Slow runs: reduce `--n`, `--n-tasks`, `eval_batch_size`, or workers while debugging.
