# ppol — Persona Policies

Evolve and evaluate structured behavioral personas for dialogue agents. A persona is a text overlay injected into a user simulator's system prompt that shapes *how* it communicates — terse, skeptical, frustrated, ambiguous — without changing its underlying task or goal.

---

## Install

```bash
pip install ppol
```

For the τ²-bench example:

```bash
pip install git+https://github.com/sierra-research/tau2-bench
```

Set your API key (any LiteLLM-compatible provider):

```bash
export OPENROUTER_API_KEY=...   # recommended — covers most models
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

---

## Choosing an LLM provider

Every LLM call in `ppol` goes through `litellm.completion` (see `ppol/llm.py`), so
you can use **any [LiteLLM](https://docs.litellm.ai/docs/providers)-supported
provider** — OpenAI, Anthropic, Gemini, Bedrock, Azure, Vertex AI, OpenRouter,
or a local Ollama/vLLM endpoint. You select a provider purely by the **model-id
string** and that provider's API key; no code changes are needed.

| Provider   | Example model id                                   | Key needed          |
|------------|----------------------------------------------------|---------------------|
| OpenAI     | `gpt-4o`                                           | `OPENAI_API_KEY`    |
| Anthropic  | `anthropic/claude-sonnet-5`                        | `ANTHROPIC_API_KEY` |
| Gemini     | `gemini/gemini-3-flash`                            | `GEMINI_API_KEY`    |
| OpenRouter | `openrouter/google/gemini-3-flash-preview`         | `OPENROUTER_API_KEY`|
| Bedrock    | `bedrock/anthropic.claude-3-haiku-20240307-v1:0`   | AWS credentials     |
| Local      | `ollama/llama3`                                    | none                |

The package **defaults to OpenRouter** model ids (see `PPolConfig` in
`ppol/config.py`). To point each stage at a different provider:

| Stage | How to set the model |
|---|---|
| **Episodes** (your own runner) | Pass the model string to your runner, e.g. `SimpleEpisodeRunner(user_sim_model="gpt-4o")`. |
| **Episodes** (τ²-bench runner) | Env vars `PERSONA_POLICIES_TAUBENCH_USER_MODEL` / `PERSONA_POLICIES_TAUBENCH_AGENT_MODEL`. |
| **Evaluator reflection** | `PPol.evolve(..., reflection_model="gpt-4o")`. |
| **Persona generator + fallbacks** | Defaults `llm_model` / `llm_fallback_models` in `PPolConfig` (`ppol/config.py`); edit there to change globally. |
| **OpenEvolve mutator** | Edit `ppol/evolution/openevolve_config.yaml`. For non-OpenAI providers also run the LiteLLM hook once per venv so worker subprocesses route correctly: `python ppol/evolution/install_openevolve_litellm_pth.py` |

> **Note on retries.** `completion_text` applies exponential-backoff retries and
> provider failover **only for `openrouter/...` model ids**. Other providers make a
> single attempt per call — wrap your own retry logic if you need resilience on
> transient 429/5xx errors, or route through OpenRouter.

---

## How it works

```
Human traces  +  Baseline rollouts (no persona)
          │
          ▼
  Train discriminator  ← RF: can it tell human from simulator?
          │
          ▼
  Evolve persona generator  ← maximize P(human) + behavioral coverage
          │
          ▼
  Generate personas per task → run episodes, track performance
```

**Human traces** — real human dialogues that define what "human-like" means.  
**Baseline rollouts** — episodes without any persona; the "not human-like" anchor.  
**Discriminator** — RF trained on human vs. baseline fingerprints. During evolution, each candidate persona is scored by P(human) from this classifier.  
**Evolved generator G(c, D, N)** — given a task context, behavioral axes, and count N, returns N diverse persona strings.

End-to-end working examples live in [`examples/`](https://github.com/harshita-chopra/persona-policies/tree/main/examples) — e.g. `examples/tau2bench/` runs the full pipeline on τ²-bench retail + airline tasks with a human-annotator reference.

---

## Full walkthrough

### Step 0 — Split tasks (recommended)

```python
from ppol import split_tasks

runner = MyRunner(agent=my_agent)
all_tasks = runner.get_tasks()

train_tasks, val_tasks, test_tasks = split_tasks(all_tasks)
# Default: test_size=0.2, val_size=0.2 → 0.64 / 0.16 / 0.20
```

`split_tasks` does a two-step split (test held out first, then val carved out of the remaining train pool). Bring your own splits if you have them — just pass the lists directly.

### Step 1 — Fingerprint your human traces

```python
from ppol import DataLoader
from ppol.pipeline import compute_human_reference

dialogs = DataLoader.load_dialogs("human_dialogs.json")
compute_human_reference(dialogs, "outputs/reference_data/human_fingerprints.json", domain="custom")
```

This writes two files:

- `human_fingerprints.json` — distribution (mean + std per feature)
- `human_fingerprints.individual.json` — per-dialog fingerprints (required for Chamfer behavioral coverage during evolution)

### Step 2 — Collect baseline episodes

Run your agent on tasks **without any persona** to establish the "not human-like" anchor. Baseline is just material for the discriminator's negative class — it isn't tied to the train/val/test split:

```python
from ppol.pipeline import collect_baseline

collect_baseline(runner, all_tasks, "outputs/reference_data/baseline.json", n_workers=4)
```

A healthy baseline has a variety of episode lengths and outcomes — it should look different from your human traces.

### Step 3 — Train the discriminator

```python
from ppol.pipeline import train_discriminator

train_discriminator(
    human_reference_path="outputs/reference_data/human_fingerprints.json",
    baseline_fingerprints_path="outputs/reference_data/baseline.json",
    output_path="outputs/reference_data/discriminator.pkl",
)
```

A healthy discriminator gets ROC-AUC > 0.7. Below 0.6 means human and baseline look too similar — collect more or better human traces.

### Step 4 — Evolve the persona generator

```python
from ppol import PPol

p = PPol(output_dir="outputs/my_run/")
p.evolve(
    runner=runner,
    runner_kwargs={"data_path": "..."},     # init args for runner class
    train_tasks=train_tasks,
    val_tasks=val_tasks,                    # used for val_curve.jsonl monitoring
    human_reference_path="outputs/reference_data/human_fingerprints.json",
    baseline_path="outputs/reference_data/baseline.json",
    discriminator_path="outputs/reference_data/discriminator.pkl",
    iterations=50,
    batch_size=3,                           # train tasks sampled per iteration
    n_personas=5,                           # personas per task (final, after curriculum)
    n_workers=4,
)
```

`evolve` writes a JSON config that `ppol/evolution/fitness.py` reads via the `PPOL_RUN_CONFIG` env var, then invokes OpenEvolve. The fitness is fully domain-agnostic: persona generator is mutated to maximise

```
combined_score = λ_h · human_likeness + λ_b · intra_set_diversity
```

where `human_likeness` is the mean `P(human)` from the trained discriminator over persona-episode fingerprints and `intra_set_diversity` is two-sided Chamfer coverage against the human cloud.

**What runs each iteration:**

- **Sliding train batch** — deterministic walk through `train_tasks` (full sweep = 1 epoch).
- **Curriculum** — `n_personas_schedule` (default `[(1,5),(2,8),(3,10)]`) grows N as epochs progress; scoring weights ramp diversity pressure with current N.
- **LLM reflection** — `reflection_model` (default Gemini Flash) writes a 300-word reflection on the iteration's metrics + best/worst persona dialogues; returned as an OpenEvolve artifact so the mutator gets domain feedback.
- **Train curve** — `training/results/train_curve.jsonl` appended.
- **Val curve** — daemon thread fires whenever the elite changes; runs the new best on `val_tasks` and appends `training/results/val_curve.jsonl`. Non-blocking.
- **`evolution_scores.png`** — refreshed after every train and every val.
- **Per-iter log** — `training/simulations/iter_NNNN/{log.json, reflection.txt}`.
- **Early stop** — if no new elite for `early_stop_iters_without_new_best` iterations (default 20), signals SIGTERM to OpenEvolve.

The best program is saved to:

```
outputs/my_run/openevolve/best/best_program.py
```

Resume a stopped run:

```python
p.evolve(..., resume=True)
```

**Useful `evolve()` knobs:**

| Argument | Default | What it controls |
|---|---|---|
| `iterations` | 70 | OpenEvolve iterations (200+ to converge) |
| `batch_size` | 5 | Train tasks sampled per iteration |
| `n_personas` | 10 | Final personas-per-task (after curriculum ramps to it) |
| `curriculum` | `True` | Whether `n_personas_schedule` is applied |
| `n_personas_schedule` | `[(1,5),(2,8),(3,10)]` | `[(epoch_threshold, N), ...]` |
| `lambda_human_likeness` | 0.5 | Final scoring weight on `P(human)` |
| `lambda_intra_diversity` | 0.5 | Final scoring weight on Chamfer coverage |
| `reflection_model` | `openrouter/google/gemini-3-flash-preview` | LLM for evaluator reflection |
| `early_stop_iters_without_new_best` | 20 | Stop after this many stagnant iterations |
| `n_workers` | 4 | Parallel episode workers per fitness call |

Tune OpenEvolve hyperparameters (population size, island count, mutator LLM, temperature) in `ppol/evolution/openevolve_config.yaml`.

### Step 5 — Generate personas and run episodes

`p.generate()` calls the evolved generator (auto-loaded from `<output_dir>/openevolve/best/best_program.py`) with a task context and N, returning N persona dicts. The `"text"` field is what you pass to `run_episode`.

```python
personas = p.generate(task_context=val_tasks[0].description, n=5)

for persona in personas:
    print(persona["persona_id"])       # e.g. "frustrated_skeptic"
    print(persona["description"])      # who this person is
    print(persona["axis_placement"])   # {"terse": True, "skeptical": True, ...}
    print(persona["text"])             # full persona string → pass to run_episode

    result = runner.run_episode(val_tasks[0], persona_policy=persona["text"])
    print(result.success, result.reward, result.n_turns)
    # result.trajectory contains the full conversation
```

Generate personas for all tasks at once:

```python
all_items = p.generate(tasks=val_tasks, n=3)

for item in all_items:
    task = item["task"]
    for persona in item["personas"]:
        result = runner.run_episode(task, persona_policy=persona["text"])
```

Pass `best_program=` to use a specific evolved file instead of auto-detecting:

```python
personas = p.generate(task_context="...", n=5, best_program="path/to/best_program.py")
```

---

## Score a persona against the discriminator

To measure how human-like a persona string is (P(human) from the trained discriminator):

```python
from ppol.pipeline import benchmark_policy

metrics = benchmark_policy(
    runner=runner,
    tasks=tasks[:20],
    persona_policy_text=personas[0]["text"],
    human_reference_path="outputs/reference_data/human_fingerprints.json",
    discriminator_path="outputs/reference_data/discriminator.pkl",
    output_dir="outputs/results/",   # optional — saves metrics.json + trajectories
)
print(metrics["human_likeness"])
print(metrics["combined_score"])
```

---

## Using τ²-bench

τ²-bench is shipped as a worked example, not as a built-in runner. The runner
class and τ² glue live in [`examples/tau2bench/`](https://github.com/harshita-chopra/persona-policies/tree/main/examples/tau2bench) —
treat that directory as the τ² integration:

```python
import sys; sys.path.insert(0, "examples/tau2bench")
from runner import Tau2BenchRunner
from tau_reference import compute_tau2bench_human_reference

from ppol import PPol, split_tasks
from ppol.pipeline import collect_baseline, train_discriminator

# Step 0: setup + splits
runner = Tau2BenchRunner()
train_tasks, val_tasks, _ = split_tasks(runner.get_tasks())

# Step 1: human reference — uses bundled tau2bench human logs
compute_tau2bench_human_reference(
    "outputs/reference_data/human_fingerprints_retail.json",
    domain="retail",   # or "airline"
)

# Step 2: baseline episodes
collect_baseline(
    runner, runner.get_tasks(),
    "outputs/reference_data/baseline_retail.json",
    n_workers=4,
)

# Step 3: discriminator
train_discriminator(
    human_reference_path="outputs/reference_data/human_fingerprints_retail.json",
    baseline_fingerprints_path="outputs/reference_data/baseline_retail.json",
    output_path="outputs/reference_data/discriminator_retail.pkl",
)

# Step 4: evolve
p = PPol(output_dir="outputs/retail/")
p.evolve(
    runner=runner,
    train_tasks=train_tasks,
    val_tasks=val_tasks,
    human_reference_path="outputs/reference_data/human_fingerprints_retail.json",
    baseline_path="outputs/reference_data/baseline_retail.json",
    discriminator_path="outputs/reference_data/discriminator_retail.pkl",
    iterations=200,
)

# Step 5: generate + run
for item in p.generate(tasks=val_tasks[:5], n=3):
    for persona in item["personas"]:
        result = runner.run_episode(item["task"], persona_policy=persona["text"])
        print(persona["persona_id"], result.reward)
```

---

## Implement your own runner

```python
from ppol import EpisodeRunner, EpisodeResult, Task, inject_persona_into_system_prompt
from typing import List

class MyRunner(EpisodeRunner):

    def __init__(self, agent):
        self.agent = agent

    def get_tasks(self) -> List[Task]:
        from ppol import DataLoader
        return DataLoader.load_tasks("tasks.json")

    def run_episode(self, task: Task, persona_policy: str = "") -> EpisodeResult:
        user_system = inject_persona_into_system_prompt(
            original_system_prompt=f"You are a customer. Your goal: {task.description}",
            persona_policy_text=persona_policy,
        )
        trajectory = []
        # ... your turn loop: user sim → agent → user sim → ...
        return EpisodeResult(
            task_id=task.task_id,
            trajectory=trajectory,
            success=True,
            reward=1.0,
            persona_policy=persona_policy,
        )
```

For a simple LLM-based user sim without a custom environment, use `SimpleEpisodeRunner`:

```python
from ppol import SimpleEpisodeRunner

runner = SimpleEpisodeRunner(
    agent=my_agent,
    user_sim_model="gpt-4o-mini",   # any LiteLLM model string
)
```

**Supported model strings:**

| Provider   | Example                                              | Env var                |
|------------|------------------------------------------------------|------------------------|
| OpenAI     | `"gpt-4o-mini"` / `"gpt-4o"`                        | `OPENAI_API_KEY`       |
| Anthropic  | `"claude-haiku-4-5-20251001"` / `"claude-sonnet-4-6"` | `ANTHROPIC_API_KEY`  |
| OpenRouter | `"openrouter/google/gemini-2.0-flash-lite"`          | `OPENROUTER_API_KEY`   |
| Google     | `"gemini/gemini-2.0-flash"`                          | `GEMINI_API_KEY`       |
| Bedrock    | `"bedrock/anthropic.claude-3-haiku-20240307-v1:0"`   | AWS credentials        |

---

## Data formats

**`tasks.json`:**
```json
[
  {
    "task_id": "t001",
    "description": "Return a defective laptop purchased 3 weeks ago.",
    "context": "Customer service — electronics retail."
  }
]
```

**`human_dialogs.json`** — real human conversations:
```json
[
  {
    "conversation": [
      {"role": "user",      "content": "Hi, I need to return something."},
      {"role": "assistant", "content": "Of course! What's the issue?"},
      {"role": "user",      "content": "Screen cracked after one day."}
    ],
    "metadata": {}
  }
]
```

Plain-text format also accepted — conversations separated by `---`, turns prefixed `USER:` / `ASSISTANT:`.
