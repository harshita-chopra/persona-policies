# Persona Policies

**ppol** is a Python framework for evolving realistic, diverse user-simulator personas for LLM agent benchmarks. It is a plug-and-play overlay that injects behavioral variation into user simulators *without* changing task goals, rewards, or environment state.

```bash
pip install ppol
```

## What it does

LLM-based user simulators (τ²-bench, ColBench, etc.) tend to be overly cooperative and homogeneous — a behavioral gap relative to real users. ppol closes the gap by:

1. **Generating a population of persona policies** — short natural-language overlays appended to the user simulator's system prompt that shape *how* the user communicates (terse, distracted, frustrated, guarded, …) while task facts stay fixed.
2. **Optimizing the persona generator via evolutionary program search** (OpenEvolve), with two objectives:
   - **Human-likeness** — mean `P(human)` from a trained Random Forest discriminator on behavioral fingerprints
   - **Behavioral coverage** — two-sided Chamfer distance against a human reference distribution

The result is a generator `G(c, D, N)` that, given a task context and number of personas, produces N diverse, human-shaped persona policies.

## Quick start

```python
from ppol import DataLoader, PPol, SimpleEpisodeRunner, split_tasks
from ppol.pipeline import collect_baseline, compute_human_reference, train_discriminator

def my_agent(messages):           # any function: messages → reply
    return "How can I help?"

runner = SimpleEpisodeRunner(agent=my_agent, user_sim_model="gpt-4o-mini")
tasks = DataLoader.load_tasks("tasks.json")          # your tasks
train, val, _ = split_tasks(tasks)

dialogs = DataLoader.load_dialogs("human_dialogs.json")   # your real human chats
compute_human_reference(dialogs, "outputs/ref/human.json")
collect_baseline(runner, tasks, "outputs/ref/baseline.json")
train_discriminator("outputs/ref/human.json", "outputs/ref/baseline.json", "outputs/ref/disc.pkl")

p = PPol(output_dir="outputs/my_run")
p.evolve(runner=runner, train_tasks=train, val_tasks=val,
         human_reference_path="outputs/ref/human.json",
         baseline_path="outputs/ref/baseline.json",
         discriminator_path="outputs/ref/disc.pkl",
         iterations=50)
```

See **[ppol/README.md](ppol/README.md)** for the full walkthrough: human reference → baseline → discriminator → evolve → benchmark, plus a custom-runner template and τ²-bench example.

## Repository layout

```
ppol/                       # The pip package — domain-agnostic
├── core/                       # Task, EpisodeResult, EpisodeRunner, SimpleEpisodeRunner
├── data/                       # DataLoader
├── evolution/                  # OpenEvolve fitness + seed generator + templates
├── analysis/                   # library-grade plot helpers
├── config.py                   # PPolConfig
├── discriminator.py            # RF behavioral classifier
├── fingerprinting.py           # 19-feature behavioral fingerprint
├── injection.py                # persona-injection template + helper
├── llm.py                      # LiteLLM completion wrapper
├── pipeline.py                 # PPol orchestrator + pipeline functions
└── README.md                   # user-facing docs

examples/
├── tau2bench/                  # τ²-bench: runner + pipeline + benchmark
├── colbench/                   # SWEET-RL/ColBench backend-programming collaboration
└── wildchat/                   # open-domain chat (RealUserSim / WildChat-derived)

tests/
pyproject.toml
```

Each `examples/<name>/` follows the same pattern: a self-contained `EpisodeRunner` subclass plus its own driver and any domain-specific data/scripts. They are *not* part of the installed wheel — use them by running their scripts directly (`python examples/<name>/run_pipeline.py`) or by adding the directory to `PYTHONPATH`.

**Data:** nothing to fetch manually. τ²-bench tasks come with the `tau2-bench` install; all human-reference datasets auto-download from the Hugging Face Hub on first use — τ² human dialogues (`cmu-lti/tau-usi`), ColBench (`facebook/collaborative_agent_bench` + `SALT-NLP/SWE-chat`), WildChat (`Salesforce/RealUserSim`).

## Running τ²-bench

```bash
python examples/tau2bench/run_pipeline.py --iterations 200 --domain retail
```

See [`examples/tau2bench/README.md`](examples/tau2bench/README.md) for individual steps and configuration. For evolution + benchmarking with a custom dataset, use the Python API in [`ppol/README.md`](ppol/README.md).

## Citation

```bibtex
@article{chopra2026persona,
  title     = {Beyond Cooperative Simulators: Generating Realistic User Personas for Robust Evaluation of LLM Agents},
  author    = {Chopra, Harshita and Ghate, Kshitish and Caliskan, Aylin and Kohno, Tadayoshi and Shah, Chirag and Jaques, Natasha},
  journal   = {arXiv preprint arXiv:2605.12894},
  year      = {2026},
}
```
