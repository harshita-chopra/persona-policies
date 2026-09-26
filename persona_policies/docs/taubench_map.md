# τ²-bench (tau2) codebase map for Persona Policies

Persona Policies inject **behavioral text** into the τ²-bench LLM user simulator while leaving task goals unchanged. τ²-bench is an external dependency — install it from its official repository and set `TAU2_DATA_DIR` to its data directory.

## Key paths (relative to your τ²-bench install root)

| Path | Role |
|------|------|
| `src/tau2/user/user_simulator.py` | `UserSimulator`: builds **system prompt** from global guidelines + scenario; uses `PersonaConfig` for runtime persona (verbosity, etc.). |
| `data/tau2/user_simulator/simulation_guidelines.md` | Global user-simulation rules (text mode). |
| `src/tau2/runner/build.py` | `build_user`, `build_text_orchestrator`: wire agent, user, env, task. |
| `src/tau2/runner/simulation.py` | `run_simulation(orchestrator)` → `SimulationRun` with `reward_info`. |
| `src/tau2/orchestrator/orchestrator.py` | Half-duplex loop; trajectory is a list of `Message`. |
| `src/tau2/data_model/tasks.py` | `Task`, `UserScenario`, `StructuredUserInstructions` (domain, reason_for_call, known_info, task_instructions, …). |
| `data/tau2/domains/<domain>/tasks.json` | Task definitions. |
| `data/tau2/domains/<domain>/split_tasks.json` | Splits: `"train"` / `"test"` task id lists. |
| `src/tau2/runner/helpers.py` | `get_tasks(task_set_name, task_split_name, ...)`. |
| `src/tau2/data_model/persona.py` | `PersonaConfig` (verbosity, interrupt_tendency) → optional extra guidelines. |

## User simulator system prompt (text)

Defined in `UserSimulator.system_prompt` (`user_simulator.py`):

```text
{global_user_sim_guidelines_with_persona}

<scenario>
{instructions}
</scenario>
```

- `instructions` is `str(task.user_scenario)` (persona + structured instructions from JSON).
- Global guidelines are loaded from `simulation_guidelines.md` (and tools/voice variants). `PersonaConfig.to_guidelines_text()` is merged via a `<PERSONA_GUIDELINES>` placeholder in **voice** guideline files; the plain text `simulation_guidelines.md` has **no** placeholder, so runtime `PersonaConfig` only affects voice prompts unless the markdown is extended.

**Persona Policies** therefore subclass `UserSimulator` and override `system_prompt` to append a behavioral block after the full base prompt (see `persona_policies/injector.py`).

## Trajectory format

- **Raw API**: `SimulationRun.messages` is a list of `Message` (`user`, `assistant`, `tool`, …).
- **For fingerprinting**, we normalize to:

```json
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]
```

Tool-only turns are skipped for behavioral fingerprints; user/assistant text turns are kept in order.

**Saved JSON** (e.g. baseline cache) uses the same list-of-dicts shape.

## Reward / success

- `simulation.reward_info.reward` is typically in **\[0, 1\]** (multiplicative over checks). **Success** is treated as `reward >= 1.0` (or `== 1.0` when binary).

## CLI vs programmatic run

- **CLI**: `tau2 run` (see τ²-bench documentation for CLI usage and Bedrock/LiteLLM setup).
- **Programmatic** (used by evaluators):

  1. `ensure_taubench_importable()` from `persona_policies.taubench_setup`.
  2. `get_tasks(domain, task_split_name="train"|"test")` — splits from `split_tasks.json`.
  3. Build `TextRunConfig(domain=..., agent="llm_agent", user="user_simulator", llm_agent=..., llm_user=..., max_steps=..., ...)`.
  4. Construct a `UserSimulator` subclass with optional persona injection; use the same pattern as `build_text_orchestrator` but swap in `PersonaPolicyUserSimulator` (`injector.py`).
  5. `run_simulation(orchestrator)` → read `get_messages()`, `reward_info`.

## Task scenario fields (user side)

From `UserScenario` / `StructuredUserInstructions`:

- **persona** (optional): static user persona string in task JSON.
- **instructions** (`StructuredUserInstructions`): **domain**, **reason_for_call**, **known_info**, **unknown_info**, **task_instructions**.

Persona Policies must **not** replace these; they only add behavioral instructions in the system prompt.

## Train / test splits (retail)

- File: `<tau2-bench-install>/data/tau2/domains/retail/split_tasks.json`.
- **Evolution / training calibration**: use split **`train`** (`PersonaPoliciesConfig.task_split_evolution`).
- **Benchmark / held-out evaluation**: use **`test`** (`task_split_benchmark`).

## Environment

- Set `TAU2_DATA_DIR` to the `data/` directory of your τ²-bench install (handled by `taubench_setup`).
- Bedrock: set `AWS_REGION_NAME`, credentials; LiteLLM model ids `bedrock/...`.
