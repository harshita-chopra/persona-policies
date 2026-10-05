# WildChat (RealUserSim) × ppol

Open-domain chat track. A persona-injected **user simulator** converses with an
AI **assistant**; ppol diversifies the otherwise homogeneous, persona-free user
sim so its behavioral fingerprint matches real WildChat users.

**Data:** [Salesforce/RealUserSim](https://huggingface.co/datasets/Salesforce/RealUserSim)
— 1,200 real WildChat conversations repurposed as a user-simulation benchmark.
**Auto-downloaded** from the HF Hub on first use (needs `pip install huggingface_hub`);
nothing to fetch manually.
Each row's `original_messages` (its `role=="user"` turns) is our **human
reference**; `user_goal` / `problem_desc` + `solution_conditions` drive the sim
and its `/close`; `user_profile.linguistic_profile` is the real user's style.

**Base user-sim prompt** (`prompts/user_simulator_prompt.txt`) is Salesforce's own
`build_user_prompt()` "linguistic mimic" template, with the real `linguistic_profile`
**removed** so the base sim is generic — that removed profile is exactly where ppol
injects a generated persona.

## Conditions
| Condition | User sim |
|---|---|
| Humans | real WildChat user turns (`original_messages`) |
| Base-simulator | linguistic-mimic prompt, **no** real profile (persona-free) |
| DP Personas | direct-prompted personas |
| PPol: Initial / Evolved | seed / evolved persona generator |
| *(oracle)* | full prompt **with** the real `linguistic_profile` (RealUserSim's method) |

## Pipeline
```bash
# 1–4: human reference (user-disjoint split) → baseline → discriminator → evolve
python examples/wildchat/run_pipeline.py --steps reference,baseline,discriminator,evolve \
    --train-pool 400 --val-size 20 --iterations 70

# held-out eval (default/dp/initial/ppol) + scatter
python examples/wildchat/benchmark.py --n-personas 10 --scatter-conditions default,ppol
```

Reward = an LLM judge on `solution_conditions` (was the request addressed?).
Splits are **user-disjoint** (no real human in both train and held-out), mirroring
colbench's repo-disjoint and τ²-bench's annotator-disjoint splits.

Structure mirrors `examples/colbench/`: `wildchat_reference.py` (build human ref),
`runner.py` (`WildChatRunner`), `run_pipeline.py`, `benchmark.py`, `prompts/`,
own `outputs/`.
