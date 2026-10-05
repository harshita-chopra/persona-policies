"""RealUserSim (WildChat) EpisodeRunner for ppol — open-domain chat track.

Salesforce/RealUserSim (WildChat-derived) pairs an AI *assistant* with a
*simulated user* over a multi-turn conversation. The simulated user is grounded
in a real WildChat request (``user_goal`` / ``problem_desc``) and closes the chat
(``/close``) once it judges the request satisfied against private
``solution_conditions``. Success = an LLM judge finds the request addressed.

The simulated user is exactly the party ppol overlays: it produces the
``role=="user"`` turns. RealUserSim's own base sim is a "linguistic mimic"
(prompts/user_simulator_prompt.txt); ppol injects a generated persona into that
prompt and leaves the task facts (request, solution conditions, reward) untouched.

Conditions this runner supports via ``persona_policy``:
  * ""                     -> Base-simulator (persona-free linguistic mimic)
  * a generated persona    -> DP / PPol conditions (ppol's injection point)
  * the real linguistic profile -> Profile-grounded oracle (RealUserSim's method)

Data: Salesforce/RealUserSim on the HF Hub (single ``test`` split, 1200 rows),
partitioned user-disjointly into train/test tasks to match the human reference.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from ppol import EpisodeResult, EpisodeRunner, Task, inject_persona_into_system_prompt
from ppol.config import default_config

_cfg = default_config()
_DEFAULT_AGENT_MODEL     = _cfg.taubench_agent_model   # the AI assistant under test
_DEFAULT_USER_SIM_MODEL  = _cfg.taubench_user_model    # the persona-injected user

_HF_REPO = "Salesforce/RealUserSim"
_HERE = Path(__file__).resolve().parent
_PROMPT_DIR = _HERE / "prompts"

# Ensure sibling modules (wildchat_reference) import inside openevolve worker
# processes, which don't inherit this dir on sys.path.
import sys as _sys
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))

_USER_RESPONSE_CHARACTER_LIMIT = 1000   # WildChat user turns run longer than ColBench
_CLOSE_MARKER = "/close"
_ASSISTANT_GREETING = "Hi! How can I help you today?"


class WildChatRunner(EpisodeRunner):
    """ppol EpisodeRunner over RealUserSim open-domain chat tasks.

    The persona-injected user simulator converses with a fixed assistant; reward
    = an LLM judge's verdict on whether the request was addressed."""

    def __init__(
        self,
        *,
        split: str = "test",
        num_tasks: int = 200,
        seed: int = 0,
        train_frac: float = 0.75,
        agent_model: str = _DEFAULT_AGENT_MODEL,
        human_sim_model: str = _DEFAULT_USER_SIM_MODEL,
        judge_model: Optional[str] = None,
        judge_reward: bool = False,          # ppol scores behavioral fingerprints, not
                                             # task success — off by default so simulation
                                             # doesn't pay a judge LLM call per episode
        max_steps: int = 12,
        agent_temperature: float = 1.0,
        human_sim_temperature: float = 0.0,
        agent_max_tokens: int = 1024,
        human_sim_max_tokens: int = 1024,
    ) -> None:
        self.split                 = split
        self.num_tasks             = num_tasks
        self.seed                  = seed
        self.train_frac            = train_frac
        self.agent_model           = agent_model
        self.human_sim_model       = human_sim_model
        self.judge_model           = judge_model or agent_model
        self.judge_reward          = judge_reward
        self.max_steps             = max_steps
        self.agent_temperature     = agent_temperature
        self.human_sim_temperature = human_sim_temperature
        self.agent_max_tokens      = agent_max_tokens
        self.human_sim_max_tokens  = human_sim_max_tokens
        self._records: Optional[List[dict]] = None
        self._user_prompt = (_PROMPT_DIR / "user_simulator_prompt.txt").read_text()
        self._agent_prompt = (_PROMPT_DIR / "assistant_prompt.txt").read_text()

    # -- data ---------------------------------------------------------------

    def _all_records(self) -> List[dict]:
        if self._records is None:
            from wildchat_reference import load_realusersim_rows
            self._records = load_realusersim_rows()
        return self._records

    def _indices(self) -> List[int]:
        """Deterministic sample of ``num_tasks`` row indices from the requested
        user-disjoint split (train/test), keyed by seed so any reconstruction
        with the same (split, num_tasks, seed) gets identical ids."""
        import random
        from wildchat_reference import canonical_user_split
        recs = self._all_records()
        side = canonical_user_split(recs, self.train_frac)
        pool = [i for i, r in enumerate(recs) if side[str(r["user_ip"])] == self.split]
        if not self.num_tasks or self.num_tasks >= len(pool):
            return sorted(pool)
        return sorted(random.Random(self.seed).sample(pool, self.num_tasks))

    def get_tasks(self) -> List[Task]:
        """Tasks framed from the *user's* perspective, so the persona generator
        produces user personas (not assistant personas)."""
        ctx = ("Open-domain chat: a human user works with an AI assistant to get "
               "help with a real request, and ends the chat once satisfied.")
        recs = self._all_records()
        tasks = []
        for i in self._indices():
            r = recs[i]
            goal = r.get("user_goal") or r.get("problem_desc") or r.get("task_description") or ""
            tasks.append(Task(
                task_id=f"wildchat_{self.split}_{i:05d}",
                description=f"You are a human user chatting with an AI assistant. Your request:\n{goal}",
                context=ctx,
                metadata={
                    "record_index": i,
                    "problem_description": goal,
                    "solution_conditions": r.get("solution_conditions") or r.get("key_context") or "",
                    "domain": r.get("domain", ""),
                    "task_type": r.get("task_type", ""),
                    "linguistic_profile": (r.get("user_profile") or {}).get("linguistic_profile", ""),
                },
            ))
        return tasks

    # -- user-sim chat formatting (roles flipped: sim's turns are 'assistant') --

    def _user_system(self, rec_meta: dict, persona_policy: str) -> str:
        base = self._user_prompt.format(
            problem_description=rec_meta["problem_description"],
            solution_conditions=rec_meta["solution_conditions"],
        )
        if persona_policy.strip():
            base = inject_persona_into_system_prompt(base, persona_policy)
        return base

    def _user_messages(self, rec_meta: dict, dialogue: list, persona_policy: str) -> list:
        """User/persona system prompt + dialogue as alternating chat messages with
        roles flipped (the sim's own turns are ``assistant``, the assistant's are
        ``user``), so a chat-tuned model sees itself as the assistant."""
        messages = [{"role": "system", "content": self._user_system(rec_meta, persona_policy)}]
        for turn in dialogue:
            role = "assistant" if turn["role"] == "user" else "user"
            messages.append({"role": role, "content": turn["content"]})
        if len(messages) > 1 and messages[1]["role"] == "assistant":
            messages.insert(1, {"role": "user", "content": "Hi! How can I help you today?"})
        return messages

    # -- episode ------------------------------------------------------------

    def run_episode(self, task: Task, persona_policy: str = "") -> EpisodeResult:
        import litellm
        meta = task.metadata

        def _llm(model, messages, *, temperature, max_tokens) -> str:
            for attempt in range(3):
                try:
                    out = litellm.completion(
                        model=model, messages=messages,
                        max_tokens=max_tokens, temperature=temperature, timeout=90,
                    ).choices[0].message.content or ""
                    return out.strip()
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(5 * (attempt + 1))
            return ""

        def _gen_user(dialogue: list) -> str:
            reply = _llm(self.human_sim_model, self._user_messages(meta, dialogue, persona_policy),
                         temperature=self.human_sim_temperature, max_tokens=self.human_sim_max_tokens)
            return reply[:_USER_RESPONSE_CHARACTER_LIMIT]

        # Assistant greets; the user sim then states its request in ITS OWN words
        # (a generated, persona-controllable opening) rather than a fixed task string.
        dialogue: list = [{"role": "assistant", "content": _ASSISTANT_GREETING}]
        dialogue.append({"role": "user", "content": _gen_user(dialogue)})
        done = False

        for step in range(self.max_steps):
            agent_messages = [{"role": "system", "content": self._agent_prompt}] + dialogue
            response = _llm(self.agent_model, agent_messages,
                            temperature=self.agent_temperature, max_tokens=self.agent_max_tokens)
            dialogue.append({"role": "assistant", "content": response})

            if step >= self.max_steps - 1:
                break
            user_turn = _gen_user(dialogue)
            if _CLOSE_MARKER in user_turn:
                done = True
                break
            dialogue.append({"role": "user", "content": user_turn})

        reward = self._judge(dialogue, meta) if self.judge_reward else 0.0
        return EpisodeResult(
            task_id=task.task_id,
            trajectory=dialogue,
            success=reward >= 0.999,
            reward=float(reward),
            persona_policy=persona_policy,
            metadata={"closed": done, "domain": meta.get("domain", "")},
        )

    def _judge(self, dialogue: list, meta: dict) -> float:
        """LLM judge: did the assistant address the request per solution_conditions?
        Returns 1.0 (addressed) / 0.0 (not) — mirrors RealUserSim's judge phase."""
        import litellm
        conds = meta.get("solution_conditions", "")
        if not conds:
            return 0.0
        convo = "\n".join(f"{t['role']}: {t['content']}" for t in dialogue)
        prompt = (
            "You are evaluating whether an AI assistant addressed a user's request.\n\n"
            f"Solution conditions (the request is addressed only if these are met):\n{conds}\n\n"
            f"Conversation:\n{convo}\n\n"
            "Answer with a single word: ADDRESSED or NOT_ADDRESSED.")
        try:
            out = litellm.completion(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=8, timeout=90,
            ).choices[0].message.content or ""
        except Exception:
            return 0.0
        return 1.0 if "ADDRESSED" in out.upper() and "NOT" not in out.upper() else 0.0
