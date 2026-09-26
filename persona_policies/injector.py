"""
Persona Injector
================
Injects a persona policy block into the τ²-bench user simulator system prompt.

Also provides ``TaubenchEpisodeRunner`` for programmatic episodes (train split for
evolution, test split for benchmark — configured via ``PersonaPoliciesConfig``).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from persona_policies.config import PersonaPoliciesConfig
from persona_policies.config import domain_list, is_combined_domain
from persona_policies.llm_bedrock import prepare_litellm_env_for_models
from persona_policies.taubench_setup import (
    apply_tau2_bedrock_judge_overrides,
    ensure_taubench_importable,
)

PERSONA_INJECTION_TEMPLATE = """
## YOUR BEHAVIORAL STYLE FOR THIS CONVERSATION

In addition to your role and task above, you must embody the following behavioral persona throughout this conversation. This persona affects HOW you communicate, NOT WHAT you want or need. Your goal, preferences, and private information remain exactly as described above.

--- PERSONA POLICY BEGIN ---
{persona_policy_text}
--- PERSONA POLICY END ---

CRITICAL RULES FOR PERSONA ADHERENCE:
1. Your GOAL and PRIVATE INFORMATION do not change. You still want to accomplish the same task.
2. Apply the behavioral style consistently across ALL your turns, not just the first one.
3. If the persona says you are terse, be terse even when you have a lot to say.
4. If the persona says you withhold information, do not give it unless the agent specifically asks.
5. If the persona involves emotional state (impatience, confusion), let it show gradually and realistically.
6. Do NOT break character. Do NOT mention that you have a persona or behavioral instructions.
7. The persona should feel like a natural human interaction style, not a caricature.
""".strip()


def inject_persona_into_system_prompt(
    original_system_prompt: str,
    persona_policy_text: str,
    injection_template: str = PERSONA_INJECTION_TEMPLATE,
) -> str:
    """Append the persona block to the full user system prompt."""
    injected_block = injection_template.format(
        persona_policy_text=persona_policy_text.strip()
    )
    return original_system_prompt.strip() + "\n\n" + injected_block.strip()


_PERSONA_USER_SIM_CLASS = None


def _persona_user_class():
    """Cached subclass so ``system_prompt`` includes persona text in ``get_init_state``."""
    global _PERSONA_USER_SIM_CLASS
    if _PERSONA_USER_SIM_CLASS is not None:
        return _PERSONA_USER_SIM_CLASS
    ensure_taubench_importable()
    from tau2.user.user_simulator import UserSimulator

    class PersonaPolicyUserSimulator(UserSimulator):
        """User simulator with optional persona policy appended to ``system_prompt``."""

        def __init__(
            self,
            *args,
            persona_policy_text: Optional[str] = None,
            **kwargs,
        ):
            self._persona_policy_text = persona_policy_text
            super().__init__(*args, **kwargs)

        @property
        def system_prompt(self) -> str:
            base = super().system_prompt
            if self._persona_policy_text:
                return inject_persona_into_system_prompt(
                    base, self._persona_policy_text
                )
            return base

    _PERSONA_USER_SIM_CLASS = PersonaPolicyUserSimulator
    return _PERSONA_USER_SIM_CLASS


def _messages_to_trajectory(messages) -> List[Dict[str, str]]:
    """Convert τ² Message list to ``[{role, content}, ...]`` for fingerprinting."""
    out: List[Dict[str, str]] = []
    for m in messages:
        role = getattr(m, "role", None)
        content = getattr(m, "content", None)
        if role not in ("user", "assistant"):
            continue
        if content is None or (isinstance(content, str) and not content.strip()):
            continue
        out.append({"role": role, "content": content})
    return out


class TaubenchEpisodeRunner:
    """
    Runs one retail (or other) τ²-bench episode and returns trajectory + reward.

    Uses task lists from ``get_tasks(domain, task_split_name)`` so **train** vs **test**
    splits match ``split_tasks.json``.
    """

    def __init__(self, config: PersonaPoliciesConfig):
        ensure_taubench_importable(config.taubench_root)
        apply_tau2_bedrock_judge_overrides(
            nl_model=config.tau2_llm_nl_assertions,
            env_model=config.tau2_llm_env_interface,
        )
        prepare_litellm_env_for_models(
            config.taubench_user_model,
            config.taubench_agent_model,
            config.tau2_llm_nl_assertions,
            config.tau2_llm_env_interface,
        )
        self.config = config
        self._tasks: List = []
        self._task_domains: List[str] = []
        self._split_name: str = config.task_split_evolution
        # task_id -> "train"|"test" when using merged train+test pool; else from single split
        self._task_id_to_split: Dict[str, str] = {}
        self._setup_tasks()

    def use_split(self, split_name: str) -> None:
        """Reload tasks for evolution (train) or benchmark (test)."""
        self._split_name = split_name
        self._setup_tasks()

    def use_train_and_test_splits(self) -> None:
        """Load the disjoint union of official **train** and **test** tasks (for baseline collection)."""
        from tau2.runner.helpers import get_tasks

        domains = domain_list(self.config.taubench_domain)
        self._task_id_to_split = {}
        self._tasks = []
        self._task_domains = []
        for domain in domains:
            train_tasks = get_tasks(domain, "train")
            test_tasks = get_tasks(domain, "test")
            for t in train_tasks:
                self._task_id_to_split[self._task_key_for(domain, t)] = "train"
                self._tasks.append(t)
                self._task_domains.append(domain)
            for t in test_tasks:
                self._task_id_to_split[self._task_key_for(domain, t)] = "test"
                self._tasks.append(t)
                self._task_domains.append(domain)
        self._split_name = "train+test"

    def _setup_tasks(self) -> None:
        from tau2.runner.helpers import get_tasks

        self._tasks = []
        self._task_domains = []
        for domain in domain_list(self.config.taubench_domain):
            tasks = get_tasks(domain, task_split_name=self._split_name)
            self._tasks.extend(tasks)
            self._task_domains.extend([domain] * len(tasks))
        self._task_id_to_split = {
            self._task_key(i): self._split_name for i, _ in enumerate(self._tasks)
        }

    def _task_key_for(self, domain: str, task) -> str:
        tid = str(getattr(task, "id", ""))
        return f"{domain}:{tid}" if is_combined_domain(self.config.taubench_domain) else tid

    def _task_key(self, task_idx: int) -> str:
        return self._task_key_for(self._task_domains[task_idx], self._tasks[task_idx])

    def get_task_key(self, task_idx: int) -> str:
        return self._task_key(task_idx)

    def _build_orchestrator(self, task, persona_policy_text: Optional[str], domain: str):
        import uuid

        from tau2.data_model.simulation import TextRunConfig
        from tau2.orchestrator.orchestrator import Orchestrator
        from tau2.registry import registry
        from tau2.runner.build import _build_env_kwargs, build_agent, build_environment

        cfg = TextRunConfig(
            domain=domain,
            agent="llm_agent",
            user="user_simulator",
            llm_agent=self.config.taubench_agent_model,
            llm_user=self.config.taubench_user_model,
            llm_args_agent={"temperature": 0.0},
            llm_args_user={"temperature": 0.0},
            max_steps=self.config.max_steps,
            seed=self.config.seed,
        )
        solo_mode = registry.get_agent_metadata(
            cfg.effective_agent, "solo_mode", default=False
        )
        env_kwargs = _build_env_kwargs(cfg, task)
        environment = build_environment(
            cfg.domain, solo_mode=solo_mode, env_kwargs=env_kwargs
        )
        agent = build_agent(
            cfg.effective_agent,
            environment,
            llm=cfg.llm_agent,
            llm_args=cfg.llm_args_agent,
            task=task,
            solo_mode=solo_mode,
        )
        try:
            user_tools = environment.get_user_tools(include=task.user_tools) or None
        except Exception:
            user_tools = None

        PersonaPolicyUserSimulator = _persona_user_class()
        user = PersonaPolicyUserSimulator(
            tools=user_tools,
            instructions=str(task.user_scenario),
            llm=cfg.llm_user,
            llm_args=cfg.llm_args_user,
            persona_policy_text=persona_policy_text,
        )

        orch = Orchestrator(
            domain=cfg.domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=cfg.effective_max_steps,
            max_errors=cfg.max_errors,
            seed=cfg.seed,
            solo_mode=solo_mode,
            simulation_id=str(uuid.uuid4()),
            validate_communication=cfg.enforce_communication_protocol,
            timeout=cfg.timeout,
        )
        return orch

    def run_episode(
        self,
        task_idx: int,
        persona_policy_text: Optional[str] = None,
        max_turns: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict:
        """
        Run a single episode.

        ``task_idx`` indexes into the task list for the active split
        (``task_split_evolution`` by default).

        ``max_turns`` overrides ``PersonaPoliciesConfig.max_turns_per_episode`` for
        this call only (e.g. smoke tests). The config default is **30** to align
        with common τ²-bench horizons; set ``max_turns_per_episode`` to
        ``None`` for no cap beyond ``max_steps``.
        """
        if not self._tasks:
            raise RuntimeError("No tasks loaded; check domain and split_tasks.json")
        if task_idx < 0 or task_idx >= len(self._tasks):
            raise IndexError(
                f"task_idx {task_idx} out of range for {len(self._tasks)} tasks"
            )
        task = self._tasks[task_idx]
        domain = self._task_domains[task_idx]
        orch = self._build_orchestrator(task, persona_policy_text, domain)
        cap = max_turns if max_turns is not None else self.config.max_turns_per_episode
        if cap is not None and cap > 0:
            orch.max_steps = min(orch.max_steps, cap)

        from tau2.evaluator.evaluator import EvaluationType
        from tau2.runner.simulation import run_simulation

        # Retail tasks often list NL_ASSERTION in reward_basis; EvaluationType.ALL
        # raises if NL is required but not evaluated. ALL_WITH_NL_ASSERTIONS is safe
        # when nl_assertions is empty (NLEvaluator returns 1.0).
        sim = run_simulation(
            orch, evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS
        )
        messages = sim.get_messages()
        trajectory = _messages_to_trajectory(messages)
        reward = sim.reward_info.reward if sim.reward_info else 0.0
        success = float(reward) >= 1.0
        failure_mode = self.classify_failure_mode(trajectory, success)

        return {
            "task_id": str(task.id),
            "task_idx": task_idx,
            "domain": domain,
            "split": self._task_id_to_split.get(self._task_key(task_idx)),
            "persona_policy": persona_policy_text,
            "trajectory": trajectory,
            "success": success,
            "reward": float(reward),
            "n_turns": len(trajectory),
            "failure_mode": failure_mode,
        }

    def get_task_scenario_text(self, task_idx: int, max_chars: int = 4000) -> str:
        """Task instructions / user scenario string for LLM-judge context (truncated)."""
        if not self._tasks or task_idx < 0 or task_idx >= len(self._tasks):
            return ""
        return str(self._tasks[task_idx].user_scenario)[:max_chars]

    def get_baseline_user_system_prompt(self, task_idx: int) -> str:
        ensure_taubench_importable(self.config.taubench_root)
        from tau2.user.user_simulator import UserSimulator

        task = self._tasks[task_idx]
        domain = self._task_domains[task_idx]
        try:
            from tau2.runner.build import build_environment
            from tau2.data_model.simulation import TextRunConfig
            from tau2.runner.build import _build_env_kwargs
            from tau2.registry import registry

            cfg = TextRunConfig(domain=domain)
            solo_mode = registry.get_agent_metadata(
                "llm_agent", "solo_mode", default=False
            )
            env = build_environment(
                domain,
                solo_mode=solo_mode,
                env_kwargs=_build_env_kwargs(cfg, task),
            )
            user_tools = env.get_user_tools(include=task.user_tools) or None
        except Exception:
            user_tools = None
        u = UserSimulator(
            llm=self.config.taubench_user_model,
            instructions=str(task.user_scenario),
            tools=user_tools,
            llm_args={"temperature": 0.0},
        )
        return u.system_prompt

    def get_available_task_indices(self) -> List[int]:
        return list(range(len(self._tasks)))

    def classify_failure_mode(
        self, trajectory: List[Dict], success: bool
    ) -> Optional[str]:
        if success:
            return None
        full_text = " ".join(t.get("content", "") for t in trajectory).lower()
        agent_texts = " ".join(
            t.get("content", "")
            for t in trajectory
            if t.get("role") == "assistant"
        ).lower()
        if any(
            p in agent_texts
            for p in ["i'm unable to", "i cannot help", "i can't assist"]
        ):
            return "agent_gave_up"
        if len(trajectory) >= 28:
            return "max_turns_exceeded"
        if agent_texts.count("could you please clarify") > 2 or agent_texts.count(
            "can you clarify"
        ) > 2:
            return "clarification_loop"
        if any(
            p in agent_texts
            for p in ["i'll go ahead and", "i'll proceed with", "assuming you want"]
        ):
            return "agent_assumed"
        if re.search(
            r"\b(never mind|forget it|i give up)\b", full_text
        ):
            return "user_abandoned"
        return "other"
