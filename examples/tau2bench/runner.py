"""τ²-bench runner for ppol.

  * ``Tau2BenchRunner``         — the ``EpisodeRunner`` subclass (user-facing API).
  * ``TaubenchEpisodeRunner``   — low-level wrapper around τ²'s Orchestrator
                                  + PersonaPolicy-aware UserSimulator.
  * ``ensure_taubench_importable`` / ``apply_tau2_bedrock_judge_overrides``
                                — sys.path + judge model overrides.

The human-reference builder lives in ``tau_reference.py`` (analog of
``swe_chat_reference.py`` / ``wildchat_reference.py``).

Requirements:
  - tau2-bench installed (``pip install git+https://github.com/sierra-research/tau2-bench``).
  - ``ppol[tau2bench]`` extras if using Bedrock models.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from ppol.config import PPolConfig, domain_list, is_combined_domain
from ppol.core.runner import EpisodeRunner
from ppol.core.types import EpisodeResult, Task
from ppol.injection import inject_persona_into_system_prompt


# ---------------------------------------------------------------------------
# τ²-bench import-path setup (was taubench_setup.py)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TAU2_BENCH = _REPO_ROOT / "tau2-bench"


def ensure_taubench_importable(taubench_root: str | Path | None = None) -> Path:
    """Set ``TAU2_DATA_DIR`` and prepend tau2 ``src`` to ``sys.path``."""
    root = Path(taubench_root).resolve() if taubench_root else _DEFAULT_TAU2_BENCH.resolve()
    src = root / "src"
    data = root / "data"
    if data.is_dir():
        os.environ.setdefault("TAU2_DATA_DIR", str(data))
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def apply_tau2_bedrock_judge_overrides(
    nl_model: str = "bedrock/deepseek.v3-v1:0",
    env_model: str = "bedrock/deepseek.v3-v1:0",
) -> None:
    """Override τ²'s default OpenAI-gpt-4.1 judges to Bedrock models for Bedrock-only runs."""
    ensure_taubench_importable()
    import tau2.config as tc

    tc.DEFAULT_LLM_NL_ASSERTIONS = nl_model
    tc.DEFAULT_LLM_ENV_INTERFACE = env_model
    try:
        import tau2.evaluator.evaluator_nl_assertions as nle
        nle.DEFAULT_LLM_NL_ASSERTIONS = nl_model
    except ImportError:
        pass
    try:
        import tau2.environment.utils.interface_agent as ia
        ia.DEFAULT_LLM_ENV_INTERFACE = env_model
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Persona-aware τ² user simulator (was injector.py)
# ---------------------------------------------------------------------------

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

        def __init__(self, *args, persona_policy_text: Optional[str] = None, **kwargs):
            self._persona_policy_text = persona_policy_text
            super().__init__(*args, **kwargs)

        @property
        def system_prompt(self) -> str:
            base = super().system_prompt
            if self._persona_policy_text:
                return inject_persona_into_system_prompt(base, self._persona_policy_text)
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
    """Runs one τ²-bench episode and returns trajectory + reward.

    Uses task lists from ``get_tasks(domain, task_split_name)`` so **train** vs **test**
    splits match ``split_tasks.json``.
    """

    def __init__(self, config: PPolConfig):
        from ppol.llm import prepare_litellm_env_for_models

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

        self._task_id_to_split = {}
        self._tasks = []
        self._task_domains = []
        for domain in domain_list(self.config.taubench_domain):
            for split in ("train", "test"):
                for t in get_tasks(domain, split):
                    self._task_id_to_split[self._task_key_for(domain, t)] = split
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

    def _build_orchestrator(
        self,
        task,
        persona_policy_text: Optional[str],
        domain: str,
        agent_temperature: float = 0.0,
    ):
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
            llm_args_agent={"temperature": agent_temperature},
            llm_args_user={"temperature": 0.0},
            max_steps=self.config.max_steps,
            seed=self.config.seed,
        )
        solo_mode = registry.get_agent_metadata(cfg.effective_agent, "solo_mode", default=False)
        env_kwargs = _build_env_kwargs(cfg, task)
        environment = build_environment(cfg.domain, solo_mode=solo_mode, env_kwargs=env_kwargs)
        agent = build_agent(
            cfg.effective_agent, environment,
            llm=cfg.llm_agent, llm_args=cfg.llm_args_agent,
            task=task, solo_mode=solo_mode,
        )
        try:
            user_tools = environment.get_user_tools(include=task.user_tools) or None
        except Exception:
            user_tools = None

        user = _persona_user_class()(
            tools=user_tools,
            instructions=str(task.user_scenario),
            llm=cfg.llm_user,
            llm_args=cfg.llm_args_user,
            persona_policy_text=persona_policy_text,
        )

        return Orchestrator(
            domain=cfg.domain, agent=agent, user=user, environment=environment, task=task,
            max_steps=cfg.effective_max_steps, max_errors=cfg.max_errors, seed=cfg.seed,
            solo_mode=solo_mode, simulation_id=str(uuid.uuid4()),
            validate_communication=cfg.enforce_communication_protocol, timeout=cfg.timeout,
        )

    def run_episode(
        self,
        task_idx: int,
        persona_policy_text: Optional[str] = None,
        max_turns: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict:
        """Run a single episode. ``task_idx`` indexes into the active split's task list."""
        if not self._tasks:
            raise RuntimeError("No tasks loaded; check domain and split_tasks.json")
        if task_idx < 0 or task_idx >= len(self._tasks):
            raise IndexError(f"task_idx {task_idx} out of range for {len(self._tasks)} tasks")
        task = self._tasks[task_idx]
        domain = self._task_domains[task_idx]
        cap = max_turns if max_turns is not None else self.config.max_turns_per_episode

        from tau2.evaluator.evaluator import EvaluationType
        from tau2.runner.simulation import run_simulation

        # Retry episodes on transient Gemma-4/OpenRouter errors (empty/malformed assistant messages).
        _temp_schedule = [0.0, 0.0, 0.4, 0.7, 1.0]
        attempts = int(getattr(self.config, "episode_max_attempts", 5) or 1)
        last_exc: Optional[Exception] = None
        sim = None
        for attempt in range(attempts):
            agent_temp = _temp_schedule[min(attempt, len(_temp_schedule) - 1)]
            orch = self._build_orchestrator(
                task, persona_policy_text, domain, agent_temperature=agent_temp
            )
            if cap is not None and cap > 0:
                orch.max_steps = min(orch.max_steps, cap)
            try:
                # Retail tasks often list NL_ASSERTION in reward_basis; ALL_WITH_NL_ASSERTIONS
                # is safe even when nl_assertions is empty (NLEvaluator returns 1.0).
                sim = run_simulation(orch, evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS)
                break
            except (ValueError, json.JSONDecodeError) as e:
                last_exc = e
                continue
        if sim is None:
            raise last_exc if last_exc is not None else RuntimeError("run_simulation failed")
        trajectory = _messages_to_trajectory(sim.get_messages())
        reward = float(sim.reward_info.reward if sim.reward_info else 0.0)
        success = reward >= 1.0

        return {
            "task_id": str(task.id),
            "task_idx": task_idx,
            "domain": domain,
            "split": self._task_id_to_split.get(self._task_key(task_idx)),
            "persona_policy": persona_policy_text,
            "trajectory": trajectory,
            "success": success,
            "reward": reward,
            "n_turns": len(trajectory),
            "failure_mode": self.classify_failure_mode(trajectory, success),
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
            from tau2.runner.build import _build_env_kwargs, build_environment
            from tau2.data_model.simulation import TextRunConfig
            from tau2.registry import registry

            cfg = TextRunConfig(domain=domain)
            solo_mode = registry.get_agent_metadata("llm_agent", "solo_mode", default=False)
            env = build_environment(domain, solo_mode=solo_mode, env_kwargs=_build_env_kwargs(cfg, task))
            user_tools = env.get_user_tools(include=task.user_tools) or None
        except Exception:
            user_tools = None
        return UserSimulator(
            llm=self.config.taubench_user_model,
            instructions=str(task.user_scenario),
            tools=user_tools,
            llm_args={"temperature": 0.0},
        ).system_prompt

    def get_available_task_indices(self) -> List[int]:
        return list(range(len(self._tasks)))

    def classify_failure_mode(self, trajectory: List[Dict], success: bool) -> Optional[str]:
        if success:
            return None
        full_text = " ".join(t.get("content", "") for t in trajectory).lower()
        agent_texts = " ".join(
            t.get("content", "") for t in trajectory if t.get("role") == "assistant"
        ).lower()
        if any(p in agent_texts for p in ("i'm unable to", "i cannot help", "i can't assist")):
            return "agent_gave_up"
        if len(trajectory) >= 28:
            return "max_turns_exceeded"
        if agent_texts.count("could you please clarify") > 2 or agent_texts.count("can you clarify") > 2:
            return "clarification_loop"
        if any(p in agent_texts for p in ("i'll go ahead and", "i'll proceed with", "assuming you want")):
            return "agent_assumed"
        if re.search(r"\b(never mind|forget it|i give up)\b", full_text):
            return "user_abandoned"
        return "other"


# ---------------------------------------------------------------------------
# EpisodeRunner adapter (user-facing API)
# ---------------------------------------------------------------------------

class Tau2BenchRunner(EpisodeRunner):
    """ppol ``EpisodeRunner`` wrapper around ``TaubenchEpisodeRunner``.

    Example::

        from runner import Tau2BenchRunner
        runner = Tau2BenchRunner()
        result = runner.run_episode(runner.get_tasks()[0], persona_policy="Be terse and skeptical.")
    """

    def __init__(self, config: Optional[PPolConfig] = None) -> None:
        self._cfg = config or PPolConfig()
        self._runner = TaubenchEpisodeRunner(self._cfg)

    def use_split(self, split: str) -> None:
        """Switch between ``'train'`` and ``'test'`` task splits."""
        self._runner.use_split(split)

    def use_train_and_test_splits(self) -> None:
        """Load the disjoint union of official train and test tasks (baseline collection)."""
        self._runner.use_train_and_test_splits()

    def get_tasks(self) -> List[Task]:
        """Return a ``Task`` for every task in the active split."""
        return [
            Task(
                task_id=str(idx),
                description=self._runner.get_task_scenario_text(idx),
                context=f"domain={self._cfg.taubench_domain}",
                metadata={"task_idx": idx, "domain": self._cfg.taubench_domain},
            )
            for idx in self._runner.get_available_task_indices()
        ]

    def run_episode(
        self, task: Task, persona_policy: str = "", *, max_turns: Optional[int] = None,
    ) -> EpisodeResult:
        task_idx = int(task.metadata.get("task_idx", task.task_id))
        raw = self._runner.run_episode(
            task_idx=task_idx,
            persona_policy_text=persona_policy or None,
            max_turns=max_turns,
        )
        metadata = {k: v for k, v in raw.items() if k != "trajectory"}
        return EpisodeResult(
            task_id=task.task_id,
            trajectory=raw["trajectory"],
            success=bool(raw.get("success", False)),
            reward=float(raw.get("reward", 0.0)),
            n_turns=int(raw.get("n_turns", len(raw["trajectory"]))),
            persona_policy=persona_policy,
            metadata=metadata,
        )

