"""
EpisodeRunner ABC and SimpleEpisodeRunner.

SimpleEpisodeRunner supports any LiteLLM-compatible model string for the user sim:

  Provider        Model string example
  --------        --------------------
  OpenAI          "gpt-4o-mini"  /  "gpt-4o"
  Anthropic       "claude-haiku-4-5-20251001"  /  "claude-sonnet-4-6"
  OpenRouter      "openrouter/google/gemini-2.0-flash-lite"
  Google          "gemini/gemini-2.0-flash"
  AWS Bedrock     "bedrock/anthropic.claude-3-haiku-20240307-v1:0"

Set the matching API key env var before running:
  OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List

from ppol.core.types import Conversation, EpisodeResult, Task
from ppol.injection import PERSONA_INJECTION_TEMPLATE as _PERSONA_INJECTION_TEMPLATE

_DONE_SIGNAL = "[DONE]"

_USER_SIM_SYSTEM_TEMPLATE = """You are playing the role of a user in a conversation with an assistant or agent.

Your task: {description}{context_block}

When you have accomplished your goal or decided to give up, add {done_signal} at the very end of your message. Otherwise, keep the conversation going naturally.""".strip()

AgentFn = Callable[[List[dict]], str]


class EpisodeRunner(ABC):
    """Base class for all episode runners.

    Subclass this to integrate a new benchmark or dataset.
    Implement ``run_episode``; optionally override ``get_tasks``.
    """

    @abstractmethod
    def run_episode(self, task: Task, persona_policy: str = "") -> EpisodeResult:
        """Run one episode and return the result."""
        ...

    def get_tasks(self) -> List[Task]:
        """Return available tasks. Override in dataset-specific runners."""
        return []


class SimpleEpisodeRunner(EpisodeRunner):
    """
    Turn-based episode runner for custom datasets.

    The user simulator is an LLM (any LiteLLM-compatible model string).
    The agent is any callable ``(messages: list[dict]) -> str``.

    Example::

        def my_agent(messages):
            # messages is the conversation so far (user/assistant turns)
            return "Hello, how can I help?"

        runner = SimpleEpisodeRunner(
            agent=my_agent,
            user_sim_model="gpt-4o-mini",
        )
        task = Task(task_id="t1", description="Return my order #12345")
        result = runner.run_episode(task, persona_policy="Be terse and impatient.")
    """

    def __init__(
        self,
        agent: AgentFn,
        *,
        user_sim_model: str = "gpt-4o-mini",
        max_turns: int = 20,
        user_sim_temperature: float = 0.7,
        user_sim_max_tokens: int = 512,
    ) -> None:
        self.agent = agent
        self.user_sim_model = user_sim_model
        self.max_turns = max_turns
        self.user_sim_temperature = user_sim_temperature
        self.user_sim_max_tokens = user_sim_max_tokens

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _user_sim_system_prompt(self, task: Task, persona_policy: str) -> str:
        context_block = f"\nAdditional context: {task.context}" if task.context.strip() else ""
        base = _USER_SIM_SYSTEM_TEMPLATE.format(
            description=task.description,
            context_block=context_block,
            done_signal=_DONE_SIGNAL,
        )
        if persona_policy.strip():
            injected = _PERSONA_INJECTION_TEMPLATE.format(
                persona_policy_text=persona_policy.strip()
            )
            return base + "\n\n" + injected
        return base

    # ------------------------------------------------------------------
    # LLM call — delegates to the ppol LiteLLM wrapper so all
    # provider routing, retries, and OpenRouter hardening are inherited.
    # ------------------------------------------------------------------

    def _call_user_sim(self, system_prompt: str, history: Conversation) -> str:
        from ppol.llm import completion_text

        messages = [{"role": "system", "content": system_prompt}] + history
        return completion_text(
            self.user_sim_model,
            messages,
            temperature=self.user_sim_temperature,
            max_tokens=self.user_sim_max_tokens,
        )

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------

    def run_episode(self, task: Task, persona_policy: str = "") -> EpisodeResult:
        system_prompt = self._user_sim_system_prompt(task, persona_policy)
        trajectory: Conversation = []

        for _ in range(self.max_turns):
            user_msg = self._call_user_sim(system_prompt, trajectory)
            done = _DONE_SIGNAL in user_msg
            clean = user_msg.replace(_DONE_SIGNAL, "").strip()
            trajectory.append({"role": "user", "content": clean})
            if done:
                break

            agent_msg = self.agent(trajectory)
            trajectory.append({"role": "assistant", "content": str(agent_msg)})

        return EpisodeResult(
            task_id=task.task_id,
            trajectory=trajectory,
            persona_policy=persona_policy,
            n_turns=len(trajectory),
        )
