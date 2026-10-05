"""SimpleEpisodeRunner end-to-end with a scripted user-sim stub (no LLM call).

We patch ``_call_user_sim`` so the test is hermetic — no network, no API key,
no model dependency. The point is to verify the turn loop:
  - alternates user/assistant
  - terminates on [DONE]
  - preserves persona_policy on the result
  - respects max_turns
"""

from __future__ import annotations

from typing import List

from ppol import SimpleEpisodeRunner, Task


def _stub_runner(user_messages: List[str], max_turns: int = 10) -> SimpleEpisodeRunner:
    def echo_agent(history):
        last = history[-1]["content"] if history else ""
        return f"agent saw: {last}"

    r = SimpleEpisodeRunner(agent=echo_agent, max_turns=max_turns)
    it = iter(user_messages)
    r._call_user_sim = lambda system_prompt, history: next(it)
    return r


def test_episode_terminates_on_done_signal():
    task = Task(task_id="t1", description="return a laptop", context="retail")
    runner = _stub_runner(["hi i need help", "thanks bye [DONE]"])

    result = runner.run_episode(task, persona_policy="be terse")

    assert result.task_id == "t1"
    assert result.persona_policy == "be terse"
    assert len(result.trajectory) == 3
    assert result.trajectory[0] == {"role": "user", "content": "hi i need help"}
    assert result.trajectory[1]["role"] == "assistant"
    assert result.trajectory[2] == {"role": "user", "content": "thanks bye"}
    assert "[DONE]" not in result.trajectory[2]["content"]


def test_episode_respects_max_turns():
    task = Task(task_id="t2", description="stuck", context="")
    # Never emit [DONE]; expect runner to cap at max_turns user turns.
    runner = _stub_runner([f"u{i}" for i in range(20)], max_turns=4)
    result = runner.run_episode(task)

    # 4 user turns + 4 agent replies (loop runs max_turns iterations,
    # each adds one user then one agent unless done)
    assert len(result.trajectory) == 8
    roles = [t["role"] for t in result.trajectory]
    assert roles == ["user", "assistant"] * 4


def test_empty_persona_policy_preserved():
    task = Task(task_id="t3", description="x", context="")
    runner = _stub_runner(["one [DONE]"])
    result = runner.run_episode(task)
    assert result.persona_policy == ""
    assert result.n_turns == 1
