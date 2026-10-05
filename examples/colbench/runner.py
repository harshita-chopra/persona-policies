"""ColBench (SWEET-RL) EpisodeRunner for ppol — backend/code track.

ColBench (Zhou et al., 2025, arXiv:2503.15478) pairs an LLM agent with a
simulated human collaborator: the agent writes a Python function, the
simulator answers clarification questions from a private reference solution,
and reward = fraction of hidden unit tests passed. The simulator produces the
role=="user" turns — the party ppol overlays; a persona is injected into
its prompt while all task facts (problem, solution, tests, reward) stay fixed.

Data: facebook/collaborative_agent_bench on the HF Hub — auto-downloaded on
first use. Vendored logic (correctness check + prompt files) is from SWEET-RL: 
https://github.com/facebookresearch/sweet_rl.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import List, Optional

from ppol import EpisodeResult, EpisodeRunner, Task, inject_persona_into_system_prompt
from ppol.config import default_config

_cfg = default_config()
_DEFAULT_AGENT_MODEL     = _cfg.taubench_agent_model      # model under evaluation
_DEFAULT_HUMAN_SIM_MODEL = _cfg.taubench_user_model       # the persona-injected user

_HF_REPO = "facebook/collaborative_agent_bench"
_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

# Constants matching HumanInteractionEnv.
_HUMAN_RESPONSE_CHARACTER_LIMIT = 500
_ANSWER_MARKER = "I WANT TO ANSWER:"
_OUTPUT_MARKER = "OUTPUT:"

# Fixed agent greeting that opens the dialogue. The user sim then states the problem
# in its OWN words (a generated, persona-controllable turn) instead of the injected
# problem_description — so opening_length reflects sim behavior, not the fixed task text.
_AGENT_GREETING = "Hi! How can I help you?"


# ---------------------------------------------------------------------------
# Sandboxed correctness check — vendored from sweet_rl/utils/code_utils.py.
# Runs the agent's function and the reference on each hidden test expression
# and compares outputs. Reward = fraction of tests whose outputs match.
# ---------------------------------------------------------------------------

def _get_function_output(function_definition: str, test_case: str):
    # Shared namespace so functions defined by exec() are visible to eval();
    # a bare exec()/eval() inside a function does not share locals.
    ns: dict = {}
    try:
        exec(function_definition, ns)  # noqa: S102 - sandboxed below via subprocess
        return eval(test_case, ns)     # noqa: S307
    except Exception:
        return None


def _run_with_timeout(function_definition: str, test_case: str, timeout: float):
    """Evaluate a test expression with a wall-clock timeout.

    We use a daemon thread rather than a subprocess on purpose: under evolution
    this module is imported from a file path under a synthetic name
    (ppol_user_runner) that `multiprocessing` spawn-workers cannot re-import, so
    a Process-based sandbox floods the run with ModuleNotFoundError and returns
    None every time. A thread has no import/pickling dependency and works in any
    start-method context. A runaway function leaks one daemon thread (bounded by
    the import/IO blocklist below) instead of wedging the episode.
    """
    result: list = [None]

    def _target():
        result[0] = _get_function_output(function_definition, test_case)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    return result[0] if not t.is_alive() else None


def _strip_code_fences(code: str) -> str:
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        parts = code.split("```")
        if len(parts) >= 3:
            code = parts[1]
    return code


def check_correctness(ground_truth_function: str, test_function: str, test_cases: dict) -> float:
    """Fraction of test cases where the candidate matches the reference output.

    Mirrors sweet_rl.utils.code_utils.check_correctness: the same import/IO
    blocklist and a killable subprocess with a hard per-test timeout.
    """
    if not test_cases:
        return 0.0
    test_function = _strip_code_fences(test_function)
    if any(tok in test_function for tok in
           ("import os", "from os", "import sys", "from sys", "sudo",
            "transformers", "exit(", "quit(", "argparse")):
        return 0.0

    num_correct = 0
    for test_case in test_cases.values():
        ground_truth_output = _get_function_output(ground_truth_function, test_case)

        if any(tok in test_function for tok in
               ("import os", "from os", "import sys", "from sys",
                "open(", "print(", "write")):
            test_output = None
        else:
            test_output = _run_with_timeout(test_function, test_case, timeout=2.0)

        try:
            if ground_truth_output is not None and ground_truth_output == test_output:
                num_correct += 1
        except ValueError:
            pass
    return num_correct / len(test_cases)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class ColBenchRunner(EpisodeRunner):
    """ppol EpisodeRunner over ColBench backend-programming tasks.

    The persona-injected human simulator (user sim) collaborates with a fixed
    coding agent. Reward = fraction of hidden unit tests the agent's final code
    passes; success = all tests pass.
    """

    def __init__(
        self,
        *,
        data_path: Optional[str] = None,
        split: str = "test",
        num_tasks: int = 200,
        seed: int = 0,
        agent_model: str = _DEFAULT_AGENT_MODEL,
        human_sim_model: str = _DEFAULT_HUMAN_SIM_MODEL,
        max_steps: int = 10,
        agent_temperature: float = 1.0,      # VLLMAgent default in simulate_interactions
        human_sim_temperature: float = 0.0,  # HumanInteractionEnv.invoke_model uses 0
        agent_max_tokens: int = 1024,        # VLLMAgent SamplingParams
        human_sim_max_tokens: int = 4096,    # HumanInteractionEnv.invoke_model
        human_sees_own_turns: bool = True,   # (blob mode only) True = faithful ColBench
                                             # full history; False = drop the sim's own
                                             # prior turns to break the verbatim-echo loop
        chat_messages: bool = True,          # True = API-style: human sim gets its own
                                             # system prompt + the dialogue as alternating
                                             # chat messages (roles flipped). False = the
                                             # legacy ColBench single text-blob prompt.
    ) -> None:
        self.data_path             = data_path
        self.split                 = split
        self.num_tasks             = num_tasks
        self.seed                  = seed
        self.agent_model           = agent_model
        self.human_sim_model       = human_sim_model
        self.max_steps             = max_steps
        self.agent_temperature     = agent_temperature
        self.human_sim_temperature = human_sim_temperature
        self.agent_max_tokens      = agent_max_tokens
        self.human_sim_max_tokens  = human_sim_max_tokens
        self.human_sees_own_turns  = human_sees_own_turns
        self.chat_messages         = chat_messages
        self._records: Optional[List[dict]] = None
        self._human_prompt = (_PROMPT_DIR / "human_simulator_code_prompt.txt").read_text()
        self._agent_prompt = (_PROMPT_DIR / "llm_agent_code_prompt.txt").read_text()

    # -- data ---------------------------------------------------------------

    def _resolve_data_path(self) -> str:
        if self.data_path:
            return self.data_path
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=_HF_REPO,
            filename=f"backend_tasks/{self.split}.jsonl",
            repo_type="dataset",
        )

    def _all_records(self) -> List[dict]:
        if self._records is None:
            with open(self._resolve_data_path()) as f:
                self._records = [json.loads(line) for line in f if line.strip()]
        return self._records

    def _indices(self) -> List[int]:
        """Sample of `num_tasks` original file indices (by seed).
        Indices — not positions — key the tasks, so any process reconstructing
        the runner with the same (split, num_tasks, seed) gets identical ids."""
        import random
        recs = self._all_records()
        n = len(recs)
        if not self.num_tasks or self.num_tasks >= n:
            return list(range(n))
        return sorted(random.Random(self.seed).sample(range(n), self.num_tasks))

    def get_tasks(self) -> List[Task]:
        """Tasks framed from the *human user's* perspective, so the persona
        generator produces user personas (not agent personas)."""
        ctx = ("Collaborative coding: a human user needs a personalized Python "
               "function and works with an AI coding agent, answering the agent's "
               "clarification questions to convey the hidden requirements.")
        recs = self._all_records()
        tasks = []
        for i in self._indices():
            r = recs[i]
            # `hidden_information` == `ground_truth` (the reference solution) in
            # ColBench; `ground_truth` + `test_cases` are the eval oracle.
            tasks.append(Task(
                task_id=f"colbench_{self.split}_{i:05d}",
                description=(
                    "You are a human user who needs help writing a Python function. "
                    f"Your request:\n{r['problem_description']}"
                ),
                context=ctx,
                metadata={
                    "record_index": i,
                    "problem_description": r["problem_description"],
                    "hidden_information": r["hidden_information"],
                    "ground_truth": r["ground_truth"],
                    "test_cases": r["test_cases"],
                },
            ))
        return tasks

    # -- faithful HumanInteractionEnv helpers -------------------------------

    @staticmethod
    def _str_dialogue_history(dialogue_history: list) -> str:
        """Verbatim port of HumanInteractionEnv.str_dialogue_history()."""
        result = ""
        for d in dialogue_history:
            result += str(d["role"]) + ":"
            result += str(d["content"]) + "\n\n\n\n"
        return result + "agent:"

    def _human_prompt_text(self, rec: dict, dialogue_history: list, persona_policy: str) -> str:
        """The single user message sent to the human simulator, exactly as
        HumanInteractionEnv.invoke_model formats it — then the ppol persona
        block appended (ppol's injection point).

        When ``human_sees_own_turns`` is False we drop the simulator's own prior
        replies (role=="user") from the re-prompted history, keeping only the
        agent's turns. The sim's own turns are the copy bait behind the verbatim
        echo loop (it re-copies a short distinctive prior reply); removing them
        breaks the loop. The problem statement is unaffected — it is passed
        separately via {problem_description}, not relied on from the history."""
        history = dialogue_history
        if not self.human_sees_own_turns:
            history = [d for d in dialogue_history if d.get("role") != "user"]
        base = self._human_prompt.format(
            problem_description=rec["problem_description"],
            hidden_information=rec["hidden_information"],
            dialogue_history=self._str_dialogue_history(history),
        )
        if persona_policy.strip():
            return inject_persona_into_system_prompt(base, persona_policy)
        return base

    def _human_messages(self, rec: dict, dialogue: list, persona_policy: str) -> list:
        """API-style human-sim call: the human/persona system prompt + the dialogue
        as alternating chat messages, with roles flipped so the simulator's own
        turns are ``assistant`` and the agent's are ``user`` (each model sees itself
        as the assistant — the tau2-style convention chat-tuned models expect).

        Turn 0 (the problem statement) is kept as the opening ``assistant`` message;
        it also appears in the system prompt for context. This replaces the legacy
        single-text-blob prompt and removes the verbatim-echo bait: the sim's own
        turns are now structured assistant messages, not copyable history text."""
        system = self._human_prompt.format(
            problem_description=rec["problem_description"],
            hidden_information=rec["hidden_information"],
        )
        if persona_policy.strip():
            system = inject_persona_into_system_prompt(system, persona_policy)
        messages = [{"role": "system", "content": system}]
        for turn in dialogue:
            role = "assistant" if turn["role"] == "user" else "user"  # flip to sim's POV
            messages.append({"role": role, "content": turn["content"]})
        # Portability guard: turn 0 is the human's own turn -> 'assistant', so the
        # first non-system message is 'assistant'. Some providers (e.g. Anthropic)
        # require it to be 'user'. Prepend a minimal agent primer to keep valid
        # user/assistant alternation on any backend. No-op for lenient providers.
        if len(messages) > 1 and messages[1]["role"] == "assistant":
            messages.insert(1, {"role": "user", "content": "How can I help you?"})
        return messages

    # -- episode ------------------------------------------------------------

    def run_episode(self, task: Task, persona_policy: str = "") -> EpisodeResult:
        import litellm

        rec = self._all_records()[task.metadata["record_index"]]

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

        def _gen_human(dialogue: list) -> str:
            """One user-sim reply given the dialogue so far (chat or blob mode)."""
            if self.chat_messages:
                msgs = self._human_messages(rec, dialogue, persona_policy)
            else:
                msgs = [{"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": self._human_prompt_text(rec, dialogue, persona_policy)}]
            reply = _llm(self.human_sim_model, msgs,
                         temperature=self.human_sim_temperature, max_tokens=self.human_sim_max_tokens)
            return reply[:_HUMAN_RESPONSE_CHARACTER_LIMIT]

        # Dialogue opens with a fixed agent greeting; the user sim then states the
        # problem in ITS OWN words (a generated, persona-controllable turn) rather than
        # the injected problem_description. So the opening is real sim behavior.
        dialogue: list = [{"role": "assistant", "content": _AGENT_GREETING}]
        dialogue.append({"role": "user", "content": _gen_human(dialogue)})
        answer = "No answer"
        done = False

        # batch_interact_environment loops for max_steps+1; each pass = one agent
        # action then one env.step (which appends the agent turn + a human reply).
        for step in range(self.max_steps):
            # --- agent.get_action: chat, system=agent_prompt, then dialogue ---
            agent_messages = [{"role": "system", "content": self._agent_prompt}] + dialogue
            response = _llm(
                self.agent_model, agent_messages,
                temperature=self.agent_temperature, max_tokens=self.agent_max_tokens,
            )

            # --- env.step parsing (verbatim logic) ---
            if _OUTPUT_MARKER in response:
                response = response.split(_OUTPUT_MARKER)[1]
            reached_cap = step >= self.max_steps - 1
            if _ANSWER_MARKER in response or reached_cap:
                done = True
                answer = (response.split(_ANSWER_MARKER)[1]
                          if _ANSWER_MARKER in response else response)

            dialogue.append({"role": "assistant", "content": response})

            if done:
                break

            # --- human simulator reply ---
            dialogue.append({"role": "user", "content": _gen_human(dialogue)})

        reward = check_correctness(rec["ground_truth"], answer, rec["test_cases"])
        success = reward >= 0.999

        return EpisodeResult(
            task_id=task.task_id,
            trajectory=dialogue,
            success=success,
            reward=float(reward),
            persona_policy=persona_policy,
            metadata={"answer": answer, "tests_passed_frac": float(reward)},
        )
