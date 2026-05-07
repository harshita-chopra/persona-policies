from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from persona_policies.config import PersonaPoliciesConfig
from persona_policies.injector import TaubenchEpisodeRunner
from tau2.evaluator.evaluator import EvaluationType
from tau2.runner.simulation import run_simulation


def serialize(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json", exclude_none=False)
    return {"content": str(message), "_unparsed": True}


def run_episode(runner: TaubenchEpisodeRunner, task_idx: int) -> dict[str, Any]:
    task = runner._tasks[task_idx]
    domain = runner._task_domains[task_idx]
    orch = runner._build_orchestrator(task, None, domain)
    sim = run_simulation(orch, evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS)
    reward = float(sim.reward_info.reward if sim.reward_info else 0.0)
    return {
        "domain": domain,
        "task_id": str(task.id),
        "split": runner._task_id_to_split.get(runner.get_task_key(task_idx), ""),
        "reward": reward,
        "messages": [serialize(m) for m in sim.get_messages()],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="retail", choices=("retail", "airline", "retail_airline"))
    ap.add_argument("--split", default="train", choices=("train", "test"))
    ap.add_argument("--agent-model", required=True)
    ap.add_argument("--user-model", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    config = PersonaPoliciesConfig()
    config.taubench_domain = args.domain
    config.taubench_agent_model = args.agent_model
    config.taubench_user_model = args.user_model
    runner = TaubenchEpisodeRunner(config)
    runner.use_split(args.split)

    task_indices = runner.get_available_task_indices()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_episode, runner, i) for i in task_indices]
        for fut in as_completed(futures):
            f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
