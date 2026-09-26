"""
Collect baseline τ²-bench trajectories (no persona injected).

Run once before evolution / discriminator training:

  python persona_policies/scripts/collect_baseline.py

By default rolls **one episode per task** in the chosen split(s). Use ``--n`` to
override (e.g. oversample).

Artifacts are named
``baseline_<taubench_domain>_<baseline_collect_split>_<taubench_agent_model>_user_<taubench_user_model>_{results,fingerprints,trajectories}.json``

Each trajectory row is ``{trajectory, domain, task_id, split}`` so you can filter
train vs test later. Fingerprints mirror the same metadata.

If all three outputs already exist, the script exits without rerolling unless ``--force``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
from pathlib import Path

# Repo root (so ``python persona_policies/scripts/...`` finds the ``persona_policies`` package)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from tqdm import tqdm

from persona_policies.config import PersonaPoliciesConfig
from persona_policies.config import domain_list
from persona_policies.fingerprinting import BehavioralFingerprintExtractor


def _baseline_outputs_present(config: PersonaPoliciesConfig) -> bool:
    return all(
        os.path.isfile(p)
        for p in (
            config.baseline_results_path,
            config.baseline_fingerprints_path,
            config.baseline_trajectories_path,
        )
    )


def _read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data, *, indent: int | None = None) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def _config_for_domain(base: PersonaPoliciesConfig, domain: str, split: str) -> PersonaPoliciesConfig:
    cfg = PersonaPoliciesConfig()
    cfg.taubench_agent_model = base.taubench_agent_model
    cfg.taubench_user_model = base.taubench_user_model
    cfg.baseline_collect_split = split
    cfg.taubench_domain = domain
    cfg.refresh_domain_artifact_paths()
    return cfg


def merge_existing_baselines(
    config: PersonaPoliciesConfig,
    *,
    collect_split: str | None = None,
    force: bool = False,
) -> tuple[list, list]:
    """Concatenate per-domain baseline JSON artifacts into one combined artifact set."""
    split = collect_split if collect_split is not None else config.baseline_collect_split
    if split != config.baseline_collect_split:
        config.baseline_collect_split = split
        config.refresh_baseline_artifact_paths()
    config.ensure_output_dirs()

    domains = domain_list(config.taubench_domain)
    if len(domains) < 2:
        raise ValueError("--merge-existing requires a combined domain such as retail_airline")
    if not force and _baseline_outputs_present(config):
        print(
            f"Combined baseline artifacts already exist for {config.baseline_stem} — skipping. "
            f"(Use --force to rewrite from per-domain artifacts.)"
        )
        return [], []

    missing: list[str] = []
    source_configs = [_config_for_domain(config, dom, split) for dom in domains]
    for cfg in source_configs:
        for path in (
            cfg.baseline_results_path,
            cfg.baseline_fingerprints_path,
            cfg.baseline_trajectories_path,
        ):
            if not os.path.isfile(path):
                missing.append(path)
    if missing:
        raise FileNotFoundError(
            "Missing per-domain baseline artifact(s); collect those first:\n  "
            + "\n  ".join(missing)
        )

    trajectory_records: list = []
    fingerprints_data: list = []
    per_episode: list[dict] = []
    successes: list[bool] = []
    n_errors = 0

    for cfg in source_configs:
        results = _read_json(cfg.baseline_results_path)
        trajectories = _read_json(cfg.baseline_trajectories_path)
        fingerprints = _read_json(cfg.baseline_fingerprints_path)
        trajectory_records.extend(trajectories)
        fingerprints_data.extend(fingerprints)
        for ep in results.get("per_episode", []):
            row = dict(ep)
            row["source_baseline_stem"] = cfg.baseline_stem
            per_episode.append(row)
            if "error" in row:
                n_errors += 1
            elif "success" in row:
                successes.append(bool(row["success"]))

    success_rate = float(np.mean(successes)) if successes else 0.0
    _write_json(
        config.baseline_results_path,
        {
            "success_rate": success_rate,
            "n_episodes": len(successes),
            "n_errors": n_errors,
            "collect_split": split,
            "merged_from_domains": domains,
            "per_episode": per_episode,
        },
        indent=2,
    )
    _write_json(config.baseline_fingerprints_path, fingerprints_data)
    _write_json(config.baseline_trajectories_path, trajectory_records)
    print(
        f"Merged existing baselines into {config.baseline_stem}: "
        f"{len(trajectory_records)} trajectories, {len(fingerprints_data)} fingerprints, "
        f"success_rate={success_rate:.3f}"
    )
    return trajectory_records, fingerprints_data


def collect_baseline(
    config: PersonaPoliciesConfig,
    n_episodes: int | None = None,
    *,
    collect_split: str | None = None,
    force: bool = False,
    workers: int | None = None,
    save_every: int = 10,
    merge_existing: bool = False,
):
    """Run τ² with NO persona and cache fingerprints + success rate + per-episode metadata."""
    split = collect_split if collect_split is not None else config.baseline_collect_split
    if split != config.baseline_collect_split:
        config.baseline_collect_split = split
        config.refresh_baseline_artifact_paths()

    config.ensure_output_dirs()
    if merge_existing:
        return merge_existing_baselines(config, collect_split=split, force=force)

    from persona_policies.injector import TaubenchEpisodeRunner

    if not force and _baseline_outputs_present(config):
        print(
            f"Baseline artifacts already exist for {config.baseline_stem} — skipping. "
            f"(Use --force to regenerate.)"
        )
        return None, None
    runner = TaubenchEpisodeRunner(config)
    if split == "all":
        runner.use_train_and_test_splits()
    else:
        runner.use_split(split)

    task_indices = runner.get_available_task_indices()
    n_roll = n_episodes if n_episodes is not None else len(task_indices)
    n_workers = max(1, int(workers if workers is not None else config.parallel_episode_workers))

    def _run_one(i: int) -> dict:
        task_idx = task_indices[i % len(task_indices)]
        try:
            result = runner.run_episode(task_idx=task_idx, persona_policy_text=None)
            fp = BehavioralFingerprintExtractor().compute_fingerprint(result["trajectory"])
            return {
                "episode_idx": i,
                "task_idx": task_idx,
                "task_id": result.get("task_id"),
                "domain": result.get("domain"),
                "split": result.get("split"),
                "trajectory": result["trajectory"],
                "success": bool(result["success"]),
                "fingerprint": fp.features,
                "reward": result.get("reward"),
                "n_turns": result.get("n_turns"),
                "failure_mode": result.get("failure_mode"),
            }
        except Exception as e:
            return {
                "episode_idx": i,
                "task_idx": task_idx,
                "error": str(e),
            }

    def _save_snapshot(records: dict[int, dict]) -> tuple[list, list[bool], list, list[dict]]:
        trajectory_records: list = []
        successes: list[bool] = []
        fingerprints_data: list = []
        per_episode: list[dict] = []
        for i in sorted(records):
            rec = records[i]
            if "error" in rec:
                per_episode.append(
                    {
                        "episode_idx": i,
                        "task_idx": rec["task_idx"],
                        "error": rec["error"],
                    }
                )
                continue
            meta = {
                "domain": rec.get("domain"),
                "task_id": str(rec.get("task_id", "")),
                "split": rec.get("split"),
            }
            trajectory_records.append({"trajectory": rec["trajectory"], **meta})
            successes.append(bool(rec["success"]))
            fingerprints_data.append({"fingerprint": rec["fingerprint"], **meta})
            per_episode.append(
                {
                    "episode_idx": i,
                    "task_idx": rec["task_idx"],
                    **meta,
                    "success": bool(rec["success"]),
                    "reward": rec.get("reward"),
                    "n_turns": rec.get("n_turns"),
                    "failure_mode": rec.get("failure_mode"),
                    "fingerprint": rec["fingerprint"],
                }
            )
        success_rate = float(np.mean(successes)) if successes else 0.0
        with open(config.baseline_results_path, "w") as f:
            json.dump(
                {
                    "success_rate": success_rate,
                    "n_episodes": len(successes),
                    "n_errors": sum(1 for ep in per_episode if "error" in ep),
                    "collect_split": split,
                    "per_episode": per_episode,
                },
                f,
                indent=2,
            )
        with open(config.baseline_fingerprints_path, "w") as f:
            json.dump(fingerprints_data, f)
        with open(config.baseline_trajectories_path, "w") as f:
            json.dump(trajectory_records, f)
        return trajectory_records, successes, fingerprints_data, per_episode

    print(
        f"Collecting {n_roll} baseline episodes "
        f"(collect_split={split}, workers={n_workers}, save_every={save_every})..."
    )
    results_by_episode: dict[int, dict] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one, i): i for i in range(n_roll)}
        for fut in tqdm(as_completed(futures), total=n_roll):
            rec = fut.result()
            results_by_episode[int(rec["episode_idx"])] = rec
            completed += 1
            if save_every > 0 and (completed % save_every == 0):
                _save_snapshot(results_by_episode)

    trajectory_records, successes, fingerprints_data, per_episode = _save_snapshot(
        results_by_episode
    )
    for ep in per_episode:
        if "error" in ep:
            print(f"Episode {ep['episode_idx']} failed: {ep['error']}")

    success_rate = float(np.mean(successes)) if successes else 0.0
    print(f"\nBaseline success rate: {success_rate:.3f}")
    print(
        f"Saved baseline data ({len(trajectory_records)} trajectories, "
        f"{len(per_episode)} episode records)"
    )
    return trajectory_records, fingerprints_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        default=None,
        help="tau2 domain (default: config.taubench_domain)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of baseline episodes (default: one per task in the chosen split(s))",
    )
    parser.add_argument(
        "--collect-split",
        choices=("train", "test", "all"),
        default=None,
        help="train | test | all (default: config.baseline_collect_split, usually all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate, or rewrite merged artifacts, even if target JSON files already exist",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="For combined domains, merge existing per-domain baseline JSON files instead of rerolling.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel rollout workers (default: config.parallel_episode_workers)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Write JSON snapshots every N completed episodes (0 disables periodic saves)",
    )
    args = parser.parse_args()
    config = PersonaPoliciesConfig()
    if args.domain:
        config.taubench_domain = args.domain
        config.refresh_domain_artifact_paths()
    collect_baseline(
        config,
        n_episodes=args.n,
        collect_split=args.collect_split,
        force=args.force,
        workers=args.workers,
        save_every=args.save_every,
        merge_existing=args.merge_existing,
    )
