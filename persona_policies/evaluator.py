"""
Multi-Objective Evaluator for Persona Policies
=================================================================
Scores evolved persona policies using:
  - Human likeness: mean per-episode ``P(human)`` from a trained RF discriminator.
    Each episode's D1–D4 fingerprint is scored, and the mean is used as the persona's ``human_likeness``.
    Sørensen–Dice vs the cached human mean is still computed and logged as a diagnostic
    but does not enter the fitness. The domain-specific discriminator is **required** at evaluator startup.
  - Behavioral coverage across persona variants (from fitness / batch).

Reflection text for the mutator uses LiteLLM / Bedrock when configured.
"""

from __future__ import annotations

import logging
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from persona_policies.config import PersonaPoliciesConfig
from persona_policies.config import canonical_domain_name
from persona_policies.discriminator import BehavioralDiscriminator
from persona_policies.fingerprinting import (
    BehavioralFingerprint,
    BehavioralFingerprintExtractor,
    DIMENSION_MAP,
    HumanBehavioralDistribution,
    REGEX_FEATURES,
    compute_aggregate_dice_alignment,
)
from persona_policies.injector import TaubenchEpisodeRunner


_eval_debug_quiet_done = False


def _quiet_eval_debug_noise() -> None:
    """Reduce τ² + HTTP/LLM client DEBUG spam during evaluation."""
    global _eval_debug_quiet_done
    if _eval_debug_quiet_done:
        return
    _eval_debug_quiet_done = True

    os.environ.setdefault("LITELLM_LOG", "ERROR")

    if os.environ.get("PERSONA_POLICIES_VERBOSE_HTTP", "").lower() not in (
        "1", "true", "yes",
    ):
        for name in (
            "LiteLLM", "litellm", "httpx", "httpcore",
            "openai", "urllib3", "botocore", "boto3",
            "tau2",
        ):
            logging.getLogger(name).setLevel(logging.WARNING)

    if os.environ.get("PERSONA_POLICIES_VERBOSE_TAU2", "").lower() in (
        "1", "true", "yes",
    ):
        return
    try:
        from loguru import logger as loguru_logger
    except ImportError:
        return
    loguru_logger.disable("tau2")

    try:
        import litellm
        litellm.set_verbose = False  # type: ignore[misc]
        if hasattr(litellm, "suppress_debug_info"):
            litellm.suppress_debug_info = True  # type: ignore[attr-defined]
    except Exception:
        pass


def _default_human_dist(
    extractor: BehavioralFingerprintExtractor,
) -> HumanBehavioralDistribution:
    names = extractor.regex_feature_names()
    return HumanBehavioralDistribution(
        mean={f: 0.0 for f in names},
        feature_names=names,
        n_dialogues=0,
    )


class PersonaPolicyEvaluator:
    """Evaluates persona policies via τ² episodes."""

    def __init__(self, config: PersonaPoliciesConfig):
        _quiet_eval_debug_noise()
        self.config = config
        self.extractor = BehavioralFingerprintExtractor()

        cache_path = Path(config.human_fingerprints_cache)
        if cache_path.is_file():
            print("Loading human behavioral distribution...")
            self.human_dist = HumanBehavioralDistribution.load(str(cache_path))
            meta_bits = []
            if self.human_dist.source:
                meta_bits.append(f"source={self.human_dist.source!r}")
            if self.human_dist.domain:
                meta_bits.append(f"domain={self.human_dist.domain!r}")
            if meta_bits:
                print("  " + ", ".join(meta_bits))
            if self.human_dist.domain and self.human_dist.domain != canonical_domain_name(config.taubench_domain):
                raise ValueError(
                    f"Human reference domain mismatch: file {cache_path} has "
                    f"domain={self.human_dist.domain!r}, config has "
                    f"domain={config.taubench_domain!r}."
                )
        else:
            print(
                f"WARNING: {cache_path} not found — using neutral prior. "
                f"Run: python persona_policies/scripts/compute_human_reference.py "
                f"--domain {config.taubench_domain}"
            )
            self.human_dist = _default_human_dist(self.extractor)

        disc_path = Path(config.discriminator_model_path)
        if not disc_path.is_file():
            raise FileNotFoundError(
                f"Behavioral discriminator is required but missing: {disc_path}\n"
                "Train it with: python persona_policies/scripts/train_discriminator.py "
                f"--domain {config.taubench_domain}"
            )
        self._discriminator = BehavioralDiscriminator.load(str(disc_path))
        print(
            "Loaded behavioral discriminator — human_likeness = mean P(human) "
            "from trained classifier."
        )

        self.runner = TaubenchEpisodeRunner(config)
        self.runner.use_split(config.task_split_evolution)
        self.task_indices = self.runner.get_available_task_indices()[:20]

    def _collect_episodes(
        self,
        persona_policy_text: str,
        eval_task_indices: List[int],
        n_episodes_per_task: int,
        pbar_desc: str = "τ² episodes",
    ) -> Tuple[List, List[bool], List[Optional[str]], List[Dict[str, Any]]]:
        """Run τ² episodes. Returns (trajectories, successes, failure_modes, raw_episode_records).

        ``raw_episode_records`` captures reward, n_turns, task_idx per rollout
        so callers can persist full detail.
        """
        trajectories: List = []
        successes: List = []
        failure_modes: List = []
        raw_records: List[Dict[str, Any]] = []

        total_eps = len(eval_task_indices) * n_episodes_per_task
        _tqdm_env = os.environ.get("PERSONA_POLICIES_TQDM", "").lower()
        if _tqdm_env in ("0", "false", "no"):
            use_tqdm = False
        elif _tqdm_env in ("1", "true", "yes"):
            use_tqdm = True
        else:
            use_tqdm = sys.stdout.isatty()
        pbar = tqdm(
            total=total_eps,
            desc=pbar_desc,
            unit="ep",
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.05,
            disable=not use_tqdm,
        )
        try:
            for task_idx in eval_task_indices:
                for _ in range(n_episodes_per_task):
                    try:
                        result = self.runner.run_episode(
                            task_idx=task_idx,
                            persona_policy_text=persona_policy_text,
                            verbose=False,
                        )
                        trajectories.append(result["trajectory"])
                        successes.append(result["success"])
                        failure_modes.append(result.get("failure_mode"))
                        raw_records.append({
                            "task_idx": task_idx,
                            "reward": result.get("reward"),
                            "n_turns": result.get("n_turns"),
                            "success": result["success"],
                            "failure_mode": result.get("failure_mode"),
                        })
                        pbar.set_postfix_str(f"task {task_idx}", refresh=False)
                    except Exception as e:
                        msg = f"Warning: Episode failed with error: {e}"
                        if use_tqdm:
                            tqdm.write(msg)
                        else:
                            print(msg, flush=True)
                    finally:
                        pbar.update(1)
        finally:
            pbar.close()
        return trajectories, successes, failure_modes, raw_records

    def _metrics_from_episodes(
        self,
        persona_policy_text: str,
        trajectories: List,
        successes: List[bool],
        failure_modes: List[Optional[str]],
        *,
        persona_policies: Optional[List[str]] = None,
        task_contexts: Optional[List[str]] = None,
        intra_set_diversity: Optional[float] = None,
        n_variants: Optional[int] = None,
        precomputed_fingerprints: Optional[List[BehavioralFingerprint]] = None,
        scoring_weights: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Dict[str, Any], List[BehavioralFingerprint]]:
        """Core scoring: discriminator-based human-likeness + behavioral coverage.

        ``human_likeness`` = mean ``P(human)`` across the persona's episodes,
        produced by the trained RF discriminator on each episode's D1–D4
        fingerprint. Sørensen–Dice against the cached human mean is still
        computed and stored as ``human_likeness_d1..d4`` for diagnostics, but is
        **not** part of the fitness.

        ``combined_score`` = ``λ_h·human_likeness + λ_b·behavioral_coverage``.
        Defaults have ``λ_h + λ_b = 1`` so the sum stays in ``[0, 1]`` when both
        subscores are in ``[0, 1]``.
        """
        if not trajectories:
            err = {
                "combined_score": 0.0,
                "human_likeness": 0.0,
                "error": 1.0,
            }
            return err, []

        # --- Regex fingerprints (reuse precomputed when caller already has them) ---
        if (
            precomputed_fingerprints is not None
            and len(precomputed_fingerprints) == len(trajectories)
        ):
            persona_fingerprints = list(precomputed_fingerprints)
        else:
            persona_fingerprints = [
                self.extractor.compute_fingerprint(traj) for traj in trajectories
            ]

        # --- Dice (kept only as a diagnostic; NOT used in fitness) ---
        dice_scores = compute_aggregate_dice_alignment(
            persona_fingerprints, self.human_dist
        )

        # --- human_likeness = mean P(human) from the trained RF discriminator ---
        per_ep_p_human = [
            self._discriminator.predict_human_probability(fp) for fp in persona_fingerprints
        ]
        human_likeness = float(np.mean(per_ep_p_human))

        # --- Build per-episode detail records ---
        # Per-episode Dice is intentionally omitted: individual dialogues are
        # not expected to match the population mean. The
        # ``fingerprint`` field here is what actually feeds the aggregate.
        per_episode: List[Dict[str, Any]] = []
        for i in range(len(trajectories)):
            ep: Dict[str, Any] = {
                "episode_idx": i,
                "success": successes[i] if i < len(successes) else None,
                "failure_mode": failure_modes[i] if i < len(failure_modes) else None,
                "n_turns": len(trajectories[i]),
                "fingerprint": persona_fingerprints[i].to_dict(),
            }
            ep["p_human"] = per_ep_p_human[i]
            per_episode.append(ep)

        # --- Combined score ---
        cfg = self.config
        intra = intra_set_diversity if intra_set_diversity is not None else 0.0
        if scoring_weights is None:
            lambda_h = float(cfg.lambda_human_likeness)
            lambda_b = float(cfg.lambda_intra_diversity)
        else:
            lambda_h, lambda_b = (float(scoring_weights[0]), float(scoring_weights[1]))
        score = (
            lambda_h * human_likeness
            + lambda_b * float(intra)
        )

        metrics: Dict[str, Any] = {
            "combined_score": score,
            "human_likeness": human_likeness,
            "human_likeness_d1": dice_scores["D1"],
            "human_likeness_d2": dice_scores["D2"],
            "human_likeness_d3": dice_scores["D3"],
            "human_likeness_d4": dice_scores["D4"],
            "persona_success_rate": float(np.mean(successes)) if successes else 0.5,
            "n_episodes": len(trajectories),
            "failure_mode_distribution": self._summarize_failure_modes(failure_modes),
            "per_episode": per_episode,
        }
        if intra_set_diversity is not None:
            metrics["intra_set_diversity"] = float(intra_set_diversity)
        metrics["score_weight_human_likeness"] = float(lambda_h)
        metrics["score_weight_intra_diversity"] = float(lambda_b)
        if n_variants is not None:
            metrics["n_persona_variants_evaluated"] = int(n_variants)
        return metrics, persona_fingerprints

    def evaluate(self, persona_policy_text: str) -> Tuple[Dict[str, Any], List]:
        """Run τ² episodes; returns ``(metrics, trajectories)``.
        """
        print(f"\n{'='*60}")
        print(f"Evaluating persona policy ({len(persona_policy_text)} chars):")
        print(
            persona_policy_text[:300] + "..."
            if len(persona_policy_text) > 300
            else persona_policy_text
        )
        print("=" * 60)

        n_tasks = min(self.config.eval_batch_size, len(self.task_indices))
        rng = np.random.default_rng(self.config.seed)
        idx = list(self.task_indices)
        rng.shuffle(idx)
        eval_task_indices = idx[:n_tasks]

        trajectories, successes, failure_modes, raw_records = self._collect_episodes(
            persona_policy_text,
            eval_task_indices,
            self.config.n_episodes_per_persona_task,
        )

        if not trajectories:
            err = {
                "combined_score": 0.0,
                "human_likeness": 0.0,
                "error": 1.0,
            }
            return err, []

        task_ctx_list = [
            self.runner.get_task_scenario_text(int(r["task_idx"]))
            for r in raw_records
        ]
        if len(task_ctx_list) != len(trajectories):
            task_ctx_list = [""] * len(trajectories)

        metrics, _ = self._metrics_from_episodes(
            persona_policy_text,
            trajectories,
            successes,
            failure_modes,
            task_contexts=task_ctx_list,
        )

        # Merge task_idx / reward / n_turns from raw_records into per_episode
        per_ep = metrics.get("per_episode", [])
        for i, ep in enumerate(per_ep):
            if i < len(raw_records):
                ep.update({k: v for k, v in raw_records[i].items() if k not in ep})

        print("\nEvaluation Results:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:30s}: {v:.4f}")

        return metrics, trajectories

    def _trajectory_excerpt(
        self,
        traj: List,
        *,
        max_turns: int = 10,
        max_chars: int = 5000,
    ) -> str:
        lines: List[str] = []
        for t in traj[:max_turns]:
            role = str(t.get("role", "") or "")
            c = t.get("content") or ""
            if not isinstance(c, str):
                c = str(c)
            if role.lower() == "assistant" and len(c) > 250:
                c = f"{c[:100]} ... {c[-100:]}"
            if c.strip():
                lines.append(f"{role.upper()}: {c}")
        return "\n".join(lines)[:max_chars]

    def build_feature_comparison_block(
        self,
        persona_fingerprints_by_task: List[List[BehavioralFingerprint]],
    ) -> str:
        """Debug-only per-feature human/persona comparison table.

        Persona values are averaged across personas within each task first,
        then those task means are averaged across the batch. This should stay
        out of mutator/reflection prompts to avoid proxy-feature hacking.
        """
        per_task = [fps for fps in (persona_fingerprints_by_task or []) if fps]
        if not per_task or not self.human_dist or not self.human_dist.mean:
            return ""
        human_mean = self.human_dist.mean
        human_std = self.human_dist.std or {}

        per_task_mean = np.array(
            [
                [
                    float(np.mean([fp.features.get(f, 0.0) for fp in fps]))
                    for f in REGEX_FEATURES
                ]
                for fps in per_task
            ],
            dtype=np.float64,
        )
        persona_mean = per_task_mean.mean(axis=0)
        persona_std = per_task_mean.std(axis=0, ddof=0)

        def _fmt(val: float, std: Optional[float]) -> str:
            if std is None or not np.isfinite(std):
                return f"{val:.3f}"
            return f"{val:.3f}±{std:.3f}"

        feat_w = max(len("feature"), max(len(f) for f in REGEX_FEATURES))
        human_cells = [
            _fmt(float(human_mean.get(f, 0.0)), float(human_std[f]) if f in human_std else None)
            for f in REGEX_FEATURES
        ]
        persona_cells = [
            _fmt(float(persona_mean[i]), float(persona_std[i]))
            for i, _ in enumerate(REGEX_FEATURES)
        ]
        human_w = max(len("human"), max(len(c) for c in human_cells))
        persona_w = max(len("persona"), max(len(c) for c in persona_cells))

        n_tasks = per_task_mean.shape[0]
        n_eps = sum(len(fps) for fps in per_task)
        dim_labels = ", ".join(
            f"{d}={len(feats)}" for d, feats in DIMENSION_MAP.items()
        )
        header = (
            f"# Fingerprint comparison  n_tasks={n_tasks}  n_episodes={n_eps}  "
            f"features=19 ({dim_labels})  human_ref=n_dialogues={self.human_dist.n_dialogues}\n"
            "# Persona column: mean across personas per task, then mean±std across tasks.\n"
            "# Human column:   mean±std across real human dialogues."
        )

        rows: List[str] = [
            f"{'feature':<{feat_w}} | {'human':<{human_w}} | {'persona':<{persona_w}}",
            f"{'-'*feat_w}-+-{'-'*human_w}-+-{'-'*persona_w}",
        ]
        for i, f in enumerate(REGEX_FEATURES):
            rows.append(
                f"{f:<{feat_w}} | {human_cells[i]:<{human_w}} | {persona_cells[i]:<{persona_w}}"
            )
        return f"{header}\n\n" + "\n".join(rows)

    @staticmethod
    def _format_fingerprint_block(fp: Any) -> str:
        """Render a single-episode fingerprint as 4-5 aligned lines (5 feats/line)."""
        if fp is None:
            return ""
        if hasattr(fp, "features"):
            feats = fp.features
        elif isinstance(fp, dict):
            feats = fp.get("features") if "features" in fp else fp
        else:
            return ""
        if not isinstance(feats, dict):
            return ""
        parts = [f"{name}={float(feats.get(name, 0.0)):.3f}" for name in REGEX_FEATURES]
        chunk = 5
        return "\n".join(
            "  " + "  ".join(parts[i : i + chunk]) for i in range(0, len(parts), chunk)
        )

    def llm_evolution_reflection(
        self,
        metrics: Dict[str, Any],
        persona_episodes: List[Dict[str, Any]],
        *,
        task_persona_contexts: Optional[List[Dict[str, Any]]] = None,
        max_samples: int = 4,
        rng: Optional[random.Random] = None,
    ) -> str:
        """Generate LLM-written prose reflection for the mutator.

        ``persona_episodes`` is an episode-aligned list of dicts with keys
        ``persona`` (roleplay instruction text), ``trajectory`` (τ² turns),
        ``fingerprint`` (single-episode ``BehavioralFingerprint`` or dict), and
        ``p_human`` (discriminator score for that episode). The method samples
        the two highest- and two lowest-scoring episodes by ``p_human`` and
        shows each persona's trajectory excerpt, its episode fingerprint, and
        its per-episode ``human_likeness`` (= ``p_human``).
        """
        rng = rng or random.Random()

        pool: List[Dict[str, Any]] = []
        for ep in persona_episodes or []:
            pp = str(ep.get("persona") or "").strip()
            traj = ep.get("trajectory")
            if not pp or not traj:
                continue
            excerpt = self._trajectory_excerpt(
                traj, max_turns=20, max_chars=2400
            ).strip()
            if not excerpt:
                continue
            pool.append(
                {
                    "persona": pp,
                    "excerpt": excerpt,
                    "fingerprint": ep.get("fingerprint"),
                    "p_human": ep.get("p_human"),
                }
            )
        if not pool:
            return ""

        scored = [
            item for item in pool
            if isinstance(item.get("p_human"), (int, float))
        ]
        if scored:
            scored.sort(key=lambda x: float(x["p_human"]))
            low = scored[:2]
            high = list(reversed(scored[-2:]))
            sampled = (high + low)[:max_samples]
        else:
            rng.shuffle(pool)
            sampled = pool[:max_samples]

        from persona_policies.llm_bedrock import completion_text

        # Only the fitness-meaningful scalars — d1..d4 Dice diagnostics are
        # excluded to keep the reflection focused on what the fitness reads.
        reflection_metrics_keys = (
            "combined_score",
            "human_likeness",
            "intra_set_diversity",
        )
        numeric: Dict[str, float] = {}
        for mkey in reflection_metrics_keys:
            v = metrics.get(mkey)
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                numeric[mkey] = float(v)
        metrics_block = "\n".join(f"- {k}: {v:.4f}" for k, v in numeric.items())

        pair_sections: List[str] = []
        for i, ep in enumerate(sampled, start=1):
            persona_clip = ep["persona"][:1200]
            if len(ep["persona"]) > 1200:
                persona_clip += "\n[...]"
            p_human_str = (
                f"{float(ep['p_human']):.4f}"
                if isinstance(ep.get("p_human"), (int, float))
                else "n/a"
            )
            fp_block = self._format_fingerprint_block(ep.get("fingerprint"))
            fp_section = (
                f"Behavioral features:\n{fp_block}\n\n"
                if fp_block
                else ""
            )
            pair_sections.append(
                f"## Sample {i} (human_likeness={p_human_str})\n"
                f"Persona policy:\n{persona_clip}\n\n"
                f"{fp_section}"
                f"Dialogue:\n{ep['excerpt']}"
            )
        pairs_block = "\n\n".join(pair_sections)

        task_sections: List[str] = []
        for ti, task in enumerate(task_persona_contexts or [], start=1):
            task_id = str(task.get("task_id") or ti)
            original = task.get("original_context")
            if isinstance(original, (dict, list)):
                ctx = json.dumps(original, indent=2, ensure_ascii=False)
            else:
                ctx = str(original or task.get("task_context") or "")
            ctx = ctx[:2200]
            personas = task.get("personas") or []
            persona_lines: List[str] = []
            for p in personas:
                if not isinstance(p, dict):
                    continue
                idx = p.get("persona_idx")
                h = p.get("human_likeness")
                h_s = f"{float(h):.4f}" if isinstance(h, (int, float)) else "n/a"
                axis = p.get("axis_placement")
                behaviors = (
                    [str(k) for k, v in axis.items() if bool(v)]
                    if isinstance(axis, dict)
                    else []
                )
                persona_lines.append(
                    f"  - p{idx}: human_likeness={h_s}; "
                    f"behaviors: {json.dumps(behaviors, ensure_ascii=False)}"
                )
            task_sections.append(
                f"## Task {task_id}\n"
                f"Original context: {ctx}\n"
                f"Generated Personas:\n"
                f"{chr(10).join(persona_lines)}"
            )
        task_context_block = "\n\n".join(task_sections)

        model = self.config.evolution_feedback_model

        prompt = f"""You are evaluating a set of personas representing human populations in provided task scenarios. 

Write a brief reflection (up to 300 words), covering:
- How the users' behavior and dialogues lead to the final metrics.
- Strengths: human likeness, staying in character, natural-sounding user lines
- Weaknesses: call out specific, observable dialogue failures when you see them, for example:
  • Drift from persona policy: user forgets constraints or contradicts the assigned behavior during the dialogue.
  • Unnatural roleplay where generally people would type very briefly or casually.
  • Overly cooperative behavior lacking any realistic friction—no typos or natural pushback when appropriate (missing things that real humans would typically do)

- Use the human likeness probability and other features to explain *why* personas scored high or low given the task.
- Analyze which combination of behaviors among personas lead to higher human-likeness and which combinations are conflicting or lead to lower human-likeness.
- Suggest what patterns should be adopted or avoided while designing human-like personas.

Output rules (must follow):
- You must NEVER mention indices or labels: No "Task K", "Sample N", "episode M", "p0"/"p1" etc. or similar. Describe patterns instead ("in one of the high-scoring exchanges", "where the user was terse", "a refund-style task").
- Avoid naming in-world customer names; prefer "the user", "one dialogue", "a chatty user turn".
- You may refer qualitatively to the scenario without numbering.

---
# Metrics
{metrics_block}

---
# This batch of task scenarios:
{task_context_block}

---
# Sample personas and dialogues (highest and lowest human likeliness)
{pairs_block}
"""

        try:
            text = completion_text(
                model,
                [{"role": "user", "content": prompt}],
                temperature=0.25,
                max_tokens=self.config.evolution_feedback_max_tokens,
            )
            return (text or "").strip()
        except Exception as e:
            return f"(reflection failed: {str(e)[:200]})"

    def _summarize_failure_modes(
        self, failure_modes: List[Optional[str]]
    ) -> Dict[str, float]:
        from collections import Counter
        counts = Counter(m for m in failure_modes if m is not None)
        total = len(failure_modes)
        if total == 0:
            return {}
        return {mode: count / total for mode, count in counts.items()}
