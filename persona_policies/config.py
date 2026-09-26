"""Central configuration for the Persona Policies framework."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


def _default_seed_behaviors() -> List[dict]:
    """Copy of ``evolution/initial_generator.DIVERSITY_AXES`` for config introspection."""
    from persona_policies.evolution.initial_generator import DIVERSITY_AXES

    return [dict(x) for x in DIVERSITY_AXES]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Only these may exist under ``outputs_root`` (see ``ensure_output_dirs``).
_CANONICAL_OUTPUT_TOP_LEVEL = frozenset({"reference_data", "training", "testing"})


def _safe_path_segment(s: str) -> str:
    """Sanitize a config value for use in filenames."""
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(s))


_DOMAIN_ALIASES = {
    "retail_airline": ("retail", "airline"),
    "airline_retail": ("airline", "retail"),
    "combined_retail_airline": ("retail", "airline"),
}


def domain_list(domain: str) -> List[str]:
    """Expand a domain selector into concrete tau2 domains.

    Examples:
      - ``retail`` -> [``retail``]
      - ``retail_airline`` -> [``retail``, ``airline``]
      - ``retail,airline`` -> [``retail``, ``airline``]
    """
    raw = str(domain or "retail").strip()
    alias = _DOMAIN_ALIASES.get(raw)
    if alias:
        return list(alias)
    parts = [
        p.strip()
        for p in raw.replace("+", ",").split(",")
        if p.strip()
    ]
    return parts or ["retail"]


def canonical_domain_name(domain: str) -> str:
    """Stable label for one or more domains, used in artifact names."""
    return "_".join(domain_list(domain))


def is_combined_domain(domain: str) -> bool:
    return len(domain_list(domain)) > 1


def _model_name_for_stem(model_id: str) -> str:
    """Short model-id segment for artifact filenames."""
    s = str(model_id or "").strip()
    return s.split("/", 1)[1] if "/" in s else s


def baseline_artifact_stem(
    domain: str,
    split: str,
    agent_model: str,
    user_model: str = "",
) -> str:
    """Stem for baseline JSON artifacts.

    Baseline trajectories depend on both the assistant model and the user
    simulator model, so include both in the filename. ``user_model`` is optional
    only for compatibility with older callers.
    """
    model_bits = [_safe_path_segment(_model_name_for_stem(agent_model))]
    if user_model:
        model_bits.append(f"user_{_safe_path_segment(_model_name_for_stem(user_model))}")
    return (
        f"baseline_{_safe_path_segment(canonical_domain_name(domain))}_{_safe_path_segment(split)}_"
        f"{'_'.join(model_bits)}"
    )


def human_fingerprints_filename(domain: str) -> str:
    """Domain-specific human reference filename."""
    return f"human_fingerprints_{_safe_path_segment(canonical_domain_name(domain))}.json"


def discriminator_filename(
    domain: str,
    agent_model: str = "",
    user_model: str = "",
) -> str:
    """Domain/model-specific discriminator filename.

    The discriminator is trained against baseline simulator fingerprints, which
    depend on the rollout agent and user simulator models.
    """
    bits = [_safe_path_segment(canonical_domain_name(domain))]
    if agent_model:
        bits.append(_safe_path_segment(_model_name_for_stem(agent_model)))
    if user_model:
        bits.append(f"user_{_safe_path_segment(_model_name_for_stem(user_model))}")
    return f"discriminator_{'_'.join(bits)}.pkl"


@dataclass
class PersonaPoliciesConfig:
    outputs_root: str = "persona_policies/outputs"

    # τ² checkout path. If τ²-bench is installed via pip, this path is only
    # used to locate the data directory (TAU2_DATA_DIR). Override by setting
    # TAU2_DATA_DIR directly or by passing taubench_root= to ensure_taubench_importable().
    taubench_root: str = field(
        default_factory=lambda: os.path.join(_REPO_ROOT, "tau2-bench")
    )
    # Human reference: τ²-bench human logs, filtered by ``taubench_domain``.
    tau_bench_human_path: str = "persona_policies/data/tau_bench_human.json"

    # Populated in __post_init__ (see ``ensure_output_dirs()`` for folder layout).
    reference_data_dir: str = field(init=False)
    human_fingerprints_cache: str = field(init=False)
    discriminator_model_path: str = field(init=False)
    baseline_stem: str = field(init=False)
    baseline_results_path: str = field(init=False)
    baseline_fingerprints_path: str = field(init=False)
    baseline_trajectories_path: str = field(init=False)
    training_root: str = field(init=False)
    simulations_dir: str = field(init=False)
    openevolve_output_dir: str = field(init=False)
    training_results_dir: str = field(init=False)
    testing_dir: str = field(init=False)
    testing_personas_dir: str = field(init=False)

    # Task splits
    task_split_evolution: str = "train"
    task_split_benchmark: str = "test"
    # Which τ² split(s) to roll when collecting baseline sim trajectories (artifacts filename tag).
    # ``all`` = train ∪ test tasks; ``train`` / ``test`` = one split only.
    baseline_collect_split: str = "all"
    val_fraction: float = 0.2

    # G(c, D, N) settings
    n_personas: int = 10
    eval_batch_size: int = 5
    n_episodes_per_persona_task: int = 1
    # Concurrency for τ² rollouts inside a batch (1 = sequential)
    parallel_episode_workers: int = 30

    # Evaluator weights — should sum to 1 so combined_score stays in [0, 1].
    lambda_human_likeness: float = 0.5
    lambda_intra_diversity: float = 0.5

    # Optional run label: if set, the training folder becomes ``training_<version>/`` instead of ``training/``
    # Edit in place or override at launch via ``--version`` on run_evolution.py (which sets ``PERSONA_POLICIES_VERSION`` env var).
    version: str = "v1"

    # Curriculum: N (number of personas) increases by epoch during training.
    # ``n_personas_schedule`` sets N for each epoch. Example: epoch 1 uses N=5, epoch 2 uses N=8, epoch 3+ uses N=10.
    curriculum: bool = True
    n_personas_schedule: List[Tuple[int, int]] = field(
        default_factory=lambda: [(1, 5), (2, 8), (3, 10)]
    )

    # Early-stop when the on-disk elite (checkpoint / final ``best/``) has not changed for this many OpenEvolve iterations. Set to 0 to disable.
    early_stop_iters_without_new_best: int = 20

    # ``benchmark --val-sweep`` auto-pick: include train_curve iterations where train HL exceeds this (unless overridden by ``--train-hl-threshold``).
    val_sweep_train_hl_threshold: float = 0.5

    # LLM models (LiteLLM provider prefixes: Gemini/Gemma via OpenRouter,
    # DeepSeek via Bedrock).
    llm_model: str = "openrouter/google/gemini-3-flash-preview"
    llm_fallback_models: List[str] = field(default_factory=lambda: [
        "gemini/gemini-3-flash-preview",
    ])
    # Evaluator reflection artifact (dynamic per-iteration feedback sent to the mutator).
    evolution_feedback_model: str = "openrouter/google/gemini-3-flash-preview"
    max_evolution_feedback_chars: int = 10000
    evolution_feedback_max_tokens: int = 2000

    # τ²-bench runtime
    taubench_user_model: str = "openrouter/qwen/qwen3-next-80b-a3b-instruct"
    taubench_agent_model: str = "openrouter/google/gemma-4-31b-it"
    tau2_llm_nl_assertions: str = "openrouter/google/gemma-4-31b-it"
    tau2_llm_env_interface: str = "openrouter/google/gemma-4-31b-it"
    taubench_domain: str = "retail"
    seed: int = 42
    max_steps: int = 100
    max_turns_per_episode: Optional[int] = 30
    # Training-only cap on tau2 orchestrator steps. This includes user, assistant,
    # and environment/tool steps; validation uses ``max_turns_per_episode``.
    train_max_steps_per_episode: Optional[int] = 20

    # Diversity axes D (canonical definitions live in ``evolution/initial_generator.py``).
    seed_behaviors: List[dict] = field(default_factory=_default_seed_behaviors)

    def __post_init__(self) -> None:
        env_v = os.environ.get("PERSONA_POLICIES_VERSION", "")
        if env_v:
            self.version = env_v
        env_domain = os.environ.get("PERSONA_POLICIES_DOMAIN", "")
        if env_domain:
            self.taubench_domain = env_domain
        env_domains = os.environ.get("PERSONA_POLICIES_DOMAINS", "")
        if env_domains:
            self.taubench_domain = env_domains
        env_user_model = os.environ.get("PERSONA_POLICIES_TAUBENCH_USER_MODEL", "")
        if env_user_model:
            self.taubench_user_model = env_user_model
        env_agent_model = os.environ.get("PERSONA_POLICIES_TAUBENCH_AGENT_MODEL", "")
        if env_agent_model:
            self.taubench_agent_model = env_agent_model
        env_val = os.environ.get("PERSONA_POLICIES_VAL_FRACTION", "")
        if env_val:
            self.val_fraction = float(env_val)
        r = self.outputs_root
        ref = os.path.join(r, "reference_data")
        tr_name = "training" if not self.version else f"training_{_safe_path_segment(self.version)}"
        tr = os.path.join(r, tr_name)
        tst = os.path.join(r, "testing")
        self.reference_data_dir = ref
        self.refresh_domain_artifact_paths(ref)
        self.training_root = tr
        self.simulations_dir = os.path.join(tr, "simulations")
        self.openevolve_output_dir = os.path.join(tr, "openevolve")
        self.training_results_dir = os.path.join(tr, "results")
        self.testing_dir = tst
        self.testing_personas_dir = os.path.join(tst, "personas")

    def _set_baseline_artifact_paths(self, reference_data_dir: str) -> None:
        stem = baseline_artifact_stem(
            self.taubench_domain,
            self.baseline_collect_split,
            self.taubench_agent_model,
            self.taubench_user_model,
        )
        self.baseline_stem = stem
        self.baseline_results_path = os.path.join(reference_data_dir, f"{stem}_results.json")
        self.baseline_fingerprints_path = os.path.join(
            reference_data_dir, f"{stem}_fingerprints.json"
        )
        self.baseline_trajectories_path = os.path.join(
            reference_data_dir, f"{stem}_trajectories.json"
        )

    def refresh_domain_artifact_paths(self, reference_data_dir: Optional[str] = None) -> None:
        """Recompute all domain-dependent reference artifact paths.

        Call this after changing ``taubench_domain`` or ``taubench_agent_model`` on an
        existing config instance.
        """
        ref = reference_data_dir or self.reference_data_dir
        self.human_fingerprints_cache = os.path.join(
            ref,
            human_fingerprints_filename(self.taubench_domain),
        )
        self.discriminator_model_path = os.path.join(
            ref,
            discriminator_filename(
                self.taubench_domain,
                self.taubench_agent_model,
                self.taubench_user_model,
            ),
        )
        self._set_baseline_artifact_paths(ref)

    def refresh_baseline_artifact_paths(self) -> None:
        """Recompute baseline JSON paths after changing ``baseline_collect_split``.

        Prefer ``refresh_domain_artifact_paths`` after changing domain/model.
        """
        self._set_baseline_artifact_paths(
            os.path.join(self.outputs_root, "reference_data"),
        )

    def ensure_output_dirs(self) -> None:
        """Create the canonical tree under ``outputs_root`` and warn on stray top-level dirs.

        Creates: ``reference_data/``, ``training/{simulations,openevolve,results}/``,
        ``testing/{personas}/``.

        Emits one ``UserWarning`` per extra top-level directory (anything other than
        ``reference_data``, ``training``, ``testing``, ``training_*`` prefixes, or dot-directories).
        """
        for d in (
            self.reference_data_dir,
            self.simulations_dir,
            self.openevolve_output_dir,
            self.training_results_dir,
            self.testing_dir,
            self.testing_personas_dir,
        ):
            os.makedirs(d, exist_ok=True)
        self._warn_noncanonical_top_level_dirs()

    def _warn_noncanonical_top_level_dirs(self) -> None:
        root = Path(self.outputs_root)
        if not root.is_dir():
            return
        for child in root.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if name in _CANONICAL_OUTPUT_TOP_LEVEL or name.startswith((".", "training_")):
                continue
            warnings.warn(
                f"Unexpected directory under {root}: {name!r} — expected only "
                f"{sorted(_CANONICAL_OUTPUT_TOP_LEVEL)}. Move or remove it.",
                UserWarning,
                stacklevel=2,
            )
