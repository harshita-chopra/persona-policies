"""
ppol — Persona Policies
=======================

Evolve diverse, human-like persona policies for evaluating dialogue agents on
any task, any domain, any episode runner.

Quick-start
-----------
::

    pip install ppol

    from ppol import Task, SimpleEpisodeRunner, PPol

    tasks = [Task(task_id="t1", description="Return a defective laptop.")]

    def my_agent(messages):
        return "How can I help?"

    runner = SimpleEpisodeRunner(agent=my_agent, user_sim_model="gpt-4o-mini")
    p = PPol(output_dir="./out")
    p.evolve(runner=runner, train_tasks=tasks, val_tasks=tasks,
             human_reference_path=..., baseline_path=..., discriminator_path=...,
             iterations=20)  # see ppol/README.md for building these references

End-to-end examples
-------------------
See ``examples/tau2bench/`` (τ²-bench retail + airline), ``examples/colbench/``,
and ``examples/wildchat/`` for full pipelines that subclass ``EpisodeRunner``
and drive the evolve → benchmark loop.

See ``ppol/README.md`` for the full pipeline (human reference → baseline →
discriminator → evolve → benchmark).
"""

# Config
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("ppol")
except PackageNotFoundError:          # running from a source checkout
    __version__ = "0.0.0.dev0"

from ppol.config import DEFAULT_EVOLVE_ITERATIONS, PPolConfig, default_config

# Core types and abstractions
from ppol.core.types import Conversation, EpisodeResult, Task
from ppol.core.runner import AgentFn, EpisodeRunner, SimpleEpisodeRunner

# Data utilities
from ppol.data.loader import DataLoader

# Pipeline
from ppol.pipeline import (
    PPol,
    benchmark_policy,
    collect_baseline,
    compute_human_reference,
    split_tasks,
    train_discriminator,
)

# Fingerprinting (advanced inspection)
from ppol.fingerprinting import (
    BehavioralFingerprint,
    BehavioralFingerprintExtractor,
)

# Helper used when implementing a custom EpisodeRunner (wraps a user-sim system
# prompt with the persona policy text).
from ppol.injection import inject_persona_into_system_prompt

__all__ = [
    "__version__",
    # Config
    "PPolConfig",
    "default_config",
    "DEFAULT_EVOLVE_ITERATIONS",
    # Core
    "Conversation",
    "EpisodeResult",
    "Task",
    "AgentFn",
    "EpisodeRunner",
    "SimpleEpisodeRunner",
    # Data
    "DataLoader",
    # Pipeline
    "PPol",
    "compute_human_reference",
    "collect_baseline",
    "split_tasks",
    "train_discriminator",
    "benchmark_policy",
    # Fingerprinting
    "BehavioralFingerprint",
    "BehavioralFingerprintExtractor",
    # Custom-runner helper
    "inject_persona_into_system_prompt",
]
