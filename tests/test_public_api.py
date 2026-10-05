"""Public API surface: every name in ppol.__all__ must be importable and
non-None. Guards against accidental drops/typos during refactors."""

from __future__ import annotations

import ppol


def test_all_listed_names_are_attributes():
    for name in ppol.__all__:
        assert hasattr(ppol, name), f"ppol.__all__ lists {name!r} but no such attribute"
        obj = getattr(ppol, name)
        assert obj is not None, f"ppol.{name} is None"


def test_expected_public_surface():
    expected = {
        "__version__",
        "PPolConfig",
        "default_config",
        "DEFAULT_EVOLVE_ITERATIONS",
        "Conversation",
        "EpisodeResult",
        "Task",
        "AgentFn",
        "EpisodeRunner",
        "SimpleEpisodeRunner",
        "DataLoader",
        "PPol",
        "compute_human_reference",
        "collect_baseline",
        "split_tasks",
        "train_discriminator",
        "benchmark_policy",
        "BehavioralFingerprint",
        "BehavioralFingerprintExtractor",
        "inject_persona_into_system_prompt",
    }
    assert set(ppol.__all__) == expected, (
        f"Public surface drift. extra={set(ppol.__all__) - expected} "
        f"missing={expected - set(ppol.__all__)}"
    )


def test_deprecation_alias_for_old_config_name():
    import warnings

    from ppol.config import PPolConfig as _New

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from ppol.config import PersonaPoliciesConfig

        assert PersonaPoliciesConfig is _New
        assert any(
            issubclass(w.category, DeprecationWarning) for w in caught
        ), "expected DeprecationWarning for PersonaPoliciesConfig alias"
