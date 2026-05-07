"""Sanity tests for behavioral fingerprinting."""

from __future__ import annotations

from persona_policies.fingerprinting import (
    ALL_FEATURES,
    BehavioralFingerprintExtractor,
    dice_coefficient,
)


def _trace(turns: list[tuple[str, str]]) -> list[dict]:
    return [{"role": r, "content": c} for r, c in turns]


def test_terse_user_has_shorter_words_per_turn():
    ext = BehavioralFingerprintExtractor()
    terse = _trace(
        [
            ("assistant", "What is your order id?"),
            ("user", "123"),
            ("assistant", "Thanks."),
            ("user", "ok"),
        ]
    )
    verbose = _trace(
        [
            ("assistant", "What is your order id?"),
            (
                "user",
                "Sure, my order id is 123 and I wanted to add that I really need help quickly.",
            ),
        ]
    )
    ft = ext.compute_fingerprint(terse)
    fv = ext.compute_fingerprint(verbose)
    assert ft.features["words_per_turn"] < fv.features["words_per_turn"]


def test_politeness_detected():
    ext = BehavioralFingerprintExtractor()
    polite = _trace([("assistant", "Hi"), ("user", "Please help me, thank you.")])
    blunt = _trace([("assistant", "Hi"), ("user", "Fix my order now.")])
    assert (
        ext.compute_fingerprint(polite).features["politeness_rate"]
        >= ext.compute_fingerprint(blunt).features["politeness_rate"]
    )


def test_empty_user_turns_yield_zero_vector():
    ext = BehavioralFingerprintExtractor()
    trace = _trace([("assistant", "Hello")])
    fp = ext.compute_fingerprint(trace)
    assert fp.features["words_per_turn"] == 0.0


def test_fingerprint_has_all_named_features():
    ext = BehavioralFingerprintExtractor()
    names = ext.feature_names()
    tr = _trace(
        [
            ("assistant", "Hi?"),
            ("user", "I need help with order 42 please."),
        ]
    )
    fp = ext.compute_fingerprint(tr)
    for n in names:
        assert n in fp.features
    assert set(names) == set(ALL_FEATURES)


def test_dice_coefficient_identity():
    assert dice_coefficient(0.0, 0.0) == 1.0
    assert dice_coefficient(5.0, 5.0) == 1.0
    assert 0.0 < dice_coefficient(1.0, 2.0) < 1.0
