"""Discriminator: training + save/load roundtrip on tiny synthetic data."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ppol.discriminator import BehavioralDiscriminator
from ppol.fingerprinting import ALL_FEATURES, BehavioralFingerprint


def _fake_fp(seed: int) -> BehavioralFingerprint:
    rng = np.random.default_rng(seed)
    return BehavioralFingerprint(
        features={name: float(rng.uniform(0, 1)) for name in ALL_FEATURES}
    )


def test_train_predict_roundtrip(tmp_path: Path):
    # Separable: human seeds 0..19, sim seeds 100..119 — different rng streams.
    humans = [_fake_fp(i) for i in range(20)]
    sims = [_fake_fp(100 + i) for i in range(20)]

    disc = BehavioralDiscriminator()
    disc.train(humans, sims)
    assert disc.is_trained

    # Predict on a held-out human-like fingerprint.
    p = disc.predict_human_probability(_fake_fp(7))
    assert 0.0 <= p <= 1.0

    # Save → load via pickle (the format used by train_discriminator.py).
    out = tmp_path / "disc.pkl"
    with out.open("wb") as f:
        pickle.dump(
            {
                "scaler": disc.scaler,
                "clf": disc.clf,
                "feature_names": disc.feature_names,
                "is_trained": disc.is_trained,
            },
            f,
        )

    with out.open("rb") as f:
        state = pickle.load(f)
    restored = BehavioralDiscriminator()
    restored.scaler = state["scaler"]
    restored.clf = state["clf"]
    restored.feature_names = state["feature_names"]
    restored.is_trained = state["is_trained"]

    p2 = restored.predict_human_probability(_fake_fp(7))
    assert abs(p - p2) < 1e-9
