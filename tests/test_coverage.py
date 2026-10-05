"""Unit tests for the shared Chamfer-coverage metric (ppol.evaluation)."""

import numpy as np
import pytest

from ppol.evaluation import chamfer_coverage, fingerprints_to_matrix
from ppol.fingerprinting import ALL_FEATURES, BehavioralFingerprint

N = len(ALL_FEATURES)


def test_perfect_coverage_scores_one():
    """P identical to H -> err = 0 -> score = 1."""
    H = np.ones((4, N))
    assert chamfer_coverage(H.copy(), H, d_ref=1.0) == pytest.approx(1.0)


def test_empty_inputs_score_zero():
    H = np.ones((4, N))
    assert chamfer_coverage(np.empty((0, N)), H, d_ref=1.0) == 0.0
    assert chamfer_coverage(H, np.empty((0, N)), d_ref=1.0) == 0.0
    assert chamfer_coverage(H, H, d_ref=0.0) == 0.0


def test_known_distance_hand_computed():
    """One human at 0, one persona at distance 1 on a single axis:
    err = 1 (h->p) + 1 (p->h) = 2, score = 1 - min(1, 2 / (2*d_ref))."""
    H = np.zeros((1, N))
    P = np.zeros((1, N)); P[0, 0] = 1.0
    assert chamfer_coverage(P, H, d_ref=2.0) == pytest.approx(1.0 - 2.0 / 4.0)
    assert chamfer_coverage(P, H, d_ref=0.5) == 0.0   # err caps at 2*d_ref


def test_far_personas_lower_score_than_near():
    rng = np.random.RandomState(0)
    H = rng.rand(6, N)
    near = H + 0.01 * rng.rand(6, N)
    far = H + 10.0
    d_ref = 1.0
    assert chamfer_coverage(near, H, d_ref) > chamfer_coverage(far, H, d_ref)


def test_standardization_is_applied_to_personas():
    """With std params, P is z-scored to match an already-standardized H."""
    rng = np.random.RandomState(1)
    raw = rng.rand(5, N) * 10
    mu, sd = raw.mean(axis=0), raw.std(axis=0)
    sd[sd < 1e-9] = 1.0
    H_std = (raw - mu) / sd
    # raw points passed as P + std params -> identical cloud -> perfect score
    assert chamfer_coverage(raw, H_std, d_ref=1.0, std_mu=mu, std_sd=sd) == pytest.approx(1.0)


def test_fingerprints_to_matrix_matches_to_vector():
    rng = np.random.RandomState(2)
    rows = rng.rand(3, N)
    fps = [BehavioralFingerprint(features=dict(zip(ALL_FEATURES, r))) for r in rows]
    M = fingerprints_to_matrix(fps, ALL_FEATURES)
    V = np.asarray([fp.to_vector(ALL_FEATURES) for fp in fps])
    assert np.array_equal(M, V)
    # plain dicts work too, missing features read as 0
    M2 = fingerprints_to_matrix([{ALL_FEATURES[0]: 1.0}], ALL_FEATURES)
    assert M2[0, 0] == 1.0 and M2[0, 1:].sum() == 0.0
