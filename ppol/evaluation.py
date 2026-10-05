"""Shared evaluation metrics for ppol.

Single source of truth for the two-sided **Chamfer behavioral-coverage** score.
Both the evolution fitness (``ppol.evolution.fitness``) and every example's
held-out benchmark call :func:`chamfer_coverage`, so the metric
is defined exactly once. Dice alignment already lives in :mod:`ppol.fingerprinting`.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def fingerprints_to_matrix(fingerprints, feature_names: Sequence[str]) -> np.ndarray:
    """Stack fingerprint feature values into an ``(n, len(feature_names))`` matrix.

    Accepts either :class:`~ppol.fingerprinting.BehavioralFingerprint` objects
    (reads ``.features``) or plain ``{feature: value}`` dicts — missing features
    read as ``0.0``. Identical to ``fp.to_vector(feature_names)`` per row.
    """
    rows = []
    for fp in fingerprints:
        feats = fp.features if hasattr(fp, "features") else fp
        rows.append([float(feats.get(k, 0.0)) for k in feature_names])
    return np.asarray(rows, dtype=np.float64)


def chamfer_coverage(
    P: np.ndarray,
    H: np.ndarray,
    d_ref: float,
    *,
    std_mu: Optional[np.ndarray] = None,
    std_sd: Optional[np.ndarray] = None,
) -> float:
    """Two-sided Chamfer coverage of persona points ``P`` vs human cloud ``H``.

    Rescaled to ``[0, 1]`` (higher = better coverage)::

        err   = mean_h min_p ||h - p||  +  mean_p min_h ||p - h||
        score = max(0, 1 - min(1, err / (2 * d_ref)))

    ``cover_humans`` (every real human has a near persona) + ``stay_on_manifold``
    (every persona sits near a real human), both in the same feature space, L2.
    ``d_ref`` is the mean pairwise distance within ``H`` — dividing by ``2*d_ref``
    makes a score of 1 mean "typical human spread".

    If ``std_mu``/``std_sd`` are given, ``H`` is assumed already z-scored and ``P``
    is z-scored here to match (ppol-evolution / ColBench / WildChat). Pass neither
    for the raw variant (τ²-bench). Returns ``0.0`` for an empty cloud / ``d_ref<=0``.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim == 1:
        P = P.reshape(1, -1)
    if P.size == 0 or H.shape[0] == 0 or d_ref <= 0:
        return 0.0
    if std_mu is not None:
        P = (P - std_mu) / std_sd
    from scipy.spatial.distance import cdist

    D = cdist(H, P, metric="euclidean")
    err = float(D.min(axis=1).mean() + D.min(axis=0).mean())
    return float(max(0.0, 1.0 - min(1.0, err / (2.0 * d_ref))))


# ---------------------------------------------------------------------------
# Shared held-out benchmark harness
#
# Used by the ColBench and WildChat examples (and reusable for any new
# benchmark): run the base-simulator / DP / initial / evolved-PPol conditions on
# a held-out test split, score each with P(human) + Chamfer coverage + Dice, and
# render the results. The only per-benchmark inputs are the ``runner`` and
# display ``title`` — everything below is identical across benchmarks.
# ---------------------------------------------------------------------------

import math as _math

_PKG = __import__("pathlib").Path(__file__).resolve().parent
_INITIAL_GENERATOR = _PKG / "evolution/initial_generator.py"
_DP_GENERATOR = _PKG / "evolution/baselines/direct_llm_personas.py"
_ORDER = ["default", "dp", "initial", "ppol"]
_NAMES = {"default": "Base-simulator", "dp": "DP Personas",
          "initial": "PPol: Initial", "ppol": "PPol: Evolved"}


def _ci95(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    sd = _math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return 1.96 * sd / _math.sqrt(n)


def evaluate_simulators(
    runner,
    *,
    output_dir: str,
    title: str,
    name: str,
    n_personas: int = 10,
    n_workers: int = 12,
    conditions=None,
    from_cache: bool = False,
    ppol_program=None,
    bench_subdir: str = "benchmark",
    scatter_conditions=None,
    sim_model_label=None,
    feature_names=None,
    coverage_ref: str = "train+test",
    standardize: bool = True,
    discriminator_path=None,
):
    """Run the held-out benchmark and write results/plots under
    ``output_dir/bench_subdir``. ``runner`` is a benchmark-specific
    ``EpisodeRunner`` already configured for the test split; ``title`` (e.g.
    ``"ColBench (backend/code)"``) and ``name`` (e.g. ``"ColBench"``) are labels."""
    import json
    import time
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    import numpy as np
    from scipy.spatial.distance import pdist

    from ppol import PPol
    from ppol.discriminator import BehavioralDiscriminator
    from ppol.fingerprinting import (ALL_FEATURES, BehavioralFingerprint,
                                     BehavioralFingerprintExtractor,
                                     HumanBehavioralDistribution,
                                     compute_aggregate_dice_alignment)

    feat = list(feature_names) if feature_names is not None else list(ALL_FEATURES)
    conditions = conditions if conditions is not None else list(_ORDER)

    out = Path(output_dir)
    rd = out / "reference_data"
    bdir = out / bench_subdir
    bdir.mkdir(parents=True, exist_ok=True)

    test = runner.get_tasks()

    # --- Human references (H_ref policy) ----------------------------------
    # Evolution always fits coverage on the TRAIN split (separate code path).
    # Here, per `coverage_ref`, the METHOD rows are scored against:
    #   "train+test" (default) — the full human cloud (train ∪ held-out test)
    #   "train"                — train humans only (matches evolution's cloud)
    #   "test"                 — held-out test humans only
    # The "Humans" ceiling row is ALWAYS scored against the TRAIN cloud, and the
    # z-score standardizer is ALWAYS from TRAIN (canonical human scale, matching
    # evolution's fitness) so Chamfer isn't dominated by the length features.
    def _load_indiv(path):
        return json.loads(Path(path).read_text())["fingerprints"]

    def _as_matrix(rows):
        return np.asarray([[fp.get(f, 0.0) for f in feat] for fp in rows], float)

    def _dist(rows):
        return HumanBehavioralDistribution(
            mean={f: float(np.mean([fp.get(f, 0.0) for fp in rows])) for f in feat},
            feature_names=feat, n_dialogues=len(rows))

    train_indiv = _load_indiv(rd / "human_fingerprints.individual.json")
    _test_path = rd / "human_fingerprints.test.individual.json"
    has_test = _test_path.exists()
    test_indiv = _load_indiv(_test_path) if has_test else train_indiv

    if coverage_ref == "train":
        method_indiv = train_indiv
    elif coverage_ref == "test":
        method_indiv = test_indiv
    else:  # "train+test"
        method_indiv = train_indiv + (test_indiv if has_test else [])

    if standardize:
        std_mu = _as_matrix(train_indiv).mean(axis=0)
        std_sd = _as_matrix(train_indiv).std(axis=0)
        std_sd[std_sd < 1e-9] = 1.0
    else:                      # raw feature space (as used by examples/tau2bench)
        std_mu = std_sd = None

    def _std(mat):
        mat = np.asarray(mat, float)
        return mat if std_mu is None else (mat - std_mu) / std_sd

    import os as _os
    _NCL = int(_os.environ.get("PERSONA_POLICIES_N_HUMAN_CLUSTERS", 0) or 0)

    def _cluster(mat_std):
        if 0 < _NCL < mat_std.shape[0]:
            from sklearn.cluster import KMeans
            return KMeans(n_clusters=_NCL, n_init=10, random_state=0).fit(mat_std).cluster_centers_.astype(float)
        return mat_std

    def _cham(fps, Hc, dr):
        """Two-sided Chamfer of feature-dicts ``fps`` vs cloud ``Hc`` (already z-scored)."""
        P = fingerprints_to_matrix(fps, feat)
        return chamfer_coverage(P, Hc, dr, std_mu=std_mu, std_sd=std_sd)

    # Method-row coverage cloud + Dice distribution (per coverage_ref policy).
    H = _cluster(_std(_as_matrix(method_indiv)))
    dref = float(pdist(H).mean()) if H.shape[0] >= 2 else 1.0
    dref = dref if dref > 1e-9 else 1.0
    human_dist = _dist(method_indiv)
    _tr = train_indiv                       # scatter "Humans" cloud (train)
    indiv = test_indiv                      # held-out humans for the disc test

    extractor = BehavioralFingerprintExtractor()
    disc = BehavioralDiscriminator.load(str(discriminator_path or (rd / "discriminator.pkl")))

    # "Humans" row: held-out TEST humans scored against the TRAIN cloud (always
    # train ref) — analytic, no episodes. Coverage = Chamfer(test vs train).
    humans_row = None
    if has_test:
        train_dist = _dist(train_indiv)
        H_tr = _cluster(_std(_as_matrix(train_indiv)))
        dref_tr = float(pdist(H_tr).mean()) if H_tr.shape[0] >= 2 else 1.0
        dref_tr = dref_tr if dref_tr > 1e-9 else 1.0
        humans_row = {
            "p_human": float(np.mean([disc.predict_human_probability(BehavioralFingerprint(features=fp)) for fp in test_indiv])),
            "chamfer": _cham(test_indiv, H_tr, dref_tr),
            "dice": compute_aggregate_dice_alignment([BehavioralFingerprint(features=fp) for fp in test_indiv], train_dist),
            "n": len(test_indiv),
        }

    def render_paper(res: dict) -> str:
        def line(nm, r):
            hl, cov, d = r["p_human"], r["chamfer"], r["dice"]
            return (f"| {nm} | {hl:.3f} | {cov:.3f} | {0.5*hl+0.5*cov:.3f} | "
                    f"{d['D1']*100:.1f} | {d['D2']*100:.1f} | {d['D3']*100:.1f} | "
                    f"{d['D4']*100:.1f} | {d['overall']*100:.1f} |")
        s = ("| Method | HL ↑ | Coverage ↑ | Score ↑ | D₁ | D₂ | D₃ | D₄ | USI |\n"
             "|---|--:|--:|--:|--:|--:|--:|--:|--:|\n")
        if humans_row is not None:
            s += line("Humans", humans_row) + "\n"
        for k in _ORDER:
            if k in res:
                s += line(_NAMES[k], res[k]) + "\n"
        return s

    def episodes_for(label):
        cache = bdir / f"episodes_{label}.json"
        if from_cache or cache.exists():
            return json.loads(cache.read_text())
        if label == "default":
            jobs = [(t, "") for t in test for _ in range(n_personas)]
        else:
            best = {"initial": str(_INITIAL_GENERATOR),
                    "dp": str(_DP_GENERATOR)}.get(label, ppol_program or None)
            p = PPol(output_dir=str(out))
            jobs = None
            for attempt in range(4):
                try:
                    jobs = [(item["task"], persona["text"])
                            for item in p.generate(tasks=test, n=n_personas, best_program=best)
                            for persona in item["personas"]]
                    break
                except Exception as e:
                    if attempt == 3:
                        raise
                    print(f"  [{label}] generation retry {attempt+1}: {e}", flush=True)
                    time.sleep(10 * (attempt + 1))

        def _run(task, text):
            try:
                r = runner.run_episode(task, persona_policy=text)
                fp = extractor.compute_fingerprint(r.trajectory)
                return {"task_id": task.task_id, "features": fp.features,
                        "p_human": float(disc.predict_human_probability(fp)),
                        "success": bool(r.success), "reward": float(r.reward),
                        "n_turns": int(r.n_turns),
                        "persona": text, "trajectory": r.trajectory}
            except Exception:
                return None

        eps = []
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_run, t, txt) for t, txt in jobs]
            for i, f in enumerate(as_completed(futs), 1):
                r = f.result()
                if r is not None:
                    eps.append(r)
                if i % 40 == 0:
                    print(f"  [{label}] {i}/{len(jobs)}", flush=True)
        cache.write_text(json.dumps(eps))
        return eps

    def metrics(eps):
        ph = [e["p_human"] for e in eps]
        sc = [1.0 if e["success"] else 0.0 for e in eps]
        rw = [e["reward"] for e in eps]
        by_task = defaultdict(list)
        for e in eps:
            by_task[e["task_id"]].append(e["features"])
        chamfer = float(np.mean([_cham(g, H, dref) for g in by_task.values() if g]))
        dice = compute_aggregate_dice_alignment(
            [BehavioralFingerprint(features=e["features"]) for e in eps], human_dist)
        return {"n": len(eps),
                "p_human": sum(ph) / len(ph), "p_human_ci": _ci95(ph),
                "chamfer": chamfer, "dice": dice,
                "success": sum(sc) / len(sc), "success_ci": _ci95(sc),
                "reward": sum(rw) / len(rw), "reward_ci": _ci95(rw),
                "turns": sum(e["n_turns"] for e in eps) / len(eps)}

    def render(res, partial):
        s = (f"\nResults{' (PARTIAL)' if partial else ''}. Held-out {title}: "
             f"{len(test)} test tasks × {n_personas} episodes/task.\n"
             "P(human) and Chamfer = the two optimization metrics, as-is on the test set "
             "(Chamfer = two-sided coverage vs human cloud, [0,1]↑, per-task averaged). "
             "D1–D4 = Sørensen–Dice alignment per dimension (D1 style, D2 disclosure, "
             "D3 clarification, D4 error-reaction); avg = mean(D1..D4). ±95% CI.\n\n"
             "| Simulator | N | P(human)↑ | Chamfer↑ | D1 | D2 | D3 | D4 | D-avg | Agent success | Agent reward | Turns |\n"
             "|---|--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|\n")
        for k in _ORDER:
            if k not in res:
                continue
            r = res[k]; d = r["dice"]
            s += (f"| {_NAMES[k]} | {r['n']} | {r['p_human']:.3f}±{r['p_human_ci']:.3f} | {r['chamfer']:.3f} | "
                  f"{d['D1']:.3f} | {d['D2']:.3f} | {d['D3']:.3f} | {d['D4']:.3f} | {d['overall']:.3f} | "
                  f"{r['success']*100:.1f}%±{r['success_ci']*100:.1f} | "
                  f"{r['reward']:.3f}±{r['reward_ci']:.3f} | {r['turns']:.1f} |\n")
        if "default" in res and "ppol" in res:
            b, e = res["default"], res["ppol"]
            s += (f"\nΔ (ppol − default): P(human) {e['p_human']-b['p_human']:+.3f}, "
                  f"Chamfer {e['chamfer']-b['chamfer']:+.3f}, "
                  f"success {(e['success']-b['success'])*100:+.1f} pts, reward {e['reward']-b['reward']:+.3f}\n")
        return s

    order = [c for c in _ORDER if c in conditions]
    eps_by_cond = {}
    res = {}
    for i, label in enumerate(order):
        eps = episodes_for(label)
        eps_by_cond[label] = eps
        res[label] = metrics(eps)
        partial = i < len(order) - 1
        summary = render(res, partial)
        (bdir / "results.md").write_text(summary)
        (bdir / "results_summary.md").write_text(render_paper(res))
        (bdir / "results.json").write_text(json.dumps(
            {"n_test_tasks": len(test), "n_personas_per_task": n_personas,
             "complete": not partial, "conditions": res,
             "humans_row": humans_row}, indent=2))
        print(f"  ✓ {label} done ({i+1}/{len(_ORDER)})", flush=True)
        print(summary, flush=True)

    # Held-out discriminator test: humans (positive) vs persona-free sim on the
    # TEST tasks (negative) — the disc's test-set generalization.
    try:
        if "default" not in eps_by_cond:
            raise RuntimeError("need the 'default' condition for the held-out disc test")
        from sklearn.metrics import roc_auc_score
        pos = [float(disc.predict_human_probability(BehavioralFingerprint(features=fp))) for fp in indiv]
        neg = [e["p_human"] for e in eps_by_cond["default"]]
        y = [1] * len(pos) + [0] * len(neg)
        auc = float(roc_auc_score(y, pos + neg))
        print(f"\nDiscriminator held-out test (human vs persona-free sim on {len(neg)} test episodes): AUC={auc:.3f}")
        (bdir / "discriminator_test.json").write_text(json.dumps(
            {"auc_test": auc, "n_human": len(pos), "n_sim_test": len(neg)}, indent=2))
    except Exception as e:
        print(f"[disc test skipped: {e}]")

    # Behavioral-fingerprint scatter via the shared PCA renderer.
    try:
        from ppol.analysis.plot_discriminator_scatter import plot_fingerprint_pca
        _STYLE = {"default": ("#c62828", "o"), "dp": ("#9467bd", "P"),
                  "initial": ("#1f77b4", "D"), "ppol": ("#ff7f0e", "s")}
        _keep = ({c.strip() for c in scatter_conditions.split(",")}
                 if scatter_conditions else set(order))
        clouds = [{"label": "Humans", "color": "#52c41a", "marker": "^", "zorder": 3,
                   "fps": [BehavioralFingerprint(features=fp) for fp in _tr],
                   "legend": f"Humans (n={len(_tr)})"}]
        for k in order:
            if k not in _keep:
                continue
            fps = [BehavioralFingerprint(features=e["features"]) for e in eps_by_cond[k]]
            color, mk = _STYLE[k]
            clouds.append({"label": _NAMES[k], "fps": fps, "color": color, "marker": mk,
                           "zorder": 3, "legend": f"{_NAMES[k]} (n={len(fps)})"})
        plot_fingerprint_pca(
            clouds, bdir / "scatter.png", basis_labels=["Humans"],
            title="Behavioral fingerprint space (standardized, PCA)\n"
                  f"{name} — sim: {(sim_model_label or 'default').split('/')[-1]}")
    except Exception as e:
        print(f"[scatter skipped: {e}]")

    print(f"Saved → {bdir}/results.{{md,json}} + episodes_*.json + scatter.png")
    return {"conditions": res, "humans_row": humans_row, "n_test_tasks": len(test)}
