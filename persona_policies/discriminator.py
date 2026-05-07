"""
Behavioral Discriminator
========================
Random forest on behavioral fingerprints: human vs τ² baseline simulator.
"""

from __future__ import annotations

import os
import pickle
from typing import List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from persona_policies.fingerprinting import BehavioralFingerprint, BehavioralFingerprintExtractor


class BehavioralDiscriminator:
    def __init__(self):
        self.extractor = BehavioralFingerprintExtractor()
        self.feature_names = self.extractor.feature_names()
        self.scaler = StandardScaler()
        self.clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.is_trained = False

    def fingerprints_to_matrix(self, fingerprints: List[BehavioralFingerprint]) -> np.ndarray:
        return np.array([fp.to_vector(self.feature_names) for fp in fingerprints])

    def train(
        self,
        human_fingerprints: List[BehavioralFingerprint],
        simulator_fingerprints: List[BehavioralFingerprint],
        verbose: bool = True,
    ):
        """Train the discriminator on human vs. simulator fingerprints."""
        X_human = self.fingerprints_to_matrix(human_fingerprints)
        X_sim = self.fingerprints_to_matrix(simulator_fingerprints)
        X = np.vstack([X_human, X_sim])
        y = np.array([1] * len(human_fingerprints) + [0] * len(simulator_fingerprints))

        X_scaled = self.scaler.fit_transform(X)

        if verbose and len(y) >= 4:
            n_splits = min(5, max(2, len(y) // 2))
            cv_scores = cross_val_score(
                self.clf, X_scaled, y, cv=n_splits, scoring="roc_auc"
            )
            print(f"Discriminator CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print("  (AUC > 0.7 means we can distinguish human from simulator — good!)")
            print("  (AUC < 0.6 means our features aren't capturing the right signal)")

        self.clf.fit(X_scaled, y)
        self.is_trained = True

        if verbose:
            imp = self.clf.feature_importances_
            sorted_idx = np.argsort(imp)[::-1]
            print("\nTop discriminative features (RandomForest importance):")
            for i in sorted_idx[:12]:
                print(f"  {self.feature_names[i]:40s}: {imp[i]:.4f}")

    def predict_human_probability(self, fingerprint: BehavioralFingerprint) -> float:
        """Return P(human) for a given fingerprint."""
        if not self.is_trained:
            raise RuntimeError("Discriminator not trained yet.")
        x = fingerprint.to_vector(self.feature_names).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        return float(self.clf.predict_proba(x_scaled)[0, 1])

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "scaler": self.scaler,
                    "clf": self.clf,
                    "feature_names": self.feature_names,
                    "is_trained": self.is_trained,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "BehavioralDiscriminator":
        disc = cls()
        with open(path, "rb") as f:
            d = pickle.load(f)
        disc.scaler = d["scaler"]
        disc.clf = d["clf"]
        disc.feature_names = d["feature_names"]
        disc.is_trained = d["is_trained"]
        return disc
