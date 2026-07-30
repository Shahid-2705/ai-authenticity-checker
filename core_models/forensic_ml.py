"""
Classical ML tie-breaker: RandomForest on hand-crafted forensic features
(core_models/forensic_features.py), not another deep net.

Purpose: training/diagnose_insight.py found InsightFace-style face-swap
fakes clustering right around the 0.60 decision boundary (0.38-0.68 across
every held-out sample) - the deep ensemble isn't blind to them, it's
just uncertain. A RandomForest on texture/noise/color statistics needs far
less data and is far less prone to overfitting than another deep model
would be for this narrow purpose, and gives a genuinely independent second
opinion precisely when the primary signal is weak.

Only consulted by core/pipeline.py when the fused risk score lands near
the boundary - never used as a standalone verdict.

Saves as: models/forensic_ml.joblib
"""

import joblib
from sklearn.ensemble import RandomForestClassifier

from core_models.forensic_features import extract_forensic_features


class ForensicMLClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba_fake(self, pil_img):
        """Returns P(fake) in [0, 1] for a single PIL image."""
        feats = extract_forensic_features(pil_img).reshape(1, -1)
        return float(self.model.predict_proba(feats)[0][1])

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
