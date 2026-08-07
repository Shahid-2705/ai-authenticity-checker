"""
Train the classical ML tie-breaker (core_models/forensic_ml.py).

RandomForest on hand-crafted forensic features - not a deep model, runs on
CPU in minutes. See core_models/forensic_ml.py and forensic_features.py
for why this exists: training/diagnose_insight.py found InsightFace-style
face-swap-on-real-photo fakes clustering right at the 0.60 decision
boundary (0.38-0.68), which is a different failure mode than "the models
don't see it at all" - a classical texture/noise classifier focused on the
face region and its blending boundary is a cheap, independent second
opinion for exactly that uncertain zone.

Usage:
    .venv/Scripts/python.exe training/train_forensic_ml.py
"""

import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(ROOT_DIR, ".hf_cache", "datasets"))

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from training.dataset_portraits import load_portrait_dataset
from core_models.forensic_features import extract_forensic_features
from core_models.forensic_ml import ForensicMLClassifier

MAX_SAMPLES = 3000
MODEL_PATH = "models/forensic_ml.joblib"


def main():
    print("Loading portrait dataset for forensic ML training...")
    # face_align=False: extract_forensic_features does its own face
    # detection internally (it needs the whole image to compute the
    # whole/face/context regions separately).
    train_data, val_data = load_portrait_dataset(
        max_samples=MAX_SAMPLES, train_split=1.0, face_align=False,
    )
    all_data = train_data + val_data

    print(f"\nExtracting hand-crafted forensic features from {len(all_data)} images...")
    X, y = [], []
    for img, label in tqdm(all_data, desc="Extracting"):
        try:
            feats = extract_forensic_features(img)
        except Exception as e:
            print(f"  WARNING: feature extraction failed on one sample: {e}")
            continue
        X.append(feats)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    print(f"Feature matrix: {X.shape}, labels: {(y == 1).sum()} fake, {(y == 0).sum()} real")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining RandomForest...")
    clf = ForensicMLClassifier()
    clf.fit(X_train, y_train)

    preds = clf.model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"\nHeld-out val accuracy: {acc:.4f}")
    print(classification_report(y_val, preds, target_names=["real", "fake"]))
    print("Confusion matrix:\n", confusion_matrix(y_val, preds))

    importances = clf.model.feature_importances_
    from core_models.forensic_features import FEATURE_NAMES
    top10 = np.argsort(importances)[::-1][:10]
    print("\nTop 10 most important features:")
    for i in top10:
        print(f"  {FEATURE_NAMES[i]}: {importances[i]:.4f}")

    os.makedirs("models", exist_ok=True)
    clf.save(MODEL_PATH)
    print(f"\nSaved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
