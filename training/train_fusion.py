"""
Train the Fusion MLP on validation-set predictions from all models.

This script:
  1. Loads all trained component models (ViT, Texture, Frequency, DINOv2,
     EfficientNet Auth, Face Deepfake) plus the Forensic heuristic
  2. Runs them on a held-out validation set to collect per-model scores
  3. Trains the FusionMLP + ModelCalibrator to optimally combine scores

The fusion layer learns:
  - Per-model temperature calibration (Task 5)
  - Optimal combination weights via MLP (Task 1)

Previously this only combined 4 of the available signals (vit, texture,
forensic, frequency), silently excluding dino/efficientnet_auth/face from
the actual fused decision even though they were computed and displayed at
inference time. Face Deepfake in particular is the strongest individual
model (90.3% held-out accuracy) - excluding it meant its correct calls on
real images were discarded before the final score. Now uses all 7.

Input:  FusionMLP.MODEL_NAMES order:
        [vit, texture, forensic, frequency, dino, efficientnet_auth, face]
Output: models/fusion_mlp.pth

Usage:
    python training/train_fusion.py
"""

import sys
import os
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ["HF_HOME"] = os.path.join(ROOT_DIR, ".hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(ROOT_DIR, ".hf_cache", "datasets")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from core_models.fusion_mlp import FusionMLP
from core_models.efficientnet_texture import EfficientNetTexture
from core_models.frequency_cnn import FrequencyCNN, fft_to_tensor
from core_models.dinov2_auth_model import DINOv2AuthModel
from core_models.efficientnet_auth_model import EfficientNetAuthModel
from core_models.face_deepfake_model import FaceDeepfakeModel
from training.dataset_portraits import (
    load_portrait_dataset, VAL_TRANSFORM,
)

# -------- CONFIG --------
FUSION_EPOCHS = 50
FUSION_LR = 1e-3
FUSION_BATCH = 64
N_INPUTS = 7
MAX_SAMPLES = 2000          # Held-out set for fusion training
MODEL_PATH = "models/fusion_mlp.pth"
MODELS_DIR = "models"
CHECKPOINT_PATH = os.path.join(ROOT_DIR, ".hf_cache", "fusion_collect_checkpoint.pt")
# -------------------------


def _load_vit(device):
    """Load ViT deepfake detector from HuggingFace."""
    try:
        from transformers import ViTForImageClassification, ViTImageProcessor
        model_id = "prithivMLmods/Deep-Fake-Detector-v2-Model"
        model = ViTForImageClassification.from_pretrained(model_id).to(device)
        processor = ViTImageProcessor.from_pretrained(model_id)
        model.eval()
        return model, processor
    except Exception as e:  # Broad catch: HF model loading can fail in many ways
        print(f"WARNING: Could not load ViT: {e}")
        return None, None


def _load_component(device, model_class, filename, name):
    """Load a trained local component model (state_dict-only checkpoint)."""
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        print(f"WARNING: {path} not found ({name} excluded from fusion training)")
        return None
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model.eval()
    return model


def collect_predictions(device):
    """
    Run all component models on a held-out dataset and collect
    (per-model scores, true label) pairs for fusion training.

    Returns:
        scores_tensor: (N, 7) — matches FusionMLP.MODEL_NAMES order:
                       [vit, texture, forensic, frequency, dino,
                        efficientnet_auth, face]
        labels_tensor: (N,) — 0=real, 1=fake
    """
    print("\n--- Collecting model predictions for fusion training ---\n")

    # Load models
    vit_model, vit_processor = _load_vit(device)
    texture_model = _load_component(device, EfficientNetTexture, "efficient.pth", "texture")
    freq_model = _load_component(device, FrequencyCNN, "frequency.pth", "frequency")
    dino_model = _load_component(device, DINOv2AuthModel, "dinov2_auth_model.pth", "dino")
    eff_auth_model = _load_component(
        device, EfficientNetAuthModel, "efficientnet_auth_model.pth", "efficientnet_auth"
    )
    face_model = _load_component(device, FaceDeepfakeModel, "image_face_model.pth", "face")

    # Load dataset — skip samples used by component models to avoid data leakage.
    # dataset_portraits.py now has many sources (JamieWithofs was dropped for
    # being unreliable). Component model consumption per class per source:
    #   EfficientNet texture: MAX_SAMPLES=3000 -> 750/class/source
    #   Frequency CNN:        MAX_SAMPLES=20000 -> 5000/class/source
    # Skip past the larger of the two so neither model's training data leaks
    # into fusion's "held-out" set. DINOv2/EfficientNet Auth/Face were
    # trained on entirely separate HF streams (not dataset_portraits.py),
    # so no additional skip is needed for them.
    val_data, _ = load_portrait_dataset(
        max_samples=MAX_SAMPLES,
        train_split=1.0,
        face_align=True,
        skip_per_class=4000,  # Skip component training data
        seed=123,             # Different seed from component training
    )

    val_transform = VAL_TRANSFORM

    # Checkpointed: this loop runs every trained model on ~2000 images on
    # CPU (~20+ min) and has been getting killed by something outside this
    # script partway through on repeated attempts, losing all progress each
    # time. val_data's order is fully deterministic (fixed seeds throughout
    # dataset_portraits.py's collection), so resuming at the same index on
    # restart is safe - it's the same image every time.
    all_scores = []
    all_labels = []
    start_idx = 0
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)
        all_scores = ckpt["scores"]
        all_labels = ckpt["labels"]
        start_idx = len(all_scores)
        print(f"Resuming from checkpoint: {start_idx}/{len(val_data)} already collected")

    print(f"Collecting predictions from {len(val_data)} samples...")

    for i in tqdm(range(start_idx, len(val_data)), desc="Scoring",
                  initial=start_idx, total=len(val_data)):
        img, label = val_data[i]
        # Order matches FusionMLP.MODEL_NAMES
        scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        rgb_img = img.convert("RGB")

        # ViT score
        if vit_model is not None and vit_processor is not None:
            try:
                inputs = vit_processor(images=rgb_img, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = vit_model(**inputs).logits
                    probs = torch.softmax(logits, dim=1)
                    fake_idx = [
                        k for k, v in vit_model.config.id2label.items()
                        if "fake" in v.lower()
                    ]
                    scores[0] = (
                        probs[0][fake_idx[0]].item()
                        if fake_idx else probs[0][1].item()
                    )
            except Exception:  # Broad catch: ViT inference varies across image inputs
                pass

        # Shared 224x224 tensor for texture/dino/efficientnet_auth/face
        try:
            tensor = val_transform(rgb_img).unsqueeze(0).to(device)
        except Exception:
            tensor = None

        # EfficientNet texture score
        if texture_model is not None and tensor is not None:
            try:
                with torch.no_grad():
                    scores[1] = texture_model(tensor).item()
            except Exception:  # Broad catch: model can fail on edge-case images
                pass

        # Forensic score (heuristic — kept as input feature)
        try:
            from core.pipeline import forensic_score
            scores[2] = forensic_score(img)
        except Exception:  # Broad catch: forensic heuristic may fail on unusual images
            scores[2] = 0.0

        # Frequency CNN score
        if freq_model is not None:
            try:
                fft_tensor = fft_to_tensor(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    scores[3] = freq_model(fft_tensor).item()
            except Exception:  # Broad catch: FFT can fail on edge-case images
                pass

        # DINOv2 score
        if dino_model is not None and tensor is not None:
            try:
                with torch.no_grad():
                    scores[4] = dino_model(tensor).item()
            except Exception:  # Broad catch: DINO can fail on edge-case images
                pass

        # EfficientNet Auth score
        if eff_auth_model is not None and tensor is not None:
            try:
                with torch.no_grad():
                    scores[5] = eff_auth_model(tensor).item()
            except Exception:  # Broad catch: model can fail on edge-case images
                pass

        # Face Deepfake score (model outputs P(real); fusion wants P(fake))
        if face_model is not None and tensor is not None:
            try:
                with torch.no_grad():
                    real_prob = face_model(tensor).item()
                    scores[6] = 1.0 - real_prob
            except Exception:  # Broad catch: face model can fail on non-face images
                pass

        all_scores.append(scores)
        all_labels.append(float(label))

        if (i + 1) % 100 == 0:
            torch.save({"scores": all_scores, "labels": all_labels}, CHECKPOINT_PATH)

    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
    labels_tensor = torch.tensor(all_labels, dtype=torch.float32)

    # Collection finished cleanly - clear the checkpoint so a genuinely
    # fresh run (e.g. after swapping in different component models) doesn't
    # accidentally resume stale scores computed against the old ones.
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    print(f"\nCollected {len(all_scores)} prediction vectors")
    names = FusionMLP.MODEL_NAMES
    means = ", ".join(
        f"{name}={scores_tensor[:, i].mean():.3f}" for i, name in enumerate(names)
    )
    print(f"Score means: {means}")
    print(f"Labels: {(labels_tensor == 1).sum().item()} fake, "
          f"{(labels_tensor == 0).sum().item()} real")

    return scores_tensor, labels_tensor


def train_fusion(scores, labels, device):
    """
    Train FusionMLP on collected model predictions.

    Args:
        scores: (N, 7) per-model scores, matching FusionMLP.MODEL_NAMES order
        labels: (N,) ground truth (0=real, 1=fake)
    """
    print("\n--- Training Fusion MLP ---\n")

    # Seeded: previously this had no fixed seed, so the train/val split and
    # FusionMLP's weight init were drawn fresh from PyTorch's global RNG
    # every run. Two runs on the IDENTICAL reverted component models landed
    # 48.3% vs 36.7% on training/eval_image_benchmark.py purely from this
    # randomness - not a real difference in the underlying models, just
    # non-reproducible fusion training. Fixed so retrains are comparable.
    torch.manual_seed(42)

    # Train/val split
    n = len(scores)
    split = int(n * 0.8)
    indices = torch.randperm(n)

    train_scores = scores[indices[:split]]
    train_labels = labels[indices[:split]]
    val_scores = scores[indices[split:]]
    val_labels = labels[indices[split:]]

    train_ds = TensorDataset(train_scores, train_labels)
    val_ds = TensorDataset(val_scores, val_labels)

    train_loader = DataLoader(train_ds, batch_size=FUSION_BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=FUSION_BATCH, shuffle=False)

    # Model
    model = FusionMLP(n_inputs=N_INPUTS).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=FUSION_LR, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=FUSION_EPOCHS, eta_min=1e-5)

    best_val_loss = float("inf")
    patience = 0

    for epoch in range(FUSION_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_scores, batch_labels in train_loader:
            batch_scores = batch_scores.to(device)
            batch_labels = batch_labels.unsqueeze(1).to(device)

            preds = model(batch_scores)
            loss = criterion(preds, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_scores, batch_labels in val_loader:
                batch_scores = batch_scores.to(device)
                batch_labels = batch_labels.unsqueeze(1).to(device)
                preds = model(batch_scores)
                val_loss += criterion(preds, batch_labels).item()
                correct += ((preds > 0.5).float() == batch_labels).sum().item()
                total += batch_labels.size(0)

        avg_val = val_loss / max(len(val_loader), 1)
        val_acc = correct / max(total, 1)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{FUSION_EPOCHS}] "
                  f"| Train: {avg_loss:.4f} | Val: {avg_val:.4f} | Acc: {val_acc:.4f}")

            # Print learned temperatures
            temps = model.calibrator.temperatures.data.cpu().numpy()
            temp_str = ", ".join(
                f"{name}={t:.3f}" for name, t in zip(FusionMLP.MODEL_NAMES, temps)
            )
            print(f"  Temperatures: {temp_str}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience += 1
            if patience >= 10:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Print final fusion weights
    print(f"\nFusion MLP trained. Saved to: {MODEL_PATH}")
    print(f"Learned temperatures: {model.calibrator.temperatures.data.cpu().numpy()}")
    print(f"Fusion layer weights:\n"
          f"  Layer 1: {model.fusion[0].weight.data.cpu().numpy()}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    scores, labels = collect_predictions(device)
    train_fusion(scores, labels, device)


if __name__ == "__main__":
    main()
