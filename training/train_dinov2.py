import sys
import os
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Redirect HF cache to D: drive (C: has no space)
os.environ["HF_HOME"] = os.path.join(ROOT_DIR, ".hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(ROOT_DIR, ".hf_cache", "datasets")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from core_models.dinov2_auth_model import DINOv2AuthModel
from training.dataset_portraits import (
    load_portrait_dataset, PortraitDataset, TRAIN_TRANSFORM, VAL_TRANSFORM,
)

# ================= CONFIG =================
BATCH_SIZE = 16
EPOCHS = 15
BACKBONE_LR = 1e-5
HEAD_LR = 1e-3
TRAIN_SPLIT = 0.85
MAX_SAMPLES = 20000
EARLY_STOPPING_PATIENCE = 5
LABEL_SMOOTHING = 0.05
MODEL_PATH = "models/dinov2_auth_model.pth"
# ========================================


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------- Load dataset via the shared multi-source loader --------
    # Now pulls from PORTRAIT_SOURCES (StyleGAN/GAN portraits, many HF-
    # streaming sources with reservoir sampling) PLUS OpenRL/DeepFakeFace
    # (diffusion-generated: SD Inpainting/text2img, InsightFace) —
    # previously this trained on the single
    # Hemg/AI-Generated-vs-Real-Images-Datasets source only, which is
    # GAN-style and taught DINOv2 nothing about diffusion output
    # (training/eval_image_benchmark.py found the whole ensemble at 30.8%
    # accuracy on diffusion fakes vs 83.3% on real photos).
    # face_align=False: tried True on the theory that InsightFace fakes
    # (face-swaps onto the same real IMDB-WIKI photos as the real baseline -
    # see training/diagnose_insight.py) would be easier to catch with a
    # tight face crop isolating the manipulated region. Measured result was
    # the opposite: InsightFace accuracy dropped 6.7%->3.3% and inpainting
    # dropped too (training/eval_image_benchmark.py, round 4 vs round 3) -
    # a tight crop likely excludes the blend boundary itself, which sits at
    # the hairline/jaw/neck edge just outside it, and losing the
    # complementary whole-image signal across 3 of 7 fusion inputs hurt
    # more than the crop helped. Reverted.
    print(f"Loading multi-source portrait dataset ({MAX_SAMPLES} samples)...")
    train_samples, val_samples = load_portrait_dataset(
        max_samples=MAX_SAMPLES, train_split=TRAIN_SPLIT, face_align=False,
    )

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples  : {len(val_samples)}")

    train_loader = DataLoader(
        PortraitDataset(train_samples, transform=TRAIN_TRANSFORM),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        PortraitDataset(val_samples, transform=VAL_TRANSFORM),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # -------- Model --------
    model = DINOv2AuthModel().to(device)
    criterion = nn.BCELoss()

    # Differential learning rates: backbone vs classifier head
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR, "weight_decay": 1e-4},
        {"params": head_params, "lr": HEAD_LR, "weight_decay": 1e-4},
    ])

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    # -------- Training with early stopping --------
    print(f"\nStarting DINOv2 auth model training (label_smoothing={LABEL_SMOOTHING})...\n")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs = imgs.to(device)
            labels = labels.unsqueeze(1).to(device)
            smoothed = labels * (1 - LABEL_SMOOTHING) + (1 - labels) * LABEL_SMOOTHING

            preds = model(imgs)
            loss = criterion(preds, smoothed)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # -------- Validation --------
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels_t = labels.unsqueeze(1).to(device)

                preds = model(imgs)
                loss = criterion(preds, labels_t)
                val_loss += loss.item()

                pred_labels = (preds > 0.5).float()
                correct += (pred_labels == labels_t).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"| Train Loss: {avg_loss:.4f} "
            f"| Val Loss: {avg_val_loss:.4f} "
            f"| Val Acc: {val_acc:.4f}"
        )

        # -------- Early stopping --------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> Best model saved (val_loss={avg_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break

    print("\nDINOv2 auth model training complete.")
    print(f"Best model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
