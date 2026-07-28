"""
Train the Video Temporal LSTM (core_models/video_lstm.py).

This replaces the hand-tuned variance/jump heuristic in
pipeline/video_analyzer.py's TemporalAnalyzer with a model trained on real
labeled video sequences to recognize genuine temporal-inconsistency
patterns — the same idea used by published video-deepfake detectors
(EVM+ResNext+LSTM / EfficientNet+TCN), which score noticeably higher than
frame-only classification on FF++/Celeb-DF precisely because of the learned
temporal component.

REQUIRES A GPU (Colab). Also requires all 7 fusion component models already
trained and present under models/ (upload your trained .pth files first, same
as train_fusion.py).

--------------------------------------------------------------------------
DATASET ACCESS (read this first)

Every real, temporally-ordered, labeled video deepfake dataset worth using
(FaceForensics++, DFDC, Celeb-DF, WildDeepfake, DeepSpeak) is GATED — you
must request access and wait for approval, because these involve real
people's faces/likeness, unlike the GAN/diffusion image datasets we've used
so far which are freely public. There is no ungated shortcut here.

Recommended: WildDeepfake (https://huggingface.co/datasets/xingjunm/WildDeepfake)
  - 7,314 face SEQUENCES from 707 real-world deepfake videos, already
    face-cropped and split into "real train/", "real test/", "fake train/",
    "fake test/" — each a set of numbered .tar.gz archives containing one
    folder per sequence, frames named by frame number.
  - Because frames are pre-cropped to faces, this script does NOT need
    video decoding at all (sidesteps the torchcodec/OpenCV codec issues
    we've hit before) — just image loading.
  - To get access: open the dataset page above, accept the agreement form,
    then set HF_TOKEN as an environment variable / Colab secret. Approval
    is manual and can take a few days.

If you'd rather not wait: DFDC-preview or Celeb-DF-v2 work too as long as
you point WILDDEEPFAKE_REPO / _load_sequences() at their actual layout —
the windowing/training logic below is dataset-agnostic, only
_iter_sequences() is dataset-specific.
--------------------------------------------------------------------------

Usage (Colab):
    !pip install -q huggingface_hub
    import os
    os.environ["HF_TOKEN"] = "hf_..."          # token with WildDeepfake access
    !python training/train_video_lstm.py
"""

import sys
import os
import glob
import random
import tarfile

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from core_models.video_lstm import VideoTemporalLSTM
from core_models.efficientnet_texture import EfficientNetTexture
from core_models.frequency_cnn import FrequencyCNN, fft_to_tensor
from core_models.dinov2_auth_model import DINOv2AuthModel
from core_models.efficientnet_auth_model import EfficientNetAuthModel
from core_models.face_deepfake_model import FaceDeepfakeModel
from core_models.fusion_mlp import FusionMLP

# -------- CONFIG --------
WINDOW_SIZE = 10
WINDOW_STRIDE = 5          # sliding-window stride when chopping a sequence into training windows
MAX_FRAMES_PER_SEQUENCE = 40  # cap compute per sequence
LSTM_HIDDEN = 32
LSTM_EPOCHS = 30
LSTM_LR = 1e-3
LSTM_BATCH = 32
MODELS_DIR = "models"
MODEL_PATH = "models/video_lstm.pth"
WILDDEEPFAKE_REPO = "xingjunm/WildDeepfake"
DATA_CACHE_DIR = os.path.join(ROOT_DIR, ".hf_cache", "wilddeepfake")
# -------------------------

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ================================================================
#  Component model loading (mirrors train_fusion.py's pattern)
# ================================================================

def _load_vit(device):
    try:
        from transformers import ViTForImageClassification, ViTImageProcessor
        model_id = "prithivMLmods/Deep-Fake-Detector-v2-Model"
        model = ViTForImageClassification.from_pretrained(model_id).to(device)
        processor = ViTImageProcessor.from_pretrained(model_id)
        model.eval()
        return model, processor
    except Exception as e:
        print(f"WARNING: Could not load ViT: {e}")
        return None, None


def _load_component(device, model_class, filename, name):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — upload your trained {name} model first "
            f"(same models/ folder train_fusion.py expects)."
        )
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model.eval()
    return model


def load_all_components(device):
    """Load every model the production Fusion MLP consumes, plus the fusion
    head itself, so we can reproduce the exact per-frame `raw_risk` the LSTM
    will see at inference time."""
    print("Loading component models...")
    vit_model, vit_processor = _load_vit(device)
    return {
        "vit_model": vit_model,
        "vit_processor": vit_processor,
        "texture": _load_component(device, EfficientNetTexture, "efficient.pth", "texture"),
        "frequency": _load_component(device, FrequencyCNN, "frequency.pth", "frequency"),
        "dino": _load_component(device, DINOv2AuthModel, "dinov2_auth_model.pth", "dino"),
        "efficientnet_auth": _load_component(
            device, EfficientNetAuthModel, "efficientnet_auth_model.pth", "efficientnet_auth"
        ),
        "face": _load_component(device, FaceDeepfakeModel, "image_face_model.pth", "face"),
        "fusion": _load_component(device, lambda: FusionMLP(n_inputs=7), "fusion_mlp.pth", "fusion"),
    }


def score_frame(pil_img, comp, device):
    """
    Score one frame with the full production ensemble and return exactly
    the fields VideoTemporalLSTM.predict_window() consumes:
    raw_risk, vit_prob, face_prob, frequency_prob.
    """
    rgb_img = pil_img.convert("RGB")
    scores = {"vit": 0.0, "texture": 0.0, "forensic": 0.0, "frequency": 0.0,
              "dino": 0.0, "efficientnet_auth": 0.0, "face": 0.0}

    if comp["vit_model"] is not None:
        try:
            inputs = comp["vit_processor"](images=rgb_img, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = comp["vit_model"](**inputs).logits
                probs = torch.softmax(logits, dim=1)
                fake_idx = [k for k, v in comp["vit_model"].config.id2label.items()
                            if "fake" in v.lower()]
                scores["vit"] = probs[0][fake_idx[0]].item() if fake_idx else probs[0][1].item()
        except Exception:
            pass

    try:
        tensor = VAL_TRANSFORM(rgb_img).unsqueeze(0).to(device)
    except Exception:
        tensor = None

    if tensor is not None:
        try:
            with torch.no_grad():
                scores["texture"] = comp["texture"](tensor).item()
        except Exception:
            pass
        try:
            with torch.no_grad():
                scores["dino"] = comp["dino"](tensor).item()
        except Exception:
            pass
        try:
            with torch.no_grad():
                scores["efficientnet_auth"] = comp["efficientnet_auth"](tensor).item()
        except Exception:
            pass
        try:
            with torch.no_grad():
                real_prob = comp["face"](tensor).item()
                scores["face"] = 1.0 - real_prob
        except Exception:
            pass

    try:
        from app import forensic_score
        scores["forensic"] = forensic_score(pil_img)
    except Exception:
        pass

    try:
        fft_tensor = fft_to_tensor(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            scores["frequency"] = comp["frequency"](fft_tensor).item()
    except Exception:
        pass

    with torch.no_grad():
        raw_risk = comp["fusion"].predict(**scores)

    return {
        "raw_risk": raw_risk,
        "vit_prob": scores["vit"],
        "face_prob": scores["face"],
        "frequency_prob": scores["frequency"],
    }


# ================================================================
#  WildDeepfake loading
# ================================================================

def _download_and_extract_wilddeepfake():
    """Downloads WildDeepfake via huggingface_hub (requires accepted gate +
    HF_TOKEN) and extracts every .tar.gz into DATA_CACHE_DIR."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {WILDDEEPFAKE_REPO} (requires accepted access request)...")
    local_dir = snapshot_download(
        repo_id=WILDDEEPFAKE_REPO, repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )

    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    for split_dir in ["real train", "real test", "fake train", "fake test"]:
        src = os.path.join(local_dir, split_dir)
        if not os.path.isdir(src):
            continue
        dest = os.path.join(DATA_CACHE_DIR, split_dir.replace(" ", "_"))
        os.makedirs(dest, exist_ok=True)
        for tar_path in glob.glob(os.path.join(src, "*.tar.gz")):
            print(f"  extracting {os.path.basename(tar_path)}...")
            with tarfile.open(tar_path) as tf:
                tf.extractall(dest)

    return DATA_CACHE_DIR


def iter_sequences(split):
    """
    Yields (video_id, sorted_frame_paths, label) for the given split
    ("train" or "test"). label: 1=fake, 0=real.

    Expects DATA_CACHE_DIR/{real,fake}_{split}/<sequence_folders>/<frames>.
    """
    if not os.path.isdir(DATA_CACHE_DIR):
        _download_and_extract_wilddeepfake()

    for label_name, label in [("real", 0), ("fake", 1)]:
        base = os.path.join(DATA_CACHE_DIR, f"{label_name}_{split}")
        if not os.path.isdir(base):
            print(f"WARNING: {base} not found — skipping {label_name}/{split}")
            continue
        for seq_dir in sorted(os.listdir(base)):
            seq_path = os.path.join(base, seq_dir)
            if not os.path.isdir(seq_path):
                continue
            frames = sorted(
                glob.glob(os.path.join(seq_path, "*")),
                key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p))) or 0),
            )
            if len(frames) < 3:
                continue
            if len(frames) > MAX_FRAMES_PER_SEQUENCE:
                # even subsample across the sequence, keep temporal order
                idx = sorted(random.sample(range(len(frames)), MAX_FRAMES_PER_SEQUENCE))
                frames = [frames[i] for i in idx]
            yield f"{label_name}_{split}_{seq_dir}", frames, label


def build_windows(video_id, score_seq, label):
    """Chop a per-frame score sequence into overlapping fixed-length windows
    (sliding window, stride WINDOW_STRIDE) — matches the causal window the
    LSTM sees at inference time (predict_window's last WINDOW_SIZE frames)."""
    windows = []
    n = len(score_seq)
    if n < 3:
        return windows
    starts = list(range(0, max(1, n - WINDOW_SIZE + 1), WINDOW_STRIDE))
    if not starts:
        starts = [0]
    for start in starts:
        window = score_seq[start:start + WINDOW_SIZE]
        feats = [[r["raw_risk"], r["vit_prob"], r["face_prob"], r["frequency_prob"]]
                 for r in window]
        while len(feats) < WINDOW_SIZE:
            feats.insert(0, [0.0, 0.0, 0.0, 0.0])
        windows.append((feats, label))
    return windows


def collect_split(split, comp, device):
    """Returns list of (video_id, feats_window, label) for every window in
    every sequence of the given split."""
    all_windows = []
    seqs = list(iter_sequences(split))
    print(f"\n{split}: {len(seqs)} sequences "
          f"({sum(1 for _, _, lbl in seqs if lbl == 1)} fake, "
          f"{sum(1 for _, _, lbl in seqs if lbl == 0)} real)")

    for video_id, frame_paths, label in tqdm(seqs, desc=f"Scoring {split} sequences"):
        score_seq = []
        for fp in frame_paths:
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                continue
            score_seq.append(score_frame(img, comp, device))
        for feats, lbl in build_windows(video_id, score_seq, label):
            all_windows.append((video_id, feats, lbl))

    return all_windows


# ================================================================
#  Training
# ================================================================

def train_lstm(train_windows, device):
    """Train/val split BY VIDEO ID (never by window) to avoid leaking
    windows from the same sequence across the split."""
    video_ids = sorted(set(vid for vid, _, _ in train_windows))
    random.seed(42)
    random.shuffle(video_ids)
    split_idx = int(len(video_ids) * 0.85)
    train_ids = set(video_ids[:split_idx])
    val_ids = set(video_ids[split_idx:])

    train_set = [(f, lbl) for vid, f, lbl in train_windows if vid in train_ids]
    val_set = [(f, lbl) for vid, f, lbl in train_windows if vid in val_ids]
    print(f"\nTrain windows: {len(train_set)} (from {len(train_ids)} videos) "
          f"| Val windows: {len(val_set)} (from {len(val_ids)} videos)")

    def to_tensors(windows):
        x = torch.tensor([w[0] for w in windows], dtype=torch.float32)
        y = torch.tensor([w[1] for w in windows], dtype=torch.float32).unsqueeze(1)
        return x, y

    train_x, train_y = to_tensors(train_set)
    val_x, val_y = to_tensors(val_set)

    model = VideoTemporalLSTM(n_features=4, hidden_size=LSTM_HIDDEN, window_size=WINDOW_SIZE).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR, weight_decay=1e-4)

    best_val_loss = float("inf")
    patience = 0

    for epoch in range(LSTM_EPOCHS):
        model.train()
        perm = torch.randperm(len(train_x))
        total_loss = 0.0
        for i in range(0, len(perm), LSTM_BATCH):
            idx = perm[i:i + LSTM_BATCH]
            bx, by = train_x[idx].to(device), train_y[idx].to(device)
            preds = model(bx)
            loss = criterion(preds, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_preds = model(val_x.to(device))
            val_loss = criterion(val_preds, val_y.to(device)).item()
            val_acc = ((val_preds.cpu() > 0.5).float() == val_y).float().mean().item()

        print(f"Epoch [{epoch+1}/{LSTM_EPOCHS}] "
              f"| Train loss: {total_loss/max(1, len(perm)//LSTM_BATCH):.4f} "
              f"| Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience += 1
            if patience >= 8:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nSaved best model to {MODEL_PATH}")
    return model


def independent_eval(test_windows, device):
    """
    Reload the SAVED (best val) checkpoint fresh and evaluate on the
    WildDeepfake "test" split — sequences the model has never seen in any
    form during training. This is the number to trust, not the training
    run's own val accuracy.
    """
    model = VideoTemporalLSTM(n_features=4, hidden_size=LSTM_HIDDEN, window_size=WINDOW_SIZE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    model.eval()

    x = torch.tensor([w[1] for w in test_windows], dtype=torch.float32).to(device)
    y = torch.tensor([w[2] for w in test_windows], dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        preds = model(x).cpu()

    acc = ((preds > 0.5).float() == y).float().mean().item()
    tp = ((preds > 0.5) & (y == 1)).sum().item()
    fp = ((preds > 0.5) & (y == 0)).sum().item()
    fn = ((preds <= 0.5) & (y == 1)).sum().item()
    tn = ((preds <= 0.5) & (y == 0)).sum().item()

    print("\n--- INDEPENDENT HELD-OUT EVAL (WildDeepfake test split, never trained on) ---")
    print(f"Accuracy: {acc:.4f}  ({len(test_windows)} windows)")
    print(f"Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
    print("This is the number to report/trust, not training-run val accuracy above.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU detected — this will be extremely slow. Use Colab with a GPU runtime.")

    comp = load_all_components(device)

    train_windows_raw = collect_split("train", comp, device)
    test_windows_raw = collect_split("test", comp, device)

    train_lstm(train_windows_raw, device)
    independent_eval(test_windows_raw, device)


if __name__ == "__main__":
    main()
