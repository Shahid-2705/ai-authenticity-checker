"""
Training script for the Audio Deepfake CNN model.

Combines samples from multiple HuggingFace real/fake speech datasets
(see HF_DATASETS) so the model learns real-vs-fake characteristics that
hold across recording conditions, rather than shortcut-learning a single
source's acoustic fingerprint.

Architecture: 2-layer CNN on mel-spectrograms with BatchNorm.

Usage:
    python training/train_audio_deepfake.py
"""

import sys
import os
import random
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ["HF_HOME"] = os.path.join(ROOT_DIR, ".hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(ROOT_DIR, ".hf_cache", "datasets")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import librosa

from core_models.audio_deepfake_model import AudioDeepfakeCNN

# ================= CONFIG =================
BATCH_SIZE = 32
EPOCHS = 20
LR = 3e-4
TRAIN_SPLIT = 0.85
MAX_SAMPLES = 50000         # 6x increase for robust audio detection
MODEL_PATH = "models/audio_deepfake_model.pth"
EARLY_STOPPING_PATIENCE = 5
LABEL_SMOOTHING = 0.05

# Audio preprocessing (matches zo9999)
SAMPLE_RATE = 22050
N_MELS = 91
MAX_TIME_STEPS = 150
N_FFT = 2048
HOP_LENGTH = 512
MAX_DURATION = 5.0          # seconds (matches zo9999 training)

# Dataset config — samples are COMBINED from all sources below, not just
# tried as a fallback chain. A held-out cross-dataset check found a model
# trained on Hemg/Deepfakeaudio alone reached 100% on its own validation
# split but only 52.7% (near-random, always guessing "real") when tested
# against garystafford/deepfake-audio-detection - it had learned Hemg's
# specific recording/compression fingerprint instead of genuine real-vs-
# fake characteristics. Combining diverse sources forces the model to
# learn features that hold across recording conditions.
# moibrahimovic/fake_or_real no longer exists on the Hub.
# ud-nlp/real-vs-fake-human-voice-deepfake-audio only has 70 total samples -
# too small to meaningfully contribute, excluded.
HF_DATASETS = [
    "Hemg/Deepfakeaudio",                     # 19,817 examples
    "garystafford/deepfake-audio-detection",  # 1,866 examples
]
# ==========================================


def audio_to_mel(waveform, sr=SAMPLE_RATE):
    """Convert waveform to mel-spectrogram (matching zo9999)."""
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Pad or truncate to fixed width. Pad with -80 dB (silence floor, since
    # power_to_db(ref=np.max) puts the loudest frame at 0 dB) rather than
    # the implicit 0 fill, which would represent peak loudness instead.
    if mel_db.shape[1] < MAX_TIME_STEPS:
        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, MAX_TIME_STEPS - mel_db.shape[1])),
            mode="constant",
            constant_values=-80.0,
        )
    else:
        mel_db = mel_db[:, :MAX_TIME_STEPS]

    return mel_db


class SpecAugment:
    """SpecAugment: frequency and time masking for mel-spectrogram augmentation."""

    def __init__(self, freq_mask_param=15, time_mask_param=20,
                 n_freq_masks=2, n_time_masks=2):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def __call__(self, mel):
        """Apply random frequency and time masks to a mel-spectrogram.

        Args:
            mel: numpy array of shape (n_mels, time_steps)
        Returns:
            augmented mel-spectrogram (same shape)
        """
        mel = mel.copy()
        n_mels, n_time = mel.shape

        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = random.randint(0, min(self.freq_mask_param, n_mels - 1))
            f0 = random.randint(0, n_mels - f)
            mel[f0:f0 + f, :] = mel.mean()

        # Time masking
        for _ in range(self.n_time_masks):
            t = random.randint(0, min(self.time_mask_param, n_time - 1))
            t0 = random.randint(0, n_time - t)
            mel[:, t0:t0 + t] = mel.mean()

        return mel


class AudioMelDataset(Dataset):
    """Dataset that holds pre-computed mel-spectrograms and labels."""

    def __init__(self, mels, labels, augment=False):
        self.mels = mels
        self.labels = labels
        self.augment = augment
        if augment:
            self.spec_augment = SpecAugment(
                freq_mask_param=12, time_mask_param=15,
                n_freq_masks=2, n_time_masks=2,
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel = self.mels[idx]
        if self.augment:
            mel = self.spec_augment(mel)
            # Random gain variation
            if random.random() < 0.3:
                gain = random.uniform(0.8, 1.2)
                mel = mel * gain
        # Normalize dB range [-80, 0] to [0, 1]. The model has no BatchNorm
        # layers; raw dB-scale input (magnitude ~40-80) caused dead ReLUs
        # and prevented any learning. Must match pipeline/audio_analyzer.py's
        # normalization used at inference time.
        mel_norm = (mel + 80.0) / 80.0
        mel = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel, label


def _parse_label(sample, label_feature=None):
    """Parse label from a HuggingFace dataset sample. Returns 0=fake, 1=real, or -1 if unknown.

    label_feature: the dataset's `features["label"]` object, if present. When
    it's a ClassLabel, its int-to-name mapping is used to resolve integer
    labels correctly instead of guessing a numeric convention (datasets are
    not consistent about whether 0 or 1 means "fake").
    """
    # ASVspoof style: 'key' column
    if "key" in sample:
        raw = sample["key"]
        return 1 if raw == "bonafide" else 0

    if "label" in sample:
        raw = sample["label"]
        if isinstance(raw, str):
            low = raw.lower().strip()
        elif label_feature is not None and hasattr(label_feature, "int2str"):
            low = label_feature.int2str(int(raw)).lower().strip()
        else:
            # No ClassLabel schema to resolve against - fall back to a
            # common convention (0=real, nonzero=fake). This is a guess.
            return 0 if int(raw) > 0 else 1

        if low in ("bonafide", "real", "genuine", "original", "authentic"):
            return 1
        elif low in ("spoof", "fake", "deepfake", "synthetic", "generated"):
            return 0
        return -1

    if "is_fake" in sample:
        return 0 if sample["is_fake"] else 1

    return -1


def _collect_from_source(ds_name, per_class_target, scan_limit):
    """Reservoir-sample up to per_class_target fake/real mel-spectrograms
    from a single HF dataset, scanning up to scan_limit total samples.

    Uses Algorithm R per class so the final samples are a uniform random
    draw across the whole scan range, not just whatever appeared first
    (this stream isn't shuffled by the source, and per-class quota
    collection that stops the instant it's met only ever samples a
    narrow early slice - see the same fix in train_dinov2.py et al).

    Returns: (reservoirs dict {0: [...], 1: [...]}, total_seen, errors)
    """
    from datasets import load_dataset, Audio

    print(f"\nAttempting to load: {ds_name}")
    try:
        ds = load_dataset(ds_name, split="train", streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE, decode=True))
        sample = next(iter(ds))
        print(f"  Columns: {list(sample.keys())}")
        # Re-create the iterator since we consumed one testing it
        ds = load_dataset(ds_name, split="train", streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE, decode=True))
        print(f"  Successfully connected to {ds_name}")
    except Exception as e:
        print(f"  Failed: {e}")
        return {0: [], 1: []}, 0, 0

    label_feature = ds.features.get("label")
    ds = ds.shuffle(seed=42, buffer_size=2000)

    reservoirs = {0: [], 1: []}  # 0=fake, 1=real
    seen_counts = {0: 0, 1: 0}
    total_seen = 0
    errors = 0
    last_printed = 0

    print(f"  Reservoir-sampling {per_class_target} per class from up to "
          f"{scan_limit} streamed samples...")

    for sample in ds:
        total_seen += 1

        label = _parse_label(sample, label_feature)
        if label == -1:
            continue

        seen_counts[label] += 1
        use_slot = len(reservoirs[label]) < per_class_target
        replace_idx = None
        if not use_slot:
            j = random.randint(0, seen_counts[label] - 1)
            if j < per_class_target:
                use_slot = True
                replace_idx = j

        if use_slot:
            try:
                audio_data = sample["audio"]
                waveform = audio_data["array"]
                sr = audio_data["sampling_rate"]

                if len(waveform) == 0:
                    continue

                max_samples = int(MAX_DURATION * sr)
                waveform = waveform[:max_samples].astype(np.float32)
                mel = audio_to_mel(waveform, sr)

                if replace_idx is None:
                    reservoirs[label].append((mel, label))
                else:
                    reservoirs[label][replace_idx] = (mel, label)

            except Exception:
                errors += 1
                if errors > 50:
                    print(f"    Too many errors ({errors}), stopping this source")
                    break
                continue

        if total_seen - last_printed >= 2000:
            last_printed = total_seen
            print(f"    scanned {total_seen}/{scan_limit} "
                  f"(Fake: {len(reservoirs[0])}, Real: {len(reservoirs[1])})")
        if total_seen >= scan_limit:
            break

    print(f"  Collected from {ds_name}: Fake={len(reservoirs[0])}, "
          f"Real={len(reservoirs[1])} (errors: {errors}) from {total_seen} streamed")
    return reservoirs, total_seen, errors


def load_dataset_hf():
    """Load audio samples, COMBINING all sources in HF_DATASETS.

    A model trained on a single source reached 100% on its own validation
    split but only 52.7% (near-random) on a different dataset - it learned
    that source's recording/compression fingerprint instead of genuine
    real-vs-fake characteristics. Combining diverse sources forces the
    model to learn features that hold across recording conditions.
    """
    import datasets.config
    datasets.config.AUDIO_DECODER_BACKEND = "soundfile"  # avoids torchcodec/FFmpeg issues on Windows

    random.seed(42)

    per_class = MAX_SAMPLES // 2
    per_class_per_source = per_class // len(HF_DATASETS)
    scan_limit_per_source = per_class_per_source * 10

    all_mels = {0: [], 1: []}
    for ds_name in HF_DATASETS:
        reservoirs, _, _ = _collect_from_source(
            ds_name, per_class_per_source, scan_limit_per_source
        )
        all_mels[0].extend(reservoirs[0])
        all_mels[1].extend(reservoirs[1])

    combined = all_mels[0] + all_mels[1]
    mels = [m for m, _ in combined]
    labels = [label_val for _, label_val in combined]
    print(f"\nTotal combined from {len(HF_DATASETS)} sources: {len(mels)} samples "
          f"(Fake: {len(all_mels[0])}, Real: {len(all_mels[1])})")

    if len(mels) < 20:
        return None, None

    return np.array(mels), np.array(labels)


def load_dataset_local():
    """Load audio from local directories: data/audio/real/ and data/audio/fake/."""
    real_dir = os.path.join(ROOT_DIR, "data", "audio", "real")
    fake_dir = os.path.join(ROOT_DIR, "data", "audio", "fake")

    if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
        return None, None

    mels = []
    labels = []
    audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

    for label, dirname in [(1, real_dir), (0, fake_dir)]:
        files = [
            os.path.join(dirname, f) for f in os.listdir(dirname)
            if os.path.splitext(f)[1].lower() in audio_exts
        ]
        print(f"Found {len(files)} {'real' if label == 1 else 'fake'} audio files")

        for fpath in tqdm(files, desc=f"Processing {'real' if label == 1 else 'fake'}"):
            try:
                waveform, sr = librosa.load(fpath, sr=SAMPLE_RATE, duration=MAX_DURATION)
                if len(waveform) < int(0.5 * SAMPLE_RATE):
                    continue
                mel = audio_to_mel(waveform, sr)
                mels.append(mel)
                labels.append(label)
            except Exception:
                continue

    if len(mels) == 0:
        return None, None

    return np.array(mels), np.array(labels)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Try local data first, then HuggingFace
    print("\nChecking for local audio data...")
    mels, labels = load_dataset_local()

    if mels is None or len(mels) < 20:
        print("Local data insufficient. Loading from HuggingFace...")
        mels, labels = load_dataset_hf()

    if mels is None or len(mels) < 20:
        print("\nError: Could not load sufficient training data.")
        print("Options:")
        print("  1. Place audio files in data/audio/real/ and data/audio/fake/")
        print("  2. Ensure HuggingFace datasets are accessible")
        return

    # Shuffle
    indices = list(range(len(labels)))
    random.seed(42)
    random.shuffle(indices)
    mels = mels[indices]
    labels = labels[indices]

    # Train/val split
    split = int(len(labels) * TRAIN_SPLIT)
    train_mels, val_mels = mels[:split], mels[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    print(f"\nTrain samples: {len(train_labels)}")
    print(f"Val samples  : {len(val_labels)}")
    print(f"Class balance: Fake={sum(labels == 0)}, Real={sum(labels == 1)}")

    train_loader = DataLoader(
        AudioMelDataset(train_mels, train_labels, augment=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        AudioMelDataset(val_mels, val_labels, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True,
    )

    # Model
    model = AudioDeepfakeCNN().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Training loop
    print("\nStarting Audio Deepfake CNN training...\n")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for mels_batch, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            mels_batch = mels_batch.to(device)
            labels_batch = labels_batch.to(device)

            # AudioDeepfakeCNN.forward() applies softmax internally (its
            # public contract, relied on elsewhere e.g. AudioAnalyzer).
            # CrossEntropyLoss also applies softmax internally, so feeding
            # it model(x) double-softmaxes and collapses gradients to
            # near-zero. Compose the submodules directly to get raw logits
            # for the loss instead.
            logits = model.classifier(model.features(mels_batch))
            loss = criterion(logits, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for mels_batch, labels_batch in val_loader:
                mels_batch = mels_batch.to(device)
                labels_batch = labels_batch.to(device)

                logits = model.classifier(model.features(mels_batch))
                loss = criterion(logits, labels_batch)
                val_loss += loss.item()

                pred_labels = logits.argmax(dim=1)
                correct += (pred_labels == labels_batch).sum().item()
                total += labels_batch.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"| Train Loss: {avg_loss:.4f} "
            f"| Val Loss: {avg_val_loss:.4f} "
            f"| Val Acc: {val_acc:.4f}"
        )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> Best model saved (val_loss={avg_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print("\nAudio Deepfake CNN training complete.")
    print(f"Best model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
