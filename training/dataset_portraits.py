"""
Dataset loader for AI-generated portrait detection.

Streams balanced datasets from HuggingFace:
  - Diffusion-generated faces
  - GAN portraits (StyleGAN, etc.)
  - Real portrait photos

All images are face-aligned and cropped before being returned.

Usage:
    from training.dataset_portraits import load_portrait_dataset

    train_data, val_data = load_portrait_dataset(
        max_samples=4000,
        train_split=0.85,
    )
"""

import sys
import os
import io
import random

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(ROOT_DIR, ".hf_cache", "datasets"))

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset


def _random_jpeg_recompress(img, quality_range=(50, 95), p=0.5):
    """Randomly re-encode through JPEG at a random quality level.

    Unlike blur/rotation/crop, this doesn't destroy the high-frequency
    generative artifacts a frequency-domain classifier relies on — it
    teaches the model to be robust to real-world compression variance
    instead of overfitting to one source's specific compression profile.
    """
    if random.random() > p:
        return img
    quality = random.randint(*quality_range)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# -------- HuggingFace dataset sources --------
# Each entry: (dataset_id, split, image_col, label_col, fake_label_value)
PORTRAIT_SOURCES = [
    # NOTE: JamieWithofs/Deepfake-and-real-images was removed — it returned
    # 0 fake samples out of 20,000+ scanned (streaming shuffle never surfaced
    # any), indicating a broken/unreliable label distribution in this source.
    # AI-Generated vs Real (diverse AI methods)
    {
        "id": "Hemg/AI-Generated-vs-Real-Images-Datasets",
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        "fake_value": 1,  # 1=AI, 0=Real
        "real_value": 0,
    },
    # 190k deepfake and real images
    {
        "id": "Hemg/deepfake-and-real-images",
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        "fake_value": 0,  # 0=Fake, 1=Real
        "real_value": 1,
    },
]


def _try_detect_face_crop(pil_img, expand_ratio=0.3):
    """
    Attempt face detection and cropping. Returns original if no face found.
    Uses OpenCV DNN face detector.
    """
    try:
        from utils.gradcam import detect_and_align_face
        face_crop, bbox = detect_and_align_face(pil_img, expand_ratio=expand_ratio)
        if face_crop is not None:
            return face_crop
    except Exception:
        pass
    return pil_img


def collect_from_source(source, per_class, face_align=True, skip_per_class=0,
                         scan_multiplier=20):
    """
    Collect balanced samples from a single HuggingFace dataset source.

    Uses reservoir sampling (Algorithm R) per class over a wide scan, so the
    final samples are a uniform random draw across the whole scanned range
    rather than just whatever appeared first. Per-class quota collection
    that stops the instant it's met only ever samples a narrow early slice
    of the stream, which risks the model overfitting to that slice's quirks
    instead of learning something that generalizes (confirmed with a real
    train/held-out accuracy gap for CorefakeNet before this fix - see the
    same fix already applied to train_dinov2.py, train_efficientnet_auth.py,
    and train_face_deepfake_hf.py).

    Args:
        source: dict with dataset config
        per_class: target samples per class
        face_align: whether to apply face detection and cropping
        skip_per_class: skip this many samples per class before collecting
                        (used to avoid overlap with component model training
                        data - e.g. train_fusion.py's held-out set). Reservoir
                        sampling is applied to what's scanned after the skip.
        scan_multiplier: scan up to (per_class + skip_per_class) * this many
                        samples total

    Returns:
        list of (PIL.Image, label) where label is 0=real, 1=fake (normalized)
    """
    print(f"  Loading {source['id']}...")
    try:
        stream = load_dataset(
            source["id"],
            split=source["split"],
            streaming=True,
        ).shuffle(seed=42, buffer_size=10000)
    except Exception as e:
        print(f"  WARNING: Could not load {source['id']}: {e}")
        return []

    scan_limit = (per_class + skip_per_class) * scan_multiplier

    fake_reservoir = []
    real_reservoir = []
    fake_skipped = 0
    real_skipped = 0
    fake_seen = 0  # count of post-skip fake samples seen (for reservoir)
    real_seen = 0
    total_seen = 0
    last_printed = 0

    random.seed(42)

    for sample in stream:
        raw_label = int(sample[source["label_col"]])
        total_seen += 1

        if raw_label == source["fake_value"]:
            if fake_skipped < skip_per_class:
                fake_skipped += 1
            else:
                fake_seen += 1
                use_slot = len(fake_reservoir) < per_class
                replace_idx = None
                if not use_slot:
                    j = random.randint(0, fake_seen - 1)
                    if j < per_class:
                        use_slot = True
                        replace_idx = j
                if use_slot:
                    img = sample[source["image_col"]].convert("RGB")
                    if face_align:
                        img = _try_detect_face_crop(img)
                    if replace_idx is None:
                        fake_reservoir.append((img, 1))  # Normalized: 1=fake
                    else:
                        fake_reservoir[replace_idx] = (img, 1)
        elif raw_label == source["real_value"]:
            if real_skipped < skip_per_class:
                real_skipped += 1
            else:
                real_seen += 1
                use_slot = len(real_reservoir) < per_class
                replace_idx = None
                if not use_slot:
                    j = random.randint(0, real_seen - 1)
                    if j < per_class:
                        use_slot = True
                        replace_idx = j
                if use_slot:
                    img = sample[source["image_col"]].convert("RGB")
                    if face_align:
                        img = _try_detect_face_crop(img)
                    if replace_idx is None:
                        real_reservoir.append((img, 0))  # Normalized: 0=real
                    else:
                        real_reservoir[replace_idx] = (img, 0)

        if total_seen - last_printed >= 5000:
            last_printed = total_seen
            print(f"    scanned {total_seen}/{scan_limit} "
                  f"(Fake: {len(fake_reservoir)}, Real: {len(real_reservoir)})")

        if total_seen >= scan_limit:
            break

    print(f"    Collected: {len(fake_reservoir)} fake, {len(real_reservoir)} real "
          f"(scanned {total_seen}, skipped {fake_skipped}+{real_skipped})")
    return fake_reservoir + real_reservoir


def load_portrait_dataset(max_samples=4000, train_split=0.85, face_align=True,
                          skip_per_class=0, seed=42):
    """
    Load balanced portrait dataset from multiple sources.

    Args:
        max_samples: Total target samples (split across sources)
        train_split: Fraction for training
        face_align: Apply face detection and cropping
        skip_per_class: Skip this many samples per class per source before
                        collecting (prevents overlap with other training sets)
        seed: Random seed for shuffling and splitting

    Returns:
        (train_samples, val_samples) — each is list of (PIL.Image, label)
        label: 0=real, 1=fake (AI-generated)
    """
    per_source = max_samples // len(PORTRAIT_SOURCES)
    per_class = per_source // 2

    print(f"Loading portrait dataset: {max_samples} target samples from "
          f"{len(PORTRAIT_SOURCES)} sources ({per_class} per class per source)"
          + (f", skipping {skip_per_class}/class/source" if skip_per_class else ""))

    all_samples = []
    for source in PORTRAIT_SOURCES:
        samples = collect_from_source(
            source, per_class, face_align=face_align,
            skip_per_class=skip_per_class,
        )
        all_samples.extend(samples)

    # Balance classes
    fake = [s for s in all_samples if s[1] == 1]
    real = [s for s in all_samples if s[1] == 0]
    min_count = min(len(fake), len(real))
    if min_count == 0:
        raise RuntimeError("No samples collected from any source")

    random.seed(seed)
    random.shuffle(fake)
    random.shuffle(real)
    balanced = fake[:min_count] + real[:min_count]
    random.shuffle(balanced)

    print(f"Balanced dataset: {len(balanced)} samples "
          f"({min_count} fake + {min_count} real)")

    # Split
    split_idx = int(len(balanced) * train_split)
    train_samples = balanced[:split_idx]
    val_samples = balanced[split_idx:]

    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")
    return train_samples, val_samples


class PortraitDataset(Dataset):
    """
    PyTorch Dataset wrapper for portrait samples.

    Args:
        data: list of (PIL.Image, label)
        transform: torchvision transform to apply
        fft_mode: if True, return FFT magnitude instead of RGB tensor
        fft_augment: if True (and fft_mode=True), apply random spatial
                     transforms before FFT to create training diversity
    """

    def __init__(self, data, transform=None, fft_mode=False, fft_augment=False):
        self.data = data
        self.transform = transform
        self.fft_mode = fft_mode
        self.fft_augment = fft_augment and fft_mode

        if self.fft_augment:
            # No blur/resize-crop/rotation here: those are low-pass and
            # scale/geometry distortions that corrupt the high-frequency
            # spectral signature the FrequencyCNN is trained to classify.
            # Horizontal flip preserves the FFT magnitude spectrum's
            # structure (mirrors symmetrically) so it's safe to keep.
            self.pre_fft_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]

        if self.fft_mode:
            from core_models.frequency_cnn import fft_to_tensor
            if self.fft_augment:
                img = self.pre_fft_transform(img)
                img = _random_jpeg_recompress(img)
            tensor = fft_to_tensor(img, size=256)
        elif self.transform:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)

        return tensor, torch.tensor(label, dtype=torch.float32)


# -------- Standard transforms --------
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    ], p=0.3),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


if __name__ == "__main__":
    train_data, val_data = load_portrait_dataset(max_samples=100, face_align=False)
    print(f"\nQuick test: {len(train_data)} train, {len(val_data)} val")
    img, label = train_data[0]
    print(f"Sample: {img.size}, label={label}")
