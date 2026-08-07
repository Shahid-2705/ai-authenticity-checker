"""
Dataset loader for AI-generated portrait detection.

Streams balanced datasets from HuggingFace:
  - GAN-generated faces (StyleGAN, etc.)
  - Diffusion-generated images (Stable Diffusion, Midjourney-style, DALL-E) —
    both via PORTRAIT_SOURCES and the dedicated DeepFakeFace source
  - Real portrait photos from diverse sources

All images are optionally face-aligned and cropped before being returned.

Usage:
    from training.dataset_portraits import load_portrait_dataset

    train_data, val_data = load_portrait_dataset(
        max_samples=40000,
        train_split=0.85,
    )
"""

import sys
import os
import random
import zipfile

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(ROOT_DIR, ".hf_cache", "datasets"))

from io import BytesIO

import torch
from PIL import Image, ImageFilter
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
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ──────────────────────────────────────────────
# HuggingFace Dataset Sources
# ──────────────────────────────────────────────
# Each entry: dataset_id, split, image_col, label_col, fake_value, real_value
# Ordered by quality/diversity. Loader tries each source and skips failures.
#
# NOTE: JamieWithofs/Deepfake-and-real-images is deliberately excluded here -
# it returned 0 fake samples out of 20,000+ scanned (streaming shuffle never
# surfaced any), indicating a broken/unreliable label distribution in this
# source. upstream/main re-added it independently; dropped again on merge.

PORTRAIT_SOURCES = [
    # --- Primary: large, well-labeled datasets ---
    {
        "id": "Hemg/AI-Generated-vs-Real-Images-Datasets",
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        "fake_value": 1,  # 1=AI, 0=Real
        "real_value": 0,
        "description": "AI-generated vs real (diverse AI methods)",
    },
    {
        "id": "Hemg/deepfake-and-real-images",
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        "fake_value": 0,
        "real_value": 1,
        "description": "190K deepfake and real images",
    },
    # --- Secondary: modern AI generators, diffusion models ---
    {
        "id": "poloclub/diffusiondb",
        "name": "random_1k",
        "split": "train",
        "image_col": "image",
        "label_col": None,   # All images are AI-generated (no label col)
        "fake_value": None,
        "real_value": None,
        "all_fake": True,     # Every image is AI-generated
        "description": "Stable Diffusion generated images (DiffusionDB)",
    },
    {
        "id": "AIML-TUDA/i_RAVEN",
        "split": "test",
        "image_col": "image",
        "label_col": "label",
        "fake_value": 1,
        "real_value": 0,
        "description": "AI vs real visual patterns",
    },
    {
        "id": "clips/deepfake_detection",
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        "fake_value": 1,
        "real_value": 0,
        "description": "Deepfake detection benchmark",
    },
    # --- Modern generators: diffusion, DALL-E, Midjourney ---
    {
        "id": "Rajarshi-Roy-research/Defactify_Image_Dataset",
        "split": "train",
        "image_col": "Image",
        "label_col": "Label_A",
        "fake_value": 1,   # 1=AI-generated, 0=Real
        "real_value": 0,
        "description": "96K images: SD2.1, SDXL, SD3, DALL-E 3, Midjourney v6",
    },
    {
        "id": "ComplexDataLab/OpenFake",
        "name": "core",
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        "fake_value": "fake",   # string labels
        "real_value": "real",
        "description": "2.3M real vs AI-generated (multi-generator, multi-source)",
    },
    {
        "id": "DataScienceProject/Art_Images_Ai_And_Real_",
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        "fake_value": 0,   # 0=fake, 1=real
        "real_value": 1,
        "description": "AI art vs real images (balanced)",
    },
    {
        "id": "ehristoforu/midjourney-images",
        "split": "train",
        "image_col": "image",
        "label_col": None,
        "fake_value": None,
        "real_value": None,
        "all_fake": True,
        "description": "Midjourney V5/V6 generated images",
    },
    # --- Tertiary: additional real-face sources for balance ---
    {
        "id": "nielsr/CelebA-faces",
        "split": "train",
        "image_col": "image",
        "label_col": None,
        "fake_value": None,
        "real_value": None,
        "all_real": True,     # Every image is real
        "description": "CelebA real celebrity faces",
    },
    {
        "id": "logasja/UTKFace",
        "split": "train",
        "image_col": "image",
        "label_col": None,
        "fake_value": None,
        "real_value": None,
        "all_real": True,
        "description": "UTKFace real faces (diverse demographics)",
    },
]

# -------- Diffusion-generated source (not HF-streamable, handled separately) --------
# training/eval_image_benchmark.py found the ensemble at 30.8% accuracy against
# this dataset (13.3%/6.7%/20.0% on inpainting/insight/text2img fakes) despite
# 83.3% on real photos — every PORTRAIT_SOURCES entry above is GAN/face-swap
# style, so nothing here has ever taught the models what diffusion-model
# output looks like. Added as a third contributor to load_portrait_dataset().
DEEPFAKEFACE_REPO = "OpenRL/DeepFakeFace"
DEEPFAKEFACE_FAKE_ZIPS = ["inpainting.zip", "insight.zip", "text2img.zip"]
DEEPFAKEFACE_REAL_ZIP = "wiki.zip"
DEEPFAKEFACE_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# Must match eval_image_benchmark.py's --seed/--per-category defaults exactly —
# this is how training samples are guaranteed to exclude the benchmark's
# held-out images (same shuffle, we just start reading further down the list).
DEEPFAKEFACE_BENCHMARK_SEED = 42
DEEPFAKEFACE_BENCHMARK_HOLDOUT = 30

# -------- Face-morph source (fake-only, no bona-fide images bundled) --------
# 2026-08-06 manual testing (test_examples/manual_checks/) found the whole
# ensemble missing 7/9 "morphy_me"/"dream.film"-style face-morph/face-average
# AI images — a category none of the sources above represent: they're not
# diffusion synthesis, not GAN synthesis, not a graft/face-swap onto one real
# photo, but a smooth blend of features from multiple real faces. This is the
# exact academic category "face morphing attack detection" research covers.
# DiM-FRLL-Morphs: diffusion-based morphs (DiM/Fast-DiM/Greedy-DiM algorithms)
# built from the FRLL research-consented face dataset, CC-BY-SA-4.0, ungated.
FACEMORPH_REPO = "zblasingame/DiM-FRLL-Morphs"
FACEMORPH_SPLIT = "train"
FACEMORPH_IMAGE_COL = "image"
# ~1.1s/file measured (see collect_facemorphs) - keeps runtime bounded to a
# few minutes and stays well clear of the ~1,500-file quota wall.
FACEMORPH_MAX_SAMPLES = 300


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
    except Exception:  # Broad catch: face detection can fail in many ways (cv2, PIL, etc.)
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

    Handles three source types:
      - Standard labeled (fake_value/real_value columns, string or int)
      - all_fake=True (every image is AI-generated, label=1)
      - all_real=True (every image is real, label=0)

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

    Returns list of (PIL.Image, label) where label is 0=real, 1=fake (normalized).
    """
    dataset_id = source["id"]
    print(f"  Loading {dataset_id}...")
    try:
        load_kwargs = {
            "path": dataset_id,
            "split": source["split"],
            "streaming": True,
        }
        if "name" in source:
            load_kwargs["name"] = source["name"]
        stream = load_dataset(**load_kwargs).shuffle(seed=42, buffer_size=10000)
    except Exception as e:  # Broad catch: HF dataset loading can fail in many ways
        print(f"  WARNING: Could not load {dataset_id}: {e}")
        return []

    scan_limit = (per_class + skip_per_class) * scan_multiplier

    is_all_fake = source.get("all_fake", False)
    is_all_real = source.get("all_real", False)
    label_col = source.get("label_col")

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
        total_seen += 1

        # Determine label
        if is_all_fake:
            normalized_label = 1
        elif is_all_real:
            normalized_label = 0
        elif label_col is not None:
            raw_label = sample[label_col]
            fake_val = source["fake_value"]
            real_val = source["real_value"]

            # Support both string labels ("real"/"fake") and int labels (0/1)
            if isinstance(fake_val, str):
                raw_str = str(raw_label).lower().strip()
                if raw_str == fake_val.lower().strip():
                    normalized_label = 1
                elif raw_str == real_val.lower().strip():
                    normalized_label = 0
                else:
                    continue
            else:
                raw_int = int(raw_label)
                if raw_int == fake_val:
                    normalized_label = 1
                elif raw_int == real_val:
                    normalized_label = 0
                else:
                    continue
        else:
            continue

        # Get image
        img_col = source.get("image_col", "image")
        try:
            img = sample[img_col].convert("RGB")
        except Exception:  # Broad catch: image decoding varies widely across sources
            continue

        # Minimum size filter — skip tiny images
        if img.width < 64 or img.height < 64:
            continue

        if normalized_label == 1:
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
                    if face_align:
                        img = _try_detect_face_crop(img)
                    if replace_idx is None:
                        fake_reservoir.append((img, 1))  # Normalized: 1=fake
                    else:
                        fake_reservoir[replace_idx] = (img, 1)
        else:
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


def _sample_deepfakeface_zip(filename, n, skip, face_align):
    """
    Sample n images from one OpenRL/DeepFakeFace zip member, via plain
    random sampling — not reservoir sampling, because zipfile.namelist()
    already gives the full member list up front (no streaming visibility
    problem to work around here, unlike the HF-streaming sources above).

    Always shuffles with DEEPFAKEFACE_BENCHMARK_SEED first and discards the
    first DEEPFAKEFACE_BENCHMARK_HOLDOUT entries — the exact slice
    eval_image_benchmark.py already consumed — so training data can never
    overlap with that held-out benchmark, by construction.
    """
    from huggingface_hub import hf_hub_download

    try:
        zip_path = hf_hub_download(
            repo_id=DEEPFAKEFACE_REPO, repo_type="dataset", filename=filename
        )
    except Exception as e:
        print(f"    WARNING: could not fetch {filename}: {e}")
        return []

    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(DEEPFAKEFACE_IMG_EXTS)]
        rng = random.Random(DEEPFAKEFACE_BENCHMARK_SEED)
        rng.shuffle(members)
        remaining = members[DEEPFAKEFACE_BENCHMARK_HOLDOUT:]
        if skip:
            remaining = remaining[skip:]
        chosen = remaining[:n]

        images = []
        for m in chosen:
            try:
                with zf.open(m) as f:
                    img = Image.open(BytesIO(f.read())).convert("RGB")
                if face_align:
                    img = _try_detect_face_crop(img)
                images.append(img)
            except Exception:
                continue
        return images


def collect_from_deepfakeface(per_class, face_align=True, skip_per_class=0):
    """
    Collect diffusion-generated fakes (Stable Diffusion Inpainting/text2img,
    InsightFace) and real photos (IMDB-WIKI) from OpenRL/DeepFakeFace.

    Fakes are split evenly across the 3 generation methods so no single
    diffusion technique dominates the sample.

    Returns:
        list of (PIL.Image, label) — label 0=real, 1=fake.
    """
    print(f"  Loading {DEEPFAKEFACE_REPO} (diffusion-generated, excludes "
          f"benchmark's held-out {DEEPFAKEFACE_BENCHMARK_HOLDOUT}/zip)...")

    per_fake_zip = max(1, per_class // len(DEEPFAKEFACE_FAKE_ZIPS))
    fake_skip = skip_per_class // len(DEEPFAKEFACE_FAKE_ZIPS) if skip_per_class else 0

    fake_images = []
    for fz in DEEPFAKEFACE_FAKE_ZIPS:
        fake_images.extend(_sample_deepfakeface_zip(fz, per_fake_zip, fake_skip, face_align))

    real_images = _sample_deepfakeface_zip(
        DEEPFAKEFACE_REAL_ZIP, per_class, skip_per_class, face_align
    )

    print(f"    Collected: {len(fake_images)} fake, {len(real_images)} real "
          f"from {DEEPFAKEFACE_REPO}")

    return [(img, 1) for img in fake_images] + [(img, 0) for img in real_images]


def collect_facemorphs(n_samples, face_align=True, skip=0):
    """
    Collect n_samples face-morph images from FACEMORPH_REPO. Every row in
    this dataset is a morph (fake) - there's no bona-fide/real class bundled
    in, so this only ever returns label=1 samples.

    Downloads ONLY the specific files needed, not the whole 3,000-file
    dataset: this repo's storage backend hits a hard download quota around
    ~1,500 files per session even when authenticated (measured: two
    separate full-dataset attempts both stalled at exactly the same ~50%
    mark, which is a quota wall, not transient rate-limiting - retries with
    backoff don't help). list_repo_files() is a single cheap API call, then
    we pick a random n_samples of them and fetch just those individually.

    n_samples is capped at FACEMORPH_MAX_SAMPLES regardless of what's
    requested: measured ~1.1s/file (same overhead whether streamed, bulk-
    downloaded, or fetched individually - it's inherent to this dataset's
    storage backend), so requesting anywhere near per_class values like
    Frequency CNN's ~2666 would take ~50 min and risk the quota wall again.
    This is meant as supplementary enrichment for one specific blind spot,
    not primary data volume, so a few hundred images is enough either way.
    """
    from huggingface_hub import HfApi, hf_hub_download

    n_samples = min(n_samples, FACEMORPH_MAX_SAMPLES)

    print(f"  Loading {FACEMORPH_REPO} (face-morph attack detection data)...")
    try:
        all_files = [
            f for f in HfApi().list_repo_files(FACEMORPH_REPO, repo_type="dataset")
            if f.lower().endswith(".png")
        ]
    except Exception as e:
        print(f"  WARNING: Could not list files for {FACEMORPH_REPO}: {e}")
        return []

    random.seed(42)
    random.shuffle(all_files)
    chosen = all_files[skip:skip + n_samples]

    images = []
    for fname in chosen:
        try:
            path = hf_hub_download(
                repo_id=FACEMORPH_REPO, repo_type="dataset", filename=fname,
            )
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        if face_align:
            img = _try_detect_face_crop(img)
        images.append(img)

    print(f"    Collected: {len(images)} face-morph fakes from {FACEMORPH_REPO}")
    return [(img, 1) for img in images]


def load_portrait_dataset(max_samples=40000, train_split=0.85, face_align=True,
                          skip_per_class=0, seed=42, sources=None):
    """
    Load balanced portrait dataset from multiple sources.

    Args:
        max_samples: Total target samples (split across sources).
                     Default increased to 40K for meaningful training.
        train_split: Fraction for training
        face_align: Apply face detection and cropping
        skip_per_class: Skip this many samples per class per source before
                        collecting (prevents overlap with other training sets)
        seed: Random seed for shuffling and splitting
        sources: Optional list of source dicts to use. Defaults to PORTRAIT_SOURCES.

    Returns:
        (train_samples, val_samples) -- each is list of (PIL.Image, label)
        label: 0=real, 1=fake (AI-generated)
    """
    if sources is None:
        sources = PORTRAIT_SOURCES

    # Filter to labeled sources (have both fake and real, or are marked all_fake/all_real)
    labeled_sources = [s for s in sources if
                       s.get("label_col") is not None or
                       s.get("all_fake") or
                       s.get("all_real")]

    # DeepFakeFace and DiM-FRLL-Morphs aren't counted in n_sources: DeepFakeFace
    # is handled separately (zip-based, not HF-streamable) and the face-morph
    # source is fake-only, so neither competes with the real-image budget -
    # they just enrich the fake pool's diversity before the balance step below
    # caps both classes to whichever is smaller. Counting them here would
    # needlessly shrink per_class for the sources that actually supply real
    # images.
    n_sources = len(labeled_sources) + 1  # +1 for DeepFakeFace (diffusion)
    per_source = max_samples // max(n_sources, 1)
    per_class = per_source // 2

    print(f"Loading portrait dataset: {max_samples} target samples from "
          f"{len(labeled_sources)} real+fake sources + 1 fake-only source "
          f"({per_class} per class per source)"
          + (f", skipping {skip_per_class}/class/source" if skip_per_class else ""))

    all_samples = []
    for source in labeled_sources:
        samples = collect_from_source(
            source, per_class, face_align=face_align,
            skip_per_class=skip_per_class,
        )
        all_samples.extend(samples)

    all_samples.extend(
        collect_from_deepfakeface(
            per_class, face_align=face_align, skip_per_class=skip_per_class,
        )
    )

    # Fake-only supplementary source (no real/bona-fide images bundled in,
    # so it only ever adds to the fake pool - the balance step below caps
    # both classes to whichever is smaller either way).
    all_samples.extend(
        collect_facemorphs(per_class, face_align=face_align, skip=skip_per_class)
    )

    # Balance classes
    fake = [s for s in all_samples if s[1] == 1]
    real = [s for s in all_samples if s[1] == 0]
    min_count = min(len(fake), len(real))
    if min_count == 0:
        raise RuntimeError(
            f"No balanced samples collected. Fake: {len(fake)}, Real: {len(real)}. "
            "Check dataset availability and network connection."
        )

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


# ──────────────────────────────────────────────
# Adversarial Augmentation
# ──────────────────────────────────────────────


def _jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    """Simulate JPEG compression at a given quality level."""
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class AdversarialAugmentation:
    """
    Augmentations that simulate real-world degradations adversarial to
    deepfake detectors: JPEG compression, resize degradation, blur,
    and social media compression pipelines.

    Applied with probability p (default 0.4) during training.
    """

    def __init__(self, p: float = 0.4):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        aug_type = random.choice(["jpeg", "resize", "blur", "social_media"])

        if aug_type == "jpeg":
            quality = random.randint(30, 85)
            return _jpeg_compress(img, quality)

        elif aug_type == "resize":
            w, h = img.size
            scale = random.uniform(0.3, 0.7)
            small = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            return small.resize((w, h), Image.BILINEAR)

        elif aug_type == "blur":
            radius = random.uniform(0.5, 2.5)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))

        else:  # social_media: resize + JPEG combined
            w, h = img.size
            scale = random.uniform(0.5, 0.8)
            small = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            resized = small.resize((w, h), Image.BILINEAR)
            quality = random.randint(50, 80)
            return _jpeg_compress(resized, quality)


# ──────────────────────────────────────────────
# Standard transforms
# ──────────────────────────────────────────────

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
    train_data, val_data = load_portrait_dataset(max_samples=200, face_align=False)
    print(f"\nQuick test: {len(train_data)} train, {len(val_data)} val")
    img, label = train_data[0]
    print(f"Sample: {img.size}, label={label}")
