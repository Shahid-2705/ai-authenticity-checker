"""
Independent held-out benchmark: run the PRODUCTION image ensemble
(core.pipeline.analyze_image, unchanged) against OpenRL/DeepFakeFace —
an ungated, freely-licensed (openrail) diffusion-generated deepfake dataset
we have never trained on.

Why this dataset specifically: our training sources (Hemg/AI-Generated-vs-
Real-Images-Datasets, Hemg/deepfake-and-real-images) skew GAN/face-swap.
Most real-world "is this AI?" checks today are against DIFFUSION output
(Stable Diffusion, Midjourney, DALL-E-style), not GANs. DeepFakeFace gives
three distinct diffusion generation methods plus a real-photo baseline:

    inpainting/  — Stable Diffusion Inpainting fakes    (label=1)
    insight/     — InsightFace toolbox fakes             (label=1)
    text2img/    — Stable Diffusion v1.5 text2img fakes  (label=1)
    wiki/        — real photos, IMDB-WIKI                (label=0)

This is evaluation only — no training, no changes to any model. It exists
to get a trustworthy accuracy number on data we've never seen, comparable
in spirit to how the research papers report FF++/Celeb-DF numbers, instead
of only ever checking ourselves against the same messy HF sources we train
on.

Usage:
    .venv/Scripts/python.exe training/eval_image_benchmark.py --per-category 40
"""

import sys
import os
import argparse
import random
import zipfile

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))

from PIL import Image
from tqdm import tqdm

REPO_ID = "OpenRL/DeepFakeFace"
CATEGORIES = {
    "inpainting.zip": 1,
    "insight.zip": 1,
    "text2img.zip": 1,
    "wiki.zip": 0,
}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def download_and_sample(filename, label, per_category, seed, cache_dir):
    """Downloads (or reuses cached) the zip, then extracts a random sample
    of `per_category` images without extracting the whole archive."""
    from huggingface_hub import hf_hub_download

    print(f"Fetching {filename} (cached after first run)...")
    zip_path = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=filename,
    )

    extract_dir = os.path.join(cache_dir, filename.replace(".zip", ""))
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(IMG_EXTS)]
        rng = random.Random(seed)
        rng.shuffle(members)
        chosen = members[:per_category]

        paths = []
        for m in chosen:
            out_path = os.path.join(extract_dir, os.path.basename(m))
            if not os.path.exists(out_path):
                with zf.open(m) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
            paths.append((out_path, label))

    print(f"  sampled {len(paths)} images from {filename}")
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=40,
                         help="images sampled per category (4 categories total)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from core.pipeline import analyze_image

    cache_dir = os.path.join(ROOT_DIR, ".hf_cache", "deepfakeface_samples")
    os.makedirs(cache_dir, exist_ok=True)

    all_samples = []
    for filename, label in CATEGORIES.items():
        all_samples.extend(
            download_and_sample(filename, label, args.per_category, args.seed, cache_dir)
        )

    print(f"\nTotal samples: {len(all_samples)}")
    print("Running production ensemble (core.pipeline.analyze_image, mode='ensemble')...\n")

    results = []  # (category, true_label, risk_score, predicted_label)
    for path, true_label in tqdm(all_samples, desc="Scoring"):
        category = os.path.basename(os.path.dirname(path))
        try:
            img = Image.open(path).convert("RGB")
            out = analyze_image(img, mode="ensemble")
            risk = out["risk_score"]
        except Exception as e:
            print(f"  ERROR on {path}: {e}")
            continue
        pred_label = 1 if risk >= 0.60 else 0  # matches Verdict.from_risk_score cutoff
        results.append((category, true_label, risk, pred_label))

    if not results:
        print("No results collected — aborting.")
        return

    # Overall accuracy
    correct = sum(1 for _, t, _, p in results if t == p)
    total = len(results)
    print(f"\n--- OVERALL: {correct}/{total} = {correct/total:.4f} accuracy ---")

    tp = sum(1 for _, t, _, p in results if t == 1 and p == 1)
    fn = sum(1 for _, t, _, p in results if t == 1 and p == 0)
    tn = sum(1 for _, t, _, p in results if t == 0 and p == 0)
    fp = sum(1 for _, t, _, p in results if t == 0 and p == 1)
    print(f"Confusion: TP={tp} FN={fn} (fake missed) | TN={tn} FP={fp} (real flagged as fake)")

    # Per-category breakdown (which generation method fools us most)
    print("\n--- PER-CATEGORY ---")
    cats = sorted(set(c for c, _, _, _ in results))
    for cat in cats:
        cat_results = [(t, p, r) for c, t, r, p in results if c == cat]
        cat_correct = sum(1 for t, p, _ in cat_results if t == p)
        avg_risk = sum(r for _, _, r in cat_results) / len(cat_results)
        print(f"  {cat:15s}: {cat_correct}/{len(cat_results)} = "
              f"{cat_correct/len(cat_results):.4f} accuracy | avg risk_score={avg_risk:.4f}")


if __name__ == "__main__":
    main()
