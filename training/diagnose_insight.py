"""
Diagnostic: why does the ensemble still miss InsightFace-generated fakes
(6.7% accuracy across every retrain round, the only category that hasn't
moved) while SD Inpainting/text2img improved substantially?

Pulls a handful of insight.zip samples (past the benchmark's held-out
slice, same as training does) and prints the FULL per-model score
breakdown for each one, so we can see whether specific models are
uniformly blind to it or whether it's more mixed.

Usage:
    .venv/Scripts/python.exe training/diagnose_insight.py --n 8
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
from huggingface_hub import hf_hub_download

REPO_ID = "OpenRL/DeepFakeFace"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
BENCHMARK_HOLDOUT = 30  # matches eval_image_benchmark.py's default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=99)  # different from benchmark's 42
    args = parser.parse_args()

    from core.pipeline import analyze_image

    zip_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename="insight.zip")
    extract_dir = os.path.join(ROOT_DIR, ".hf_cache", "diagnose_insight")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(IMG_EXTS)]
        # exclude the exact slice the benchmark used (seed=42, first 30)
        bench_rng = random.Random(42)
        bench_rng.shuffle(members)
        remaining = members[BENCHMARK_HOLDOUT:]

        rng = random.Random(args.seed)
        rng.shuffle(remaining)
        chosen = remaining[:args.n]

        paths = []
        for m in chosen:
            out_path = os.path.join(extract_dir, os.path.basename(m))
            if not os.path.exists(out_path):
                with zf.open(m) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
            paths.append(out_path)

    print(f"Sampled {len(paths)} insight.zip images -> {extract_dir}\n")

    for path in paths:
        img = Image.open(path).convert("RGB")
        out = analyze_image(img, mode="ensemble")
        scores = out.get("model_scores", {})
        print(f"{os.path.basename(path)}")
        print(f"  risk_score={out['risk_score']:.4f}  verdict={out['verdict']}")
        print(f"  face_detected={out.get('face_detected')}  image_size={img.size}")
        print(f"  scores: {scores}")
        print()


if __name__ == "__main__":
    main()
