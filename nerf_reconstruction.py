# ============================================================
# Week 7 Assignment: Neural Radiance Fields (NeRF)
# 3D Scene Reconstruction from 2D Images
# Dataset: NeRF Synthetic Chair Dataset
# ============================================================

# ── STEP 1: Imports ─────────────────────────────────────────
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import imageio.v2 as imageio
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Create output folder
os.makedirs("outputs", exist_ok=True)

# ── STEP 2: Load and Preprocess 2D Images ───────────────────
# Commit: "Loaded and preprocessed 2D images for NeRF"

DATASET_PATH = "./data/nerf_chair_images"

# Recursively gather all PNG/JPG images (handles subfolders like train/test/val)
image_files = sorted(
    glob.glob(os.path.join(DATASET_PATH, "**", "*.png"), recursive=True) +
    glob.glob(os.path.join(DATASET_PATH, "**", "*.jpg"), recursive=True)
)

if len(image_files) == 0:
    raise FileNotFoundError(f"No images found in {DATASET_PATH}. Check the path.")

print(f"[INFO] Found {len(image_files)} images.")

# Load and normalize images (RGB, resize to 64x64 for speed)
TARGET_SIZE = (64, 64)

def load_image(path):
    """Load image, convert to RGB, resize, and normalize to [0,1]."""
    img = imageio.imread(path)
    # Handle RGBA by keeping RGB channels only
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    # Resize using numpy slicing (simple nearest-neighbour)
    from PIL import Image
    img_pil = Image.fromarray(img).convert("RGB").resize(TARGET_SIZE)
    return np.array(img_pil) / 255.0

processed_images = [load_image(f) for f in image_files[:50]]  # cap at 50 for speed
print(f"[INFO] Loaded {len(processed_images)} images, shape: {processed_images[0].shape}")

# --- OUTPUT 1: Sample Images Grid ---
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("Sample Chair Images from NeRF Dataset", fontsize=14, fontweight="bold")
for i, ax in enumerate(axes.flat):
    if i < len(processed_images):
        ax.imshow(processed_images[i])
        ax.set_title(f"View {i+1}", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/sample_images.png", dpi=150)
plt.close()
print("[SAVED] outputs/sample_images.png")