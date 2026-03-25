"""
Data Augmentation for MACS Project 2
Uses Augmentor to expand the dataset for static gestures.
Temporal gestures (hello, come) are augmented frame-by-frame inside each sequence folder.

Run:  python augment.py
"""

import os
import Augmentor

DATA_DIR        = "data"
SAMPLES_TO_GEN  = 300    # extra images to generate per gesture/sequence

STATIC_GESTURES = ["yes", "no", "please", "stop", "water",
                   "help", "sorry", "thank_you", "up", "down"]

TEMPORAL_GESTURES = ["hello", "come"]


def make_pipeline(folder):
    """Create a standard augmentation pipeline for a given folder."""
    p = Augmentor.Pipeline(folder)
    p.rotate(probability=0.7, max_left_rotation=15, max_right_rotation=15)
    p.flip_left_right(probability=0.5)
    p.zoom(probability=0.5, min_factor=0.9, max_factor=1.1)
    p.random_brightness(probability=0.5, min_factor=0.7, max_factor=1.3)
    p.random_contrast(probability=0.5, min_factor=0.7, max_factor=1.3)
    p.shear(probability=0.3, max_shear_left=10, max_shear_right=10)
    return p


# ── Static gestures ───────────────────────────────────────────────────────────
print("Augmenting static gestures...")
for gesture in STATIC_GESTURES:
    folder = os.path.join(DATA_DIR, gesture)
    if not os.path.isdir(folder):
        print(f"  [SKIP] {gesture} — folder not found")
        continue

    jpg_count = len([f for f in os.listdir(folder) if f.endswith(".jpg")])
    if jpg_count == 0:
        print(f"  [SKIP] {gesture} — no images found")
        continue

    print(f"  {gesture}: {jpg_count} images → +{SAMPLES_TO_GEN} augmented")
    p = make_pipeline(folder)
    p.sample(SAMPLES_TO_GEN)


# ── Temporal gestures (augment each sequence folder) ─────────────────────────
print("\nAugmenting temporal gestures (per sequence)...")
for gesture in TEMPORAL_GESTURES:
    gesture_dir = os.path.join(DATA_DIR, gesture)
    if not os.path.isdir(gesture_dir):
        print(f"  [SKIP] {gesture} — folder not found")
        continue

    seq_dirs = sorted([
        d for d in os.listdir(gesture_dir)
        if os.path.isdir(os.path.join(gesture_dir, d))
    ])

    if not seq_dirs:
        print(f"  [SKIP] {gesture} — no sequence folders found")
        continue

    print(f"  {gesture}: {len(seq_dirs)} sequences")
    for seq in seq_dirs:
        seq_path = os.path.join(gesture_dir, seq)
        p = make_pipeline(seq_path)
        p.sample(SAMPLES_TO_GEN)

print("\nDone! Augmented images saved in 'output/' subfolder inside each gesture folder.")
print("Move them into the main gesture folder before retraining if needed.")