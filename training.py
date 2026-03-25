"""
Training script for MACS Project 2 gesture recognition CNN.

Data layout (produced by data collector + Augmentor):
    data/
        <static_gesture>/
            <gesture>_0000.jpg
            ...
            output/                  ← Augmentor puts generated images here
                gesture_XXXX_*.jpg
        <temporal_gesture>/
            seq_0000/
                frame_000.jpg
                ...
                output/              ← Augmentor puts generated frames here
            seq_0001/
            ...

Static gestures  : yes, no, please, stop, water, help, sorry, thank_you, up, down
Temporal gestures: hello, come  (sequences averaged into one image)

Train / val split: 80 / 20, stratified by class.
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR        = "data"
IMG_SIZE        = 224
VAL_SPLIT       = 0.1    # 10% validation
TEST_SPLIT      = 0.1    # 10% test  → 80% train / 10% val / 10% test
RANDOM_SEED     = 42
EPOCHS          = 30
BATCH_SIZE      = 32
MODEL_SAVE_PATH = "gesture_model.h5"

GESTURE_CLASSES = [
    "hello",      # 0  temporal
    "come",       # 1  temporal
    "yes",        # 2
    "no",         # 3
    "please",     # 4
    "stop",       # 5
    "water",      # 6
    "help",       # 7
    "sorry",      # 8
    "thank_you",  # 9
    "up",         # 10
    "down",       # 11
]

TEMPORAL_GESTURES = {"hello", "come"}
NUM_CLASSES       = len(GESTURE_CLASSES)

# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_jpgs(folder: str) -> list[str]:
    """
    Return sorted list of all .jpg paths in *folder* AND its output/ subfolder
    (where Augmentor saves generated images), if they exist.
    """
    paths = []
    for search_dir in [folder, os.path.join(folder, "output")]:
        if os.path.isdir(search_dir):
            for fname in sorted(os.listdir(search_dir)):
                if fname.lower().endswith(".jpg"):
                    paths.append(os.path.join(search_dir, fname))
    return paths


def load_image(path: str) -> np.ndarray | None:
    """Read a grayscale image, resize to IMG_SIZE, normalise to [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.astype(np.float32) / 255.0

# ── Data loading ──────────────────────────────────────────────────────────────

def load_static_samples(gesture: str, label: int) -> list:
    """
    Load every .jpg from data/<gesture>/ and data/<gesture>/output/.
    Each image is one sample.
    """
    samples = []
    folder  = os.path.join(DATA_DIR, gesture)

    if not os.path.isdir(folder):
        print(f"  [WARN] folder not found: {folder}")
        return samples

    for path in collect_jpgs(folder):
        img = load_image(path)
        if img is not None:
            samples.append((img, label))

    return samples


def load_temporal_samples(gesture: str, label: int) -> list:
    """
    Load every sequence folder from data/<gesture>/seq_NNNN/.
    Frames are read from both the sequence folder AND its output/ subfolder,
    then averaged into one representative image per sequence.
    """
    samples     = []
    gesture_dir = os.path.join(DATA_DIR, gesture)

    if not os.path.isdir(gesture_dir):
        print(f"  [WARN] folder not found: {gesture_dir}")
        return samples

    seq_dirs = sorted([
        d for d in os.listdir(gesture_dir)
        if os.path.isdir(os.path.join(gesture_dir, d))
        and d != "output"          # skip top-level output/ if it exists
    ])

    for seq_name in seq_dirs:
        seq_path = os.path.join(gesture_dir, seq_name)
        frames   = []

        for path in collect_jpgs(seq_path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            frames.append(img.astype(np.float32))

        if not frames:
            continue

        mean_frame = np.mean(frames, axis=0) / 255.0
        samples.append((mean_frame, label))

    return samples


def load_all_data():
    """
    Load every gesture class (original + Augmentor output subfolders).
    Returns:
        X         : float32 (N, IMG_SIZE, IMG_SIZE, 1)
        y         : int32   (N, NUM_CLASSES)  one-hot
        y_raw     : int32   (N,)              for stratified split
    """
    all_samples = []

    for label, gesture in enumerate(GESTURE_CLASSES):
        if gesture in TEMPORAL_GESTURES:
            samples = load_temporal_samples(gesture, label)
        else:
            samples = load_static_samples(gesture, label)

        print(f"  {gesture:12s} (class {label:2d}): {len(samples)} samples")
        all_samples.extend(samples)

    if not all_samples:
        raise RuntimeError(
            "No samples loaded. Run the data collector (and optionally augment.py) first.")

    images = np.array([s[0] for s in all_samples], dtype=np.float32)
    labels = np.array([s[1] for s in all_samples], dtype=np.int32)

    images = images[..., np.newaxis]                          # (N,H,W) → (N,H,W,1)
    labels_onehot = to_categorical(labels, num_classes=NUM_CLASSES)

    return images, labels_onehot, labels

# ── Model ─────────────────────────────────────────────────────────────────────

def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("MACS Project 2 - Gesture Recognition Training")

    print("\nLoading dataset (including Augmentor output/ subfolders)...")
    X, y, y_raw = load_all_data()
    print(f"\nTotal samples : {len(X)}")
    print(f"Image shape   : {X.shape[1:]}")
    print(f"Classes       : {NUM_CLASSES}")

    # First split off the test set, then split remainder into train / val
    X_temp, X_test, y_temp, y_test, y_raw_temp, _ = train_test_split(
        X, y, y_raw,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y_raw
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=VAL_SPLIT / (1.0 - TEST_SPLIT),   # correct proportion
        random_state=RANDOM_SEED,
        stratify=y_raw_temp
    )
    print(f"\nTrain samples : {len(X_train)}")
    print(f"Val samples   : {len(X_val)}")
    print(f"Test samples  : {len(X_test)}")

    print("\nBuilding CNN...")
    model = build_cnn((IMG_SIZE, IMG_SIZE, 1), NUM_CLASSES)
    model.summary()

    print("\nTraining...")
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val)
    )

    # ── Accuracy summary ──────────────────────────────────────────────────────
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss,   val_acc   = model.evaluate(X_val,   y_val,   verbose=0)
    test_loss,  test_acc  = model.evaluate(X_test,  y_test,  verbose=0)

    print("\n" + "="*45)
    print(f"  {'Split':<10} {'Loss':>8}  {'Accuracy':>10}")
    print("-"*45)
    print(f"  {'Train':<10} {train_loss:>8.4f}  {train_acc*100:>9.2f}%")
    print(f"  {'Validation':<10} {val_loss:>8.4f}  {val_acc*100:>9.2f}%")
    print(f"  {'Test':<10} {test_loss:>8.4f}  {test_acc*100:>9.2f}%")
    print("="*45)

    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")

    # ── Per-class breakdown on test set ───────────────────────────────────────
    print("\nPer-class accuracy on test set:")
    preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
    truth = np.argmax(y_test, axis=1)
    for i, gesture in enumerate(GESTURE_CLASSES):
        mask = truth == i
        if mask.sum() == 0:
            continue
        correct = (preds[mask] == truth[mask]).sum()
        total   = mask.sum()
        print(f"  {gesture:12s}: {correct}/{total}  ({correct/total*100:.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()