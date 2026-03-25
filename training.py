"""
Training script for MACS Project 2 gesture recognition CNN.

Data layout (produced by data collector):
    data/
        <static_gesture>/
            <gesture>_0000.jpg   (grayscale, 224x224)
            ...
        <temporal_gesture>/
            seq_0000/
                frame_000.jpg    (grayscale, 224x224)
                frame_001.jpg
                ...
            seq_0001/
            ...

Static gestures  (one image = one sample):
    yes, no, please, stop, water, help, sorry, thank_you, up, down

Temporal gestures (one sequence folder = one sample, averaged into one image):
    hello, come

The temporal sequences are collapsed into a single representative image by
computing the per-pixel mean across all frames in the sequence. This gives
the CNN a single 224x224 input regardless of gesture type, keeping the
architecture simple and consistent.

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

# CONFIGURATION 

DATA_DIR         = "data"
IMG_SIZE         = 224
TEST_SPLIT       = 0.2
RANDOM_SEED      = 42
EPOCHS           = 30
BATCH_SIZE       = 32
MODEL_SAVE_PATH  = "gesture_model.h5"

# Matches the GESTURES dict in data collector
GESTURE_CLASSES = [
    "hello",      # 0  - temporal
    "come",       # 1  - temporal
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

NUM_CLASSES = len(GESTURE_CLASSES)

# DATA LOADING 

def load_static_samples(gesture, label):
    """
    Load all JPG frames from data/<gesture>/ as individual samples.
    Returns list of (image_array, label) tuples.
    """
    samples = []
    folder = os.path.join(DATA_DIR, gesture)

    if not os.path.isdir(folder):
        print(f"[WARN] Folder not found: {folder}")
        return samples

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".jpg"):
            continue
        path = os.path.join(folder, fname)
        img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Collector already saves at IMG_SIZE x IMG_SIZE, but resize defensively
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        samples.append((img, label))

    return samples


def load_temporal_samples(gesture, label):
    """
    Load all sequence folders from data/<gesture>/seq_NNNN/.
    Each sequence is collapsed to one image by averaging all frames.
    Returns list of (image_array, label) tuples.
    """
    samples = []
    gesture_dir = os.path.join(DATA_DIR, gesture)

    if not os.path.isdir(gesture_dir):
        print(f"[WARN] Folder not found: {gesture_dir}")
        return samples

    seq_dirs = sorted([
        d for d in os.listdir(gesture_dir)
        if os.path.isdir(os.path.join(gesture_dir, d))
    ])

    for seq_name in seq_dirs:
        seq_path = os.path.join(gesture_dir, seq_name)
        frames = []

        for fname in sorted(os.listdir(seq_path)):
            if not fname.lower().endswith(".jpg"):
                continue
            path = os.path.join(seq_path, fname)
            img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            frames.append(img.astype(np.float32))

        if len(frames) == 0:
            continue

        # Average all frames in the sequence into one representative image
        mean_frame = np.mean(frames, axis=0) / 255.0
        samples.append((mean_frame, label))

    return samples


def load_all_data():
    """
    Load every gesture class and return X, y arrays ready for training.
    X shape: (N, IMG_SIZE, IMG_SIZE, 1)
    y shape: (N, NUM_CLASSES)  - one-hot encoded
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
            "No samples loaded. Run the data collector first and check DATA_DIR.")

    images = np.array([s[0] for s in all_samples], dtype=np.float32)
    labels = np.array([s[1] for s in all_samples], dtype=np.int32)

    # Add channel dimension: (N, H, W) -> (N, H, W, 1)
    images = images[..., np.newaxis]

    # One-hot encode labels
    labels_onehot = to_categorical(labels, num_classes=NUM_CLASSES)

    return images, labels_onehot, labels   # return raw labels too for stratify


#  MODEL 

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
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# TRAINING 
def main():

    print("MACS Project 2 - Gesture Recognition Training")
    # Load data
    print("\nLoading dataset...")
    X, y, y_raw = load_all_data()
    print(f"\nTotal samples : {len(X)}")
    print(f"Image shape   : {X.shape[1:]}")
    print(f"Classes       : {NUM_CLASSES}")

    # Train / validation split (stratified so each class is represented equally)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y_raw
    )

    print(f"\nTrain samples : {len(X_train)}")
    print(f"Val samples   : {len(X_val)}")

    # Build model
    print("\nBuilding CNN...")
    input_shape = (IMG_SIZE, IMG_SIZE, 1)
    model = build_cnn(input_shape, NUM_CLASSES)
    model.summary()

    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val)
    )

    # Final evaluation
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation accuracy : {val_acc * 100:.2f}%")
    print(f"Validation loss     : {val_loss:.4f}")

    # Save
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")

    # Per-class breakdown
    print("\nPer-class accuracy on validation set:")
    preds = np.argmax(model.predict(X_val, verbose=0), axis=1)
    truth = np.argmax(y_val, axis=1)
    for i, gesture in enumerate(GESTURE_CLASSES):
        mask    = truth == i
        if mask.sum() == 0:
            continue
        correct = (preds[mask] == truth[mask]).sum()
        total   = mask.sum()
        print(f"  {gesture:12s}: {correct}/{total}  ({correct/total*100:.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()