# Hand Gesture Recognition with Speech Output

A real-time hand gesture recognition system that classifies 12 sign language gestures using a CNN and converts the recognized gesture into spoken output via text-to-speech.

---

## Project Structure

```
.
├── datacollection.py   Webcam-based data collector (static + temporal gestures)
├── augment.py          Data augmentation using Augmentor
├── training.py         CNN training script with train / val / test evaluation
├── TTS.py              Real-time gesture detection and speech output
├── requirements.txt    Python dependencies
└── data/               Dataset folder (git-ignored, created at collection time)
    ├── yes/
    │   ├── yes_0000.jpg ...
    │   └── output/         Augmentor-generated images (auto-loaded by training.py)
    ├── hello/
    │   ├── seq_0000/
    │   │   ├── frame_000.jpg ...
    │   │   └── output/
    │   └── seq_0001/ ...
    └── ...
```

---

## Gestures

| Key | Gesture    | Type     | Description                              |
|-----|------------|----------|------------------------------------------|
| 0   | hello      | Temporal | Wave hand left-to-right near forehead    |
| 1   | come       | Temporal | Curl fingers inward repeatedly           |
| 2   | yes        | Static   | Thumbs up                                |
| 3   | no         | Static   | Thumbs down                              |
| 4   | please     | Static   | Peace symbol                             |
| 5   | stop       | Static   | Open palm facing outward                 |
| 6   | water      | Static   | Three fingers extended                   |
| 7   | help       | Static   | Closed fist, palm inward                 |
| 8   | sorry      | Static   | Four fingers extended                    |
| 9   | thank_you  | Static   | I love you sign (thumb + pinky extended) |
| A   | up         | Static   | Index finger pointing up                 |
| B   | down       | Static   | Index finger pointing down               |

Temporal gestures are captured as 30-frame sequences. Static gestures are captured as individual frames.

---

## Setup

**Requirements:** Python 3.9+, a working webcam.

```bash
# Create and activate virtual environment
python -m venv .venvmacs
source .venvmacs/bin/activate        # Windows: .venvmacs\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Workflow

### Step 1 - Collect Data

```bash
python datacollection.py
```

Controls inside the collector window:

| Key   | Action                              |
|-------|-------------------------------------|
| 0-9 / A-B | Select gesture                  |
| Space | Start / stop collecting             |
| C     | Clear all data for current gesture  |
| Q     | Quit                                |

Collected data is saved to the `data/` folder. Each static gesture saves 120 frames. Each temporal gesture saves 80 sequences of 30 frames each.

### Step 2 - Augment Data (optional but recommended)

```bash
python augment.py
```

Generates 300 additional images per gesture using random rotation, flipping, zoom, brightness, contrast, and shear. Augmented images are saved into `output/` subfolders inside each gesture/sequence folder. Training automatically picks them up — no manual file moving required.

### Step 3 - Train the Model

```bash
python training.py
```

The dataset is split 80% train / 10% validation / 10% test. After training, the script prints a summary table:

```
=============================================
  Split       Loss    Accuracy
---------------------------------------------
  Train      0.0812      97.50%
  Validation 0.1543      94.20%
  Test       0.1687      93.80%
=============================================
```

A per-class accuracy breakdown on the test set is also printed. The trained model is saved as `gesture_model.h5`.

### Step 4 - Run Gesture Detection with Speech

```bash
python TTS.py
```

Points the webcam at your hand. When a gesture is confirmed, the system speaks the corresponding word aloud.

Detection behaviour:
- Static gestures require 10 consecutive frames to agree with confidence above 75% before confirming.
- Temporal gestures (hello, come) accumulate 30 frames, compute a per-pixel mean, and run a single inference on the averaged image, matching exactly how the model was trained.
- A 3-second cooldown prevents the same word from being repeated continuously.

Press Q to quit.

---

## Model Architecture

```
Input: 224 x 224 x 1 (grayscale)

Conv2D(32)  -> MaxPool
Conv2D(64)  -> MaxPool
Conv2D(128) -> MaxPool
Flatten
Dense(128) + Dropout(0.5)
Dense(12, softmax)
```

Optimizer: Adam. Loss: categorical cross-entropy. Epochs: 30. Batch size: 32.

---

## Hand Isolation

All scripts use YCrCb colour-space skin segmentation combined with MOG2 background subtraction to isolate the hand from the webcam frame. The same preprocessing pipeline is used in both the data collector and the detection script to ensure consistency between training and inference.

---

## Notes

- Ensure consistent lighting when collecting data and during inference.
- The `data/` folder is git-ignored. Back it up separately before clearing.
- If `pyttsx3` produces no audio on Linux, install the `espeak` backend: `sudo apt install espeak`.
- On Windows, pyttsx3 uses the built-in SAPI5 engine and requires no additional setup.
