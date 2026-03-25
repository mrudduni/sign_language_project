"""
MACS Project 2 — Real-time Gesture Detection + Speech Synthesis

Uses the CNN trained by training.py to classify hand gestures from
a webcam feed, then speaks the result aloud via pyttsx3.

Detection strategy
------------------
  Static gestures  : Run inference every frame. A "confirmed" detection
                     requires VOTE_WINDOW consecutive frames where the
                     top-1 prediction is the same gesture AND its mean
                     confidence exceeds CONFIDENCE_THRESHOLD.

  Temporal gestures: Accumulate TEMPORAL_SEQ_FRAMES frames into a
                     buffer, compute the per-pixel mean (mirroring what
                     training.py learned), run inference once on the
                     averaged image.

Speech cooldown
---------------
  After speaking a gesture label, the same label is silenced for
  SPEECH_COOLDOWN_SEC seconds to prevent rapid repetition.

Controls
--------
  Q  → quit
  S  → toggle speech on / off
  R  → reset temporal buffer
"""

import os
import sys
import time
import threading
import collections

import cv2
import numpy as np
import pyttsx3
import tensorflow as tf

# ──────────────────────────── CONFIGURATION ──────────────────────────────────

MODEL_PATH           = "gesture_model.h5"   # path to saved Keras model
IMG_SIZE             = 224                  # must match training.py

# Detection / voting
CONFIDENCE_THRESHOLD = 0.75   # min softmax confidence to accept a prediction
VOTE_WINDOW          = 10     # frames that must agree for a static confirmation

# Temporal gesture collection (mirrors training.py / data collector)
TEMPORAL_SEQ_FRAMES  = 30     # frames averaged into one representative image
TEMPORAL_GESTURES    = {"hello", "come"}

# Speech
SPEECH_COOLDOWN_SEC  = 3.0    # seconds before the same label can be spoken again
SPEECH_RATE          = 150    # words-per-minute for pyttsx3

# Hand isolation (YCrCb — mirrors data collector)
Y_MIN,  Y_MAX  =  0,  255
CR_MIN, CR_MAX = 133, 173
CB_MIN, CB_MAX =  77, 127
MIN_HAND_AREA  = 3000
MARGIN         = 30

# Class list — ORDER must match training.py GESTURE_CLASSES
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

# Human-readable speech phrases (index-aligned with GESTURE_CLASSES)
SPEECH_PHRASES = {
    "hello"    : "Hello!",
    "come"     : "Come here.",
    "yes"      : "Yes.",
    "no"       : "No.",
    "please"   : "Please.",
    "stop"     : "Stop!",
    "water"    : "Water.",
    "help"     : "Help me!",
    "sorry"    : "Sorry.",
    "thank_you": "Thank you.",
    "up"       : "Up.",
    "down"     : "Down.",
}

# UI colour palette (BGR)
PALETTE = {
    "bg"      : (18,  18,  18),
    "accent"  : (0,  210, 255),   # cyan
    "success" : (0,  220, 100),   # green
    "warning" : (0,  165, 255),   # orange
    "danger"  : (60,  60, 220),   # red
    "temporal": (200, 90, 255),   # purple
    "white"   : (240, 240, 240),
    "dim"     : (120, 120, 120),
}

# SPEECH ENGINE 
class SpeechEngine:
    """Thread-safe, non-blocking wrapper around pyttsx3."""

    def __init__(self, rate: int = SPEECH_RATE):
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)
        self._lock    = threading.Lock()
        self._busy    = False
        self.enabled  = True
        self._cooldowns: dict[str, float] = {}   # label → last-spoken timestamp

    def speak(self, label: str) -> bool:
        """
        Speak the phrase for *label* in a background thread.
        Returns True if speech was triggered, False if suppressed.
        """
        if not self.enabled:
            return False
        now = time.time()
        if now - self._cooldowns.get(label, 0) < SPEECH_COOLDOWN_SEC:
            return False   # still in cooldown

        with self._lock:
            if self._busy:
                return False
            self._busy = True

        self._cooldowns[label] = now
        phrase = SPEECH_PHRASES.get(label, label.replace("_", " "))

        def _run():
            self._engine.say(phrase)
            self._engine.runAndWait()
            with self._lock:
                self._busy = False

        threading.Thread(target=_run, daemon=True).start()
        return True

    def cooldown_remaining(self, label: str) -> float:
        elapsed = time.time() - self._cooldowns.get(label, 0)
        return max(0.0, SPEECH_COOLDOWN_SEC - elapsed)


# HAND ISOLATION 

_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=200, varThreshold=25, detectShadows=False
)


def _skin_mask(frame: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mask  = cv2.inRange(ycrcb,
                        (Y_MIN,  CR_MIN, CB_MIN),
                        (Y_MAX,  CR_MAX, CB_MAX))
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    return mask


def _motion_mask(frame: np.ndarray) -> np.ndarray:
    fg = _bg_subtractor.apply(frame)
    _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    return fg


def get_hand_region(frame: np.ndarray):
    """
    Returns (contour, x_min, y_min, x_max, y_max) or None.
    Identical logic to the data collector for consistent preprocessing.
    """
    smask    = _skin_mask(frame)
    mmask    = _motion_mask(frame)
    combined = cv2.bitwise_or(smask, cv2.bitwise_and(smask, mmask))

    contours, _ = cv2.findContours(combined,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_HAND_AREA:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    size = max(w, h) + MARGIN * 2
    cx, cy = x + w // 2, y + h // 2

    x_min = max(0, cx - size // 2)
    y_min = max(0, cy - size // 2)
    x_max = min(frame.shape[1], cx + size // 2)
    y_max = min(frame.shape[0], cy + size // 2)

    return largest, x_min, y_min, x_max, y_max


def crop_and_preprocess(frame: np.ndarray, region) -> np.ndarray | None:
    """
    Crop the hand region, convert to grayscale, resize to IMG_SIZE×IMG_SIZE,
    and return a float32 array in [0, 1] shaped (IMG_SIZE, IMG_SIZE).
    Returns None if the crop is empty.
    """
    _, x1, y1, x2, y2 = region
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    return resized.astype(np.float32) / 255.0


# INFERENCE

def predict(model, image: np.ndarray) -> tuple[str, float]:
    """
    Run one forward pass.
    *image* : float32 (IMG_SIZE, IMG_SIZE) in [0, 1]
    Returns (gesture_label, confidence).
    """
    tensor = image[np.newaxis, ..., np.newaxis]   # (1, H, W, 1)
    probs  = model.predict(tensor, verbose=0)[0]
    idx    = int(np.argmax(probs))
    return GESTURE_CLASSES[idx], float(probs[idx])


# HUD RENDERING

def _put(frame, text, pos, scale=0.5, color=None, thickness=1):
    color = color or PALETTE["white"]
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                cv2.LINE_AA)


def draw_hud(frame, state: dict):
    """Overlay all status information onto *frame* in-place."""
    h, w = frame.shape[:2]

    # Semi-transparent sidebar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.line(frame, (260, 0), (260, h), PALETTE["accent"], 1)

    # Title
    _put(frame, "MACS  GESTURE DETECTION", (10, 26),
         scale=0.52, color=PALETTE["accent"], thickness=1)
    cv2.line(frame, (10, 34), (250, 34), PALETTE["dim"], 1)

    y = 58

    #Hand status 
    hand_ok = state.get("hand_detected", False)
    col  = PALETTE["success"] if hand_ok else PALETTE["danger"]
    icon = "HAND  DETECTED" if hand_ok else "NO HAND"
    _put(frame, icon, (14, y), scale=0.48, color=col)
    y += 24

    # Mode 
    mode = state.get("mode", "static")
    mode_col = PALETTE["temporal"] if mode == "temporal" else PALETTE["accent"]
    _put(frame, f"MODE: {mode.upper()}", (14, y), scale=0.45, color=mode_col)
    y += 28

    # Temporal buffer bar
    if mode == "temporal":
        buf_len  = state.get("buf_len", 0)
        bar_w    = 220
        fill_w   = int(bar_w * buf_len / TEMPORAL_SEQ_FRAMES)
        _put(frame, f"BUFFER  {buf_len:2d}/{TEMPORAL_SEQ_FRAMES}", (14, y),
             scale=0.45, color=PALETTE["temporal"])
        y += 16
        cv2.rectangle(frame, (14, y),      (14 + bar_w, y + 8), PALETTE["dim"], 1)
        cv2.rectangle(frame, (14, y), (14 + fill_w,     y + 8), PALETTE["temporal"], -1)
        y += 22

    # Confidence bar 
    conf = state.get("confidence", 0.0)
    label_raw = state.get("label", "---")
    bar_w = 220
    fill_w = int(bar_w * conf)
    bar_col = (PALETTE["success"] if conf >= CONFIDENCE_THRESHOLD
               else PALETTE["warning"])

    _put(frame, f"CONF  {conf * 100:5.1f}%", (14, y), scale=0.48, color=bar_col)
    y += 16
    cv2.rectangle(frame, (14, y),         (14 + bar_w, y + 8), PALETTE["dim"], 1)
    cv2.rectangle(frame, (14, y), (14 + fill_w,        y + 8), bar_col, -1)
    # threshold marker
    tx = 14 + int(bar_w * CONFIDENCE_THRESHOLD)
    cv2.line(frame, (tx, y - 2), (tx, y + 10), PALETTE["white"], 1)
    y += 22

    # Prediction label 
    confirmed = state.get("confirmed", False)
    pred_col  = PALETTE["success"] if confirmed else PALETTE["dim"]
    display_label = label_raw.replace("_", " ").upper()
    _put(frame, display_label, (14, y),
         scale=0.75, color=pred_col, thickness=2)
    y += 34

    if confirmed:
        _put(frame, "CONFIRMED", (14, y), scale=0.45, color=PALETTE["success"])
    y += 24

    # Vote progress (static mode) 
    if mode == "static":
        votes    = state.get("votes", 0)
        vote_col = PALETTE["success"] if votes == VOTE_WINDOW else PALETTE["dim"]
        _put(frame, f"VOTE  {votes}/{VOTE_WINDOW}", (14, y),
             scale=0.45, color=vote_col)
        y += 24

    # Speech status 
    speech_on  = state.get("speech_enabled", True)
    speech_col = PALETTE["success"] if speech_on else PALETTE["danger"]
    _put(frame, f"SPEECH  {'ON' if speech_on else 'OFF'}", (14, y),
         scale=0.48, color=speech_col)
    y += 22

    cd = state.get("cooldown", 0.0)
    if cd > 0:
        _put(frame, f"cooldown {cd:.1f}s", (14, y), scale=0.4, color=PALETTE["dim"])
        y += 18

    # Last spoken 
    spoken = state.get("last_spoken", "")
    if spoken:
        _put(frame, f"SAID: {spoken.replace('_',' ').upper()}", (14, y),
             scale=0.45, color=PALETTE["accent"])
        y += 22

    # Controls legend (bottom of sidebar) 
    legend_y = h - 60
    cv2.line(frame, (10, legend_y - 6), (250, legend_y - 6), PALETTE["dim"], 1)
    _put(frame, "Q  quit    S  speech    R  reset",
         (10, legend_y + 6), scale=0.38, color=PALETTE["dim"])

    # Hand bounding box on main image 
    region = state.get("region")
    if region is not None:
        _, x1, y1, x2, y2 = region
        box_col = PALETTE["success"] if confirmed else PALETTE["accent"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, 2)

    # Small hand preview (top-right)
    preview = state.get("preview")
    if preview is not None:
        pw  = 110
        pfr = cv2.resize(preview, (pw, pw))
        pfr = cv2.cvtColor(pfr, cv2.COLOR_GRAY2BGR)
        fx  = w - pw - 10
        fy  = 10
        frame[fy:fy + pw, fx:fx + pw] = pfr
        cv2.rectangle(frame, (fx, fy), (fx + pw, fy + pw), PALETTE["accent"], 1)
        _put(frame, "hand preview", (fx, fy + pw + 12),
             scale=0.36, color=PALETTE["dim"])

    # FPS 
    fps = state.get("fps", 0.0)
    _put(frame, f"{fps:.0f} fps", (w - 68, h - 10),
         scale=0.4, color=PALETTE["dim"])


#  MAIN 

def main():
    # Load model
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] Model not found at '{MODEL_PATH}'.")
        print("        Run training.py first to generate the model file.")
        sys.exit(1)

    print(f"Loading model from '{MODEL_PATH}' …")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.")

    # Init speech engine 
    tts = SpeechEngine(rate=SPEECH_RATE)
    print("Speech engine ready. (pyttsx3)")

    # Webcam 
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    # Detection state
    vote_queue:    collections.deque = collections.deque(maxlen=VOTE_WINDOW)
    temporal_buf:  list              = []   # raw float32 frames
    last_confirmed: str              = ""
    last_spoken:    str              = ""

    # FPS tracking
    fps_counter = 0
    fps_timer   = time.time()
    fps_display = 0.0

    print("\nControls:  Q = quit   S = toggle speech   R = reset temporal buffer\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame read failed.")
            break

        frame = cv2.flip(frame, 1)   # mirror so it feels like a selfie-cam

        # Hand detection
        region       = get_hand_region(frame)
        hand_detected = region is not None

        label      = "---"
        confidence = 0.0
        confirmed  = False
        mode       = "static"
        preview    = None

        if hand_detected:
            img = crop_and_preprocess(frame, region)

            if img is not None:
                preview = (img * 255).astype(np.uint8)

                # Run inference
                raw_label, confidence = predict(model, img)

                # Decide mode from current prediction 
                if raw_label in TEMPORAL_GESTURES:
                    mode = "temporal"
                    temporal_buf.append(img)

                    if len(temporal_buf) >= TEMPORAL_SEQ_FRAMES:
                        mean_img              = np.mean(temporal_buf, axis=0)
                        label, confidence     = predict(model, mean_img)
                        temporal_buf.clear()
                        vote_queue.clear()

                        if confidence >= CONFIDENCE_THRESHOLD:
                            confirmed      = True
                            last_confirmed = label
                            if tts.speak(label):
                                last_spoken = label
                    else:
                        label = raw_label   # show live guess while buffering
                else:
                    mode = "static"
                    # Only buffer frames whose top prediction is NOT temporal
                    # (avoids polluting the static vote with transient predictions)
                    if raw_label not in TEMPORAL_GESTURES:
                        vote_queue.append((raw_label, confidence))

                    label = raw_label

                    if len(vote_queue) == VOTE_WINDOW:
                        labels_in_window = [v[0] for v in vote_queue]
                        confs_in_window  = [v[1] for v in vote_queue]
                        top_label        = max(set(labels_in_window),
                                               key=labels_in_window.count)
                        agreement        = labels_in_window.count(top_label)
                        mean_conf        = float(np.mean(
                            [c for l, c in zip(labels_in_window, confs_in_window)
                             if l == top_label]
                        ))

                        if (agreement == VOTE_WINDOW
                                and mean_conf >= CONFIDENCE_THRESHOLD):
                            confirmed      = True
                            last_confirmed = top_label
                            label          = top_label
                            confidence     = mean_conf
                            vote_queue.clear()

                            if tts.speak(top_label):
                                last_spoken = top_label
        else:
            # No hand — reset short-term buffers so stale votes don't linger
            vote_queue.clear()
            # Don't clear temporal_buf here; hand can briefly disappear mid-motion

        # FPS 
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display  = fps_counter / (time.time() - fps_timer)
            fps_counter  = 0
            fps_timer    = time.time()

        # Build HUD state dict 
        cd = tts.cooldown_remaining(last_spoken) if last_spoken else 0.0
        hud_state = {
            "hand_detected" : hand_detected,
            "mode"          : mode,
            "label"         : label,
            "confidence"    : confidence,
            "confirmed"     : confirmed,
            "votes"         : len(vote_queue),
            "buf_len"       : len(temporal_buf),
            "speech_enabled": tts.enabled,
            "cooldown"      : cd,
            "last_spoken"   : last_spoken,
            "region"        : region,
            "preview"       : preview,
            "fps"           : fps_display,
        }

        draw_hud(frame, hud_state)
        cv2.imshow("MACS — Gesture Detection", frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[QUIT] Exiting…")
            break
        elif key == ord('s'):
            tts.enabled = not tts.enabled
            print(f"[SPEECH] {'enabled' if tts.enabled else 'disabled'}")
        elif key == ord('r'):
            temporal_buf.clear()
            vote_queue.clear()
            print("[RESET] Buffers cleared.")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()