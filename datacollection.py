"""
  Hand Isolation Method : Skin-colour segmentation (YCrCb) + morphology
                          + largest-contour detection 
  Temporal gestures     : HELLO, COME  → saved as numbered FRAME SEQUENCES
  Static-motion gestures: YES, NO, PLEASE, STOP, WATER, HELP, SORRY,
                          THANK_YOU, UP, DOWN
  Storage layout:
      data/
        <gesture>/                ← static-motion: individual JPG frames
        <gesture>/<seq_NNNN>/     ← temporal     : one folder per sequence
          frame_000.jpg … frame_T.jpg
===============================================================================
  Controls:
    0-9 / A-K  → select gesture
    SPACE      → start / stop collecting
    C          → clear ALL data for current gesture (confirmation required)
    Q          → quit
===============================================================================
"""

import cv2
import numpy as np
import os
import sys

# ─────────────────────────── CONFIGURATION ──────────────────────────────────

BASE_FOLDER          = "data"
IMG_SIZE             = 224          # saved image size (px)
FRAMES_PER_GESTURE   = 120          # frames per static-motion gesture
MARGIN               = 30           # bounding-box margin (px)
MIN_HAND_AREA        = 3000         # min contour area to consider as hand

# Temporal gesture config
TEMPORAL_SEQ_FRAMES  = 30           # frames per single sequence clip
TEMPORAL_SEQUENCES   = 80           # number of sequences per temporal gesture

# ─── Skin detection thresholds (YCrCb colour space) ─────────────────────────
Y_MIN,  Y_MAX  =  0,  255
CR_MIN, CR_MAX = 133, 173
CB_MIN, CB_MAX =  77, 127

# ─────────────────────────── GESTURE REGISTRY ───────────────────────────────

# key → (display_name, is_temporal)
GESTURES = {
    '0': ('hello',     True ),   # temporal
    '1': ('come',      True ),   # temporal
    '2': ('yes',       False),
    '3': ('no',        False),
    '4': ('please',    False),
    '5': ('stop',      False),
    '6': ('water',     False),
    '7': ('help',      False),
    '8': ('sorry',     False),
    '9': ('thank_you', False),
    'a': ('up',        False),
    'b': ('down',      False),
}

GESTURE_HINTS = {
    'hello'    : "Wave hand left-to-right near forehead",
    'come'     : "Curl fingers inward repeatedly (beckoning)",
    'yes'      : "thumbs up",
    'no'       : "thumbs down",
    'please'   : "peace symbol",
    'stop'     : "Open palm facing outward, hold still",
    'water'    : "3 fingers extended",
    'help'     : "closed fist palm inward",
    'sorry'    : "4 fingers extended",
    'thank_you': "i love you sign (thumb + pinky extended)",
    'up'       : "Index finger pointing up, raise arm",
    'down'     : "Index finger pointing down, lower arm",
}

# ─────────────────────────── DIRECTORY SETUP ────────────────────────────────

for key, (gesture, is_temporal) in GESTURES.items():
    path = os.path.join(BASE_FOLDER, gesture)
    os.makedirs(path, exist_ok=True)

# ─────────────────────────── STATE ──────────────────────────────────────────

current_key       = None
is_collecting     = False

# counters for static-motion gestures
frame_counts = {g: 0 for g, _ in GESTURES.values()}

# load existing frame counts from disk
for g, is_t in GESTURES.values():
    folder = os.path.join(BASE_FOLDER, g)
    if is_t:
        seq_dirs = [d for d in os.listdir(folder)
                    if os.path.isdir(os.path.join(folder, d))]
        frame_counts[g] = len(seq_dirs)          # count completed sequences
    else:
        jpgs = [f for f in os.listdir(folder) if f.endswith('.jpg')]
        frame_counts[g] = len(jpgs)

# temporal gesture collection state
temporal_buffer     = []             # frames accumulated for current sequence
temporal_seq_idx    = {}             # next sequence index per temporal gesture
for g, is_t in GESTURES.values():
    if is_t:
        folder = os.path.join(BASE_FOLDER, g)
        existing = [d for d in os.listdir(folder)
                    if os.path.isdir(os.path.join(folder, d))]
        temporal_seq_idx[g] = len(existing)

# calibration / ROI state
calibration_done    = False
bg_subtractor       = cv2.createBackgroundSubtractorMOG2(
                          history=200, varThreshold=25, detectShadows=False)

# ═══════════════════════════ HAND ISOLATION ══════════════════════════════════

def skin_mask(frame: np.ndarray) -> np.ndarray:
    """Return binary mask of likely skin pixels using YCrCb thresholding."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mask  = cv2.inRange(ycrcb,
                        (Y_MIN,  CR_MIN, CB_MIN),
                        (Y_MAX,  CR_MAX, CB_MAX))
    # Morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    return mask


def motion_mask(frame: np.ndarray) -> np.ndarray:
    """Return foreground mask using background subtraction."""
    fg = bg_subtractor.apply(frame)
    _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg     = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    return fg


def get_hand_region(frame: np.ndarray):
    """
    Combine skin segmentation + motion cues to find the dominant hand contour.
    Returns (contour, x_min, y_min, x_max, y_max) or None if not found.
    """
    smask = skin_mask(frame)
    mmask = motion_mask(frame)

    # For static gestures use skin-only; blend with motion for robustness
    combined = cv2.bitwise_or(smask, cv2.bitwise_and(smask, mmask))

    contours, _ = cv2.findContours(combined,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Largest contour above minimum area
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_HAND_AREA:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    size = max(w, h) + MARGIN * 2
    cx   = x + w // 2
    cy   = y + h // 2

    x_min = max(0, cx - size // 2)
    y_min = max(0, cy - size // 2)
    x_max = min(frame.shape[1], cx + size // 2)
    y_max = min(frame.shape[0], cy + size // 2)

    return largest, x_min, y_min, x_max, y_max


# ═══════════════════════════ SAVING ══════════════════════════════════════════

def save_frame(frame: np.ndarray, gesture: str, count: int,
               region) -> bool:
    """Crop, grayscale, resize and save one frame for a static gesture."""
    if region is None:
        return False
    _, x_min, y_min, x_max, y_max = region
    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return False
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    path    = os.path.join(BASE_FOLDER, gesture,
                           f"{gesture}_{count:04d}.jpg")
    return cv2.imwrite(path, resized)


def save_temporal_sequence(frames_list: list, gesture: str,
                            seq_idx: int) -> bool:
    """
    Save a list of cropped frames as a numbered sequence folder.
    Folder: data/<gesture>/seq_<NNNN>/frame_000.jpg … frame_NNN.jpg
    """
    if not frames_list:
        return False
    seq_name = f"seq_{seq_idx:04d}"
    seq_dir  = os.path.join(BASE_FOLDER, gesture, seq_name)
    os.makedirs(seq_dir, exist_ok=True)
    for i, f in enumerate(frames_list):
        path = os.path.join(seq_dir, f"frame_{i:03d}.jpg")
        cv2.imwrite(path, f)
    return True


def prepare_temporal_frame(frame: np.ndarray, region) -> np.ndarray | None:
    """Crop + grayscale + resize one frame for a temporal sequence."""
    if region is None:
        return None
    _, x_min, y_min, x_max, y_max = region
    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    return resized


# ═══════════════════════════ HUD / DRAWING ═══════════════════════════════════

PALETTE = {
    'bg'      : (15,  15,  20),
    'accent'  : (0,   220, 180),
    'warn'    : (0,   165, 255),
    'danger'  : (50,  50,  220),
    'success' : (50,  200, 80),
    'text'    : (230, 230, 230),
    'dim'     : (100, 100, 100),
    'temporal': (220, 100, 220),
}


def overlay_rect(frame, x1, y1, x2, y2, color, alpha=0.55):
    roi    = frame[y1:y2, x1:x2]
    canvas = np.full_like(roi, color[::-1])   # BGR
    cv2.addWeighted(canvas, alpha, roi, 1 - alpha, 0, roi)
    frame[y1:y2, x1:x2] = roi


def put(frame, text, pos, scale=0.55, color=None, thickness=1, bold=False):
    if color is None:
        color = PALETTE['text']
    if bold:
        cv2.putText(frame, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness + 2)
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_progress_bar(frame, x, y, w, h, ratio, fg_color, label="", count_text=""):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), -1)
    filled = int(w * min(ratio, 1.0))
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + h), fg_color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), PALETTE['dim'], 1)
    if label:
        put(frame, label, (x - 155, y + h - 2), scale=0.42,
            color=PALETTE['text'])
    if count_text:
        put(frame, count_text, (x + w + 6, y + h - 2), scale=0.42,
            color=PALETTE['dim'])


def draw_hud(frame, hand_detected: bool, region,
             temp_buf_len: int):
    h, w = frame.shape[:2]

    # ── top bar ──────────────────────────────────────────────────────────────
    overlay_rect(frame, 0, 0, w, 55, PALETTE['bg'])
    put(frame, "MACS P2  |  GESTURE DATA COLLECTOR",
        (12, 35), scale=0.85, color=PALETTE['accent'],
        thickness=2, bold=True)

    # ── right panel ──────────────────────────────────────────────────────────
    panel_x = w - 310
    overlay_rect(frame, panel_x, 55, w, h, PALETTE['bg'], alpha=0.65)

    put(frame, "GESTURES", (panel_x + 10, 80),
        scale=0.55, color=PALETTE['accent'], thickness=1)

    for i, (k, (g, is_t)) in enumerate(GESTURES.items()):
        total   = TEMPORAL_SEQUENCES if is_t else FRAMES_PER_GESTURE
        count   = frame_counts[g]
        ratio   = count / total
        col     = PALETTE['temporal'] if is_t else (
                  PALETTE['success']  if ratio >= 1.0 else PALETTE['warn'])
        y_bar   = 95 + i * 38

        label   = f"[{k.upper()}] {g.upper()}"
        if is_t:
            label += " ◈"
        draw_progress_bar(frame, panel_x + 10, y_bar, 160, 14,
                          ratio, col,
                          label=label,
                          count_text=f"{count}/{total}")

    # ── current gesture status ────────────────────────────────────────────────
    if current_key:
        g, is_t = GESTURES[current_key]
        total   = TEMPORAL_SEQUENCES if is_t else FRAMES_PER_GESTURE
        count   = frame_counts[g]

        overlay_rect(frame, 0, h - 130, panel_x, h, PALETTE['bg'], alpha=0.65)

        tag = "TEMPORAL SEQUENCE" if is_t else "STATIC-MOTION"
        tc  = PALETTE['temporal'] if is_t else PALETTE['accent']
        put(frame, tag, (12, h - 108), scale=0.5, color=tc)

        put(frame, g.upper(), (12, h - 80),
            scale=1.1, color=PALETTE['text'], thickness=2, bold=True)

        hint = GESTURE_HINTS.get(g, "")
        put(frame, hint, (12, h - 55), scale=0.45, color=PALETTE['dim'])

        if is_t and is_collecting:
            buf_ratio = temp_buf_len / TEMPORAL_SEQ_FRAMES
            bar_col   = PALETTE['temporal']
            draw_progress_bar(frame, 12, h - 38, 300, 16,
                              buf_ratio, bar_col,
                              count_text=f"buf {temp_buf_len}/{TEMPORAL_SEQ_FRAMES}")
            put(frame, "RECORDING SEQUENCE…", (320, h - 26),
                scale=0.5, color=PALETTE['temporal'])
        elif not is_t:
            put(frame, f"{count} / {total}  frames",
                (12, h - 28), scale=0.55,
                color=PALETTE['success'] if count >= total else PALETTE['warn'])

    # ── controls footer ───────────────────────────────────────────────────────
    overlay_rect(frame, 0, 55, panel_x - 0, 90, PALETTE['bg'], alpha=0.55)
    put(frame, "0-9/A-B: select   SPACE: collect   C: clear   Q: quit",
        (12, 78), scale=0.46, color=PALETTE['dim'])

    # ── hand detection indicator ─────────────────────────────────────────────
    if hand_detected and region:
        _, x1, y1, x2, y2 = region
        col = (PALETTE['temporal'] if (current_key and
                                        GESTURES[current_key][1] and
                                        is_collecting)
               else PALETTE['accent'] if is_collecting
               else PALETTE['warn'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

        put(frame, "HAND", (x1 + 4, y1 - 6), scale=0.45, color=col)

    hd_col  = PALETTE['success'] if hand_detected else PALETTE['danger']
    hd_text = "● HAND DETECTED" if hand_detected else "○ NO HAND"
    put(frame, hd_text, (12, 112), scale=0.5, color=hd_col)

    col_col  = PALETTE['temporal'] if (is_collecting and current_key and
                                        GESTURES.get(current_key, ('', False))[1]) \
               else PALETTE['success'] if is_collecting else PALETTE['dim']
    col_text = "● COLLECTING" if is_collecting else "○ STANDBY"
    put(frame, col_text, (12, 133), scale=0.5, color=col_col)


def draw_preview(frame, region, h_frame):
    """Render a small grayscale preview of the cropped hand region."""
    if region is None:
        return
    _, x1, y1, x2, y2 = region
    crop = h_frame[y1:y2, x1:x2]
    if crop.size == 0:
        return
    pw = 120
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    preview = cv2.resize(gray, (pw, pw))
    preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

    fx = frame.shape[1] - 320
    fy = frame.shape[0] - pw - 30
    frame[fy:fy + pw, fx:fx + pw] = preview
    cv2.rectangle(frame, (fx, fy), (fx + pw, fy + pw),
                  PALETTE['accent'], 1)
    put(frame, "preview", (fx, fy - 6), scale=0.38, color=PALETTE['dim'])


# ═══════════════════════════ MAIN LOOP ═══════════════════════════════════════

def main():
    global current_key, is_collecting, temporal_buffer

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    print("  MACS — GESTURE DATA COLLECTION")
    print("   marks TEMPORAL gestures (HELLO, COME)")
    print("     Temporal gestures save SEQUENCE FOLDERS")
    print("     Each sequence = 30 frames of continuous motion")
    print("  Controls:")
    print("    0-9 / A-B  → select gesture")
    print("    SPACE      → start / stop collecting")
    print("    C          → clear data for current gesture")
    print("    Q          → quit")
    for k, (g, is_t) in GESTURES.items():
        tag = " [TEMPORAL]" if is_t else ""
        print(f"  [{k.upper()}] {g.upper():15s}{tag}")
        print(f"       hint: {GESTURE_HINTS.get(g, '')}")
    print("=" * 70)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame capture failed.")
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        # ── Hand detection ────────────────────────────────────────────────────
        region       = get_hand_region(frame)
        hand_detected = region is not None

        # ── Collection logic ─────────────────────────────────────────────────
        if is_collecting and hand_detected and current_key:
            g, is_t = GESTURES[current_key]

            if is_t:
                # ── TEMPORAL gesture ─────────────────────────────────────────
                f = prepare_temporal_frame(frame, region)
                if f is not None:
                    temporal_buffer.append(f)

                if len(temporal_buffer) >= TEMPORAL_SEQ_FRAMES:
                    seq_idx = temporal_seq_idx[g]
                    ok = save_temporal_sequence(temporal_buffer, g, seq_idx)
                    if ok:
                        temporal_seq_idx[g] += 1
                        frame_counts[g]     += 1
                        print(f"  [{g}] sequence {seq_idx:04d} saved "
                              f"({frame_counts[g]}/{TEMPORAL_SEQUENCES})")

                        if frame_counts[g] >= TEMPORAL_SEQUENCES:
                            is_collecting  = False
                            temporal_buffer = []
                            print(f"   [{g}] all sequences complete!")
                        else:
                            temporal_buffer = []   # ready for next sequence

            else:
                # ── STATIC-MOTION gesture ────────────────────────────────────
                total = FRAMES_PER_GESTURE
                if frame_counts[g] < total:
                    if save_frame(frame, g, frame_counts[g], region):
                        frame_counts[g] += 1
                        print(f"  ✓ [{g}] {frame_counts[g]}/{total}")

                        if frame_counts[g] >= total:
                            is_collecting = False
                            print(f"  [{g}] complete!")

        # ── Draw preview & HUD ────────────────────────────────────────────────
        draw_preview(display, region, frame)
        draw_hud(display, hand_detected, region,
                 temp_buf_len=len(temporal_buffer))

        cv2.imshow("MACS — Data Collector", display)

        # ── Key handling ─────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        key_char = chr(key).lower() if 0 <= key < 256 else ''

        if key_char == 'q':
            print("\n[QUIT] Exiting…")
            break

        elif key_char == ' ':
            if current_key is None:
                print("[WARN] Select a gesture first.")
            else:
                g, is_t = GESTURES[current_key]
                total   = TEMPORAL_SEQUENCES if is_t else FRAMES_PER_GESTURE

                if frame_counts[g] >= total:
                    print(f"[INFO] [{g}] already complete.")
                else:
                    is_collecting   = not is_collecting
                    temporal_buffer = []        # reset buffer on toggle
                    verb = "STARTED" if is_collecting else "STOPPED"
                    print(f"[{verb}] collecting [{g}]")

        elif key_char == 'c' and current_key:
            g, is_t = GESTURES[current_key]
            print(f"[CLEAR] Confirm delete all data for [{g}]? (y/n)")
            confirm = cv2.waitKey(5000) & 0xFF
            if chr(confirm).lower() == 'y':
                folder = os.path.join(BASE_FOLDER, g)
                import shutil
                for item in os.listdir(folder):
                    item_path = os.path.join(folder, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                frame_counts[g]        = 0
                temporal_seq_idx[g]    = 0
                temporal_buffer        = []
                is_collecting          = False
                print(f"[CLEAR] [{g}] data deleted.")
            else:
                print("[CLEAR] Cancelled.")

        elif key_char in GESTURES:
            new_key = key_char
            if new_key != current_key:
                is_collecting   = False
                temporal_buffer = []
                current_key     = new_key
                g, is_t         = GESTURES[current_key]
                total           = TEMPORAL_SEQUENCES if is_t else FRAMES_PER_GESTURE
                tag             = " [TEMPORAL SEQUENCE]" if is_t else ""
                print(f"\n[SELECT] {g.upper()}{tag}")
                print(f"         progress: {frame_counts[g]}/{total}")
                print(f"         hint    : {GESTURE_HINTS.get(g, '')}")
                print("         SPACE to start collecting")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("  COLLECTION SUMMARY")
   
    total_all = 0
    for k, (g, is_t) in GESTURES.items():
        total    = TEMPORAL_SEQUENCES if is_t else FRAMES_PER_GESTURE
        count    = frame_counts[g]
        status   = " COMPLETE" if count >= total else "PARTIAL "
        tag      = " ◈ temporal" if is_t else ""
        print(f"  {status}  {g.upper():15s}: {count:3d}/{total}{tag}")
        total_all += count

    print(f"  Total saved units : {total_all}")
    print(f"  Data root         : {os.path.abspath(BASE_FOLDER)}/")
    


if __name__ == "__main__":
    main()