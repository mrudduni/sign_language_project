"""
Gesture Detection + Speech 
Press Q to quit.
"""

import cv2
import numpy as np
import pyttsx3
import tensorflow as tf
import time

# Config 
MODEL_PATH  = "gesture_model.h5"
IMG_SIZE    = 224
THRESHOLD   = 0.75      # min confidence to speak
COOLDOWN    = 3.0       # seconds before repeating the same word
VOTE_FRAMES = 10        # frames that must agree before confirming

GESTURE_CLASSES = [
    "hello", "come", "yes", "no", "please",
    "stop", "water", "help", "sorry", "thank_you", "up", "down"
]
TEMPORAL = {"hello", "come"}

# Speech 
engine = pyttsx3.init()
engine.setProperty("rate", 150)
last_spoken = ""
last_time   = 0.0

def speak(label):
    global last_spoken, last_time
    if label == last_spoken and time.time() - last_time < COOLDOWN:
        return
    last_spoken = label
    last_time   = time.time()
    engine.say(label.replace("_", " "))
    engine.runAndWait()

# Hand detection (YCrCb skin mask) 
def get_hand(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mask  = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 3000:
        return None

    x, y, w, h = cv2.boundingRect(c)
    size = max(w, h) + 60
    cx, cy = x + w // 2, y + h // 2
    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(frame.shape[1], cx + size // 2)
    y2 = min(frame.shape[0], cy + size // 2)
    return x1, y1, x2, y2

def preprocess(frame, box):
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    img  = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    return img

#Load model 
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Ready. Press Q to quit.\n")

cap  = cv2.VideoCapture(0)
votes    = []
temp_buf = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    box   = get_hand(frame)
    label, conf, confirmed = "---", 0.0, False

    if box:
        img = preprocess(frame, box)
        if img is not None:
            probs = model.predict(img[np.newaxis, ..., np.newaxis], verbose=0)[0]
            label = GESTURE_CLASSES[int(np.argmax(probs))]
            conf  = float(np.max(probs))

            if label in TEMPORAL:
                temp_buf.append(img)
                if len(temp_buf) >= 30:
                    mean_img = np.mean(temp_buf, axis=0)
                    p2       = model.predict(mean_img[np.newaxis, ..., np.newaxis], verbose=0)[0]
                    label, conf = GESTURE_CLASSES[int(np.argmax(p2))], float(np.max(p2))
                    temp_buf.clear()
                    if conf >= THRESHOLD:
                        confirmed = True
                        speak(label)
            else:
                temp_buf.clear()
                votes.append(label)
                if len(votes) > VOTE_FRAMES:
                    votes.pop(0)
                if len(votes) == VOTE_FRAMES and len(set(votes)) == 1 and conf >= THRESHOLD:
                    confirmed = True
                    speak(label)
                    votes.clear()

        color = (0, 255, 0) if confirmed else (0, 200, 255)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
    else:
        votes.clear()

    # Display
    cv2.putText(frame, f"Gesture: {label}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(frame, f"Conf: {conf*100:.1f}%", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    if confirmed:
        cv2.putText(frame, "SPEAKING", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()