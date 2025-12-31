THIS_IS_THE_FILE_I_AM_EDITING = False

import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
import os
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import tempfile

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Weights", "violence_detector_v1.h5")

SEQUENCE_LENGTH = 16
IMAGE_HEIGHT, IMAGE_WIDTH = 96, 96
CLASSES_LIST = ["NonViolence", "Violence"]

SMOOTHING_WINDOW = 5
CONFIDENCE_ALERT_THRESHOLD = 0.85

# ---------------- LOAD MODEL (ONCE) ----------------
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# ---------------- FASTAPI APP ----------------
app = FastAPI(title="Violence Detection API")

# ---------------- CORE PREDICTION LOGIC ----------------
def predict_video(video_path: str):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Unable to open video file")

    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)

    final_label = "NonViolence"
    final_confidence = 0.0
    alert_triggered = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        normalized = resized / 255.0
        frame_buffer.append(normalized)

        if len(frame_buffer) == SEQUENCE_LENGTH:
            input_tensor = np.expand_dims(np.array(frame_buffer), axis=0)
            pred = model.predict(input_tensor, verbose=0)[0]

            prediction_buffer.append(pred)
            avg_pred = np.mean(prediction_buffer, axis=0)

            class_index = np.argmax(avg_pred)
            confidence = float(avg_pred[class_index])
            label = CLASSES_LIST[class_index]

            if label == "Violence" and confidence >= CONFIDENCE_ALERT_THRESHOLD:
                final_label = "Violence"
                final_confidence = confidence
                alert_triggered = True
                break

            final_label = label
            final_confidence = confidence

    cap.release()

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "label": final_label,
        "confidence": round(final_confidence, 4),
        "alert": alert_triggered
    }

# ---------------- API ENDPOINT ----------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = predict_video(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

    return result
