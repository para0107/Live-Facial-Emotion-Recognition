import base64
import io
import os
import sys
import json
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
FER_ROOT = os.environ.get("FER_ROOT", PROJECT_ROOT)
sys.path.insert(0, FER_ROOT)

from src.inference.predictor import EmotionPredictor

CHECKPOINT = os.environ.get(
    "CHECKPOINT_PATH",
    os.path.join(FER_ROOT, "checkpoints", "best_model.pth")
)
HAAR = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

app = FastAPI(title="FER WebSocket API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"Loading checkpoint from: {CHECKPOINT}")
if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

# Reduced from 10 → 5 frames.
# A 10-frame window at ~12fps = ~833ms of inertia. Once the model sees a
# genuine happy frame (e.g. a slight smile) it takes 10 more frames of
# contrary evidence to shift — causing the "stuck on happy" effect.
# 5 frames (~417ms) is still smooth enough to suppress jitter but reacts
# fast enough to track real expression changes.
predictor = EmotionPredictor(
    checkpoint_path=CHECKPOINT,
    smoothing_window=5,
)
print(f"Model loaded on {predictor.device}")

face_detector = cv2.CascadeClassifier(HAAR)


def decode_frame(b64: str) -> np.ndarray:
    data = base64.b64decode(b64.split(",")[-1])
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    predictor.reset_buffer()

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            frame_b64 = data.get("frame")
            if not frame_b64:
                continue

            frame = decode_frame(frame_b64)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )

            results = []
            for (x, y, w, h) in faces:
                face_crop = gray[y:y + h, x:x + w]
                probs = predictor.predict_smoothed(face_crop)
                top_idx = int(np.argmax(probs))
                top_class = predictor.class_names[top_idx]

                results.append({
                    "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "probs": {
                        predictor.class_names[i]: round(float(probs[i]), 4)
                        for i in range(len(predictor.class_names))
                    },
                    "emotion": top_class,
                    "confidence": round(float(probs[top_idx]), 4),
                })

            await ws.send_text(json.dumps({
                "faces": results,
                "face_count": len(results),
            }))

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(predictor.device),
        "checkpoint": CHECKPOINT,
        "model": "ResNet-18 FER",
    }