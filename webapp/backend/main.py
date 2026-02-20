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

# ── Path setup ────────────────────────────────────────────────────────────────
# __file__ = Live-FER/webapp/backend/main.py  →  3x dirname = Live-FER/
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
FER_ROOT = os.environ.get("FER_ROOT", PROJECT_ROOT)
sys.path.insert(0, FER_ROOT)

from src.inference.predictor import EmotionPredictor

# ── Constants ─────────────────────────────────────────────────────────────────
CHECKPOINT = os.environ.get(
    "CHECKPOINT_PATH",
    os.path.join(FER_ROOT, "checkpoints", "best_model.pth")
)
HAAR = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="FER WebSocket API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model ─────────────────────────────────────────────────────────────────────
print(f"Loading checkpoint from: {CHECKPOINT}")
if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

# Identical instantiation to webcam.py:
#   WebcamFER.__init__ → EmotionPredictor(checkpoint_path, smoothing_window=10)
predictor = EmotionPredictor(
    checkpoint_path=CHECKPOINT,
    smoothing_window=10,
)
print(f"Model loaded on {predictor.device}")

face_detector = cv2.CascadeClassifier(HAAR)


def decode_frame(b64: str) -> np.ndarray:
    """Decode base64 JPEG from browser canvas into a BGR numpy array."""
    data = base64.b64decode(b64.split(",")[-1])
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ── WebSocket endpoint ────────────────────────────────────────────────────────
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

            # ── Identical to WebcamFER._process_frame ─────────────────────────
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
                # Raw grayscale crop — NO extra preprocessing.
                # predictor.predict() handles everything internally:
                #   numpy → PIL.L → Resize(48) → ToTensor → Normalize(0.507, 0.255)
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