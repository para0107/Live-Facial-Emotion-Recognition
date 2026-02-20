import base64
import io
import os
import sys
import json
import yaml
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
CONFIG_PATH = os.path.join(FER_ROOT, 'configs', 'config.yaml')

app = FastAPI(title="FER WebSocket API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load configuration (Same idea as run_webcam.py)
print(f"Loading config from: {CONFIG_PATH}")
try:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    inf_cfg = config['inference']
except FileNotFoundError:
    print(f"Warning: {CONFIG_PATH} not found. Falling back to default values.")
    inf_cfg = {
        'smoothing_window': 10,
        'face_scale_factor': 1.1,
        'face_min_neighbors': 5,
        'face_min_size': [30, 30],
        'uncertainty_threshold': 0.40
    }

UNCERTAINTY_THRESHOLD = inf_cfg.get('uncertainty_threshold', 0.40)

print(f"Loading checkpoint from: {CHECKPOINT}")
if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

# 2. Use dynamic smoothing window
predictor = EmotionPredictor(
    checkpoint_path=CHECKPOINT,
    smoothing_window=inf_cfg['smoothing_window'],
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

            # 3. Use dynamic face detection parameters
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=inf_cfg['face_scale_factor'],
                minNeighbors=inf_cfg['face_min_neighbors'],
                minSize=tuple(inf_cfg['face_min_size']),
            )

            results = []
            for (x, y, w, h) in faces:
                face_crop = gray[y:y + h, x:x + w]
                probs = predictor.predict_smoothed(face_crop)

                # 4. Apply Uncertainty Thresholding (Same idea as webcam.py)
                sorted_idx = np.argsort(probs)[::-1]
                top_idx = int(sorted_idx[0])
                top_class = predictor.class_names[top_idx]
                top_conf = round(float(probs[top_idx]), 4)

                is_uncertain = top_conf < UNCERTAINTY_THRESHOLD

                second_class = None
                second_conf = None

                if is_uncertain:
                    second_idx = int(sorted_idx[1])
                    second_class = predictor.class_names[second_idx]
                    second_conf = round(float(probs[second_idx]), 4)

                results.append({
                    "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "probs": {
                        predictor.class_names[i]: round(float(probs[i]), 4)
                        for i in range(len(predictor.class_names))
                    },
                    "emotion": top_class,
                    "confidence": top_conf,
                    "is_uncertain": is_uncertain,
                    "secondary_emotion": second_class,
                    "secondary_confidence": second_conf
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
        "smoothing_window": inf_cfg['smoothing_window'],
        "uncertainty_threshold": UNCERTAINTY_THRESHOLD
    }