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

from src.inference.analyzer import setup_face_analyzer

CHECKPOINT = os.environ.get(
    "CHECKPOINT_PATH",
    os.path.join(FER_ROOT, "checkpoints", "best_model.pth")
)
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
        'uncertainty_threshold': 0.40,
        'face_iou_threshold': 0.3,
        'face_max_missed_frames': 5
    }

UNCERTAINTY_THRESHOLD = inf_cfg.get('uncertainty_threshold', 0.40)

print(f"Loading checkpoint from: {CHECKPOINT}")
if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

# 2. Setup shared face analyzer
analyzer = setup_face_analyzer(CHECKPOINT, inf_cfg)
print(f"Model loaded on {analyzer.device}")


def decode_frame(b64: str) -> np.ndarray:
    data = base64.b64decode(b64.split(",")[-1])
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    analyzer.reset()

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            frame_b64 = data.get("frame")
            if not frame_b64:
                continue

            frame = decode_frame(frame_b64)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            results = []
            for face in analyzer.analyze_frame(gray):
                x, y, w, h = face.box
                results.append({
                    "face_id": face.face_id,
                    "box": {"x": x, "y": y, "w": w, "h": h},
                    "probs": {
                        name: round(float(face.probs[i]), 4)
                        for i, name in enumerate(analyzer.class_names)
                    },
                    "emotion": face.emotion,
                    "confidence": face.confidence,
                    "is_uncertain": face.is_uncertain,
                    "secondary_emotion": face.secondary_emotion,
                    "secondary_confidence": face.secondary_confidence
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
        "device": str(analyzer.device),
        "checkpoint": CHECKPOINT,
        "model": "ResNet-18 FER",
        "smoothing_window": inf_cfg['smoothing_window'],
        "uncertainty_threshold": UNCERTAINTY_THRESHOLD
    }