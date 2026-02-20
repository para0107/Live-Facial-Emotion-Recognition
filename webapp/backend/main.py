import base64
import io
import os
import sys
import json
import asyncio
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from collections import deque

# ── Path setup ────────────────────────────────────────────────────────────────
# __file__ = Live-FER/webapp/backend/main.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FER_ROOT = os.environ.get("FER_ROOT", PROJECT_ROOT)
sys.path.insert(0, FER_ROOT)

from src.model.resnet_fer import ResNetFER
from src.data.transforms import get_inference_transforms

# ── Constants ─────────────────────────────────────────────────────────────────
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
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

# ── Model (loaded once at startup) ────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetFER(num_classes=7, pretrained=False)
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()
print(f"Model loaded on {device}")

transform = get_inference_transforms(image_size=48)
face_detector = cv2.CascadeClassifier(HAAR)


def predict_face(face_gray_crop: np.ndarray) -> list[float]:
    img = Image.fromarray(face_gray_crop).convert("L")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs.tolist()


def decode_frame(b64: str) -> np.ndarray:
    data = base64.b64decode(b64.split(",")[-1])
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    smoothing_buffer = deque(maxlen=8)

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            frame_b64 = data.get("frame")
            if not frame_b64:
                continue

            # Decode frame
            frame = decode_frame(frame_b64)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            results = []
            for (x, y, w, h) in faces:
                crop = gray[y:y+h, x:x+w]
                probs = predict_face(crop)
                smoothing_buffer.append(probs)
                smoothed = np.mean(list(smoothing_buffer), axis=0).tolist()
                top_idx = int(np.argmax(smoothed))
                results.append({
                    "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "probs": {CLASSES[i]: round(smoothed[i], 4) for i in range(7)},
                    "emotion": CLASSES[top_idx],
                    "confidence": round(smoothed[top_idx], 4),
                })

            await ws.send_text(json.dumps({
                "faces": results,
                "face_count": len(results),
            }))

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        await ws.close()


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "model": "ResNet-18 FER"}
