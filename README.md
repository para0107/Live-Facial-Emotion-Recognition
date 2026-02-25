# 🎭 Facial Emotion Recognition

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18.2+-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-22C55E?style=for-the-badge)

**Real-time facial emotion recognition from webcam using a fine-tuned ResNet-18.**  
7 universal emotion classes · Temporal smoothing · Uncertainty visualization · WebSocket inference · Deployable web app.

[Overview](#-overview) · [Architecture](#-architecture) · [Setup](#-setup) · [Training](#-training) · [Inference](#-real-time-inference) · [Web App](#-web-app) · [Results](#-results) · [References](#-references)

</div>

---

## 📖 Overview

This project implements a complete end-to-end pipeline for **real-time Facial Emotion Recognition (FER)** , from raw dataset to live webcam inference and a deployable web application. Every component is built from scratch: data loading and augmentation, transfer learning fine-tuning, loss function design for class imbalance, a live inference loop with temporal smoothing, and a React + FastAPI web app with WebSocket streaming.

**What this is not:** a wrapper around a cloud API. Every design decision, which layers to freeze, how to handle the `disgust` class having 16× fewer samples than `happy`, why temporal smoothing over 10 frames matters, when to show uncertainty labels, is implemented and justified from first principles, grounded in the academic literature.

### Recognized Emotions

| Label | Description |
|-------|-------------|
| 😠 `angry` | Raised inner brows, lip corners pulled down |
| 🤢 `disgust` | Nose wrinkle, upper lip raise |
| 😨 `fear` | Wide eyes, raised upper lip |
| 😊 `happy` | Lip corner pull, cheek raise |
| 😐 `neutral` | No dominant muscle activation |
| 😢 `sad` | Inner brow raise, lip corner depression |
| 😲 `surprise` | Wide eyes, dropped jaw |

---

## 🏗 Architecture

The pipeline follows a modular, sequential design:

```
┌─────────────────────────────────────────────────────────────────┐
│                        WEBCAM FRAME (BGR)                        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │      HAAR CASCADE DETECTOR      │
              │   (Viola-Jones face localizer)  │
              │   Output: (x, y, w, h) boxes    │
              └────────────────┬───────────────┘
                               │  crop per face
                               ▼
              ┌────────────────────────────────┐
              │         PREPROCESSING           │
              │  Grayscale → Resize 48×48       │
              │  Normalize μ=0.507, σ=0.255     │
              └────────────────┬───────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                      ResNet-18 BACKBONE                           │
│                                                                    │
│   Conv1(1→64, 7×7, s=2) ──► BN ──► ReLU                         │
│         │                                                          │
│   MaxPool2d(kernel=2, stride=1, padding=0)                       │
│         │                                                          │
│       Layer1 (2× BasicBlock, 64ch)                                │
│         │                                                          │
│       Layer2 (2× BasicBlock, 128ch)                               │
│         │                                                          │
│       Layer3 (2× BasicBlock, 256ch)                               │
│         │                                                          │
│       Layer4 (2× BasicBlock, 512ch) ◄── Grad-CAM hook            │
│         │                                                          │
│   AdaptiveAvgPool(1×1) → Flatten → 512-dim vector                │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │       CLASSIFICATION HEAD       │
              │   Dropout(0.5)                  │
              │   Linear(512 → 256)             │
              │   BatchNorm1d → ReLU            │
              │   Dropout(0.25)                 │
              │   Linear(256 → 7)               │
              │   [raw logits ]     │
              └────────────────┬───────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │      TEMPORAL SMOOTHER          │
              │   Sliding window N=10 frames    │
              │   Mean of softmax probability   │
              │   vectors over window           │
              └────────────────┬───────────────┘
                               │
                               ▼
          ┌───────────────────────────────────────────────┐
          │  UNCERTAINTY CHECK  (threshold = 0.40)         │
          │  top_conf < 0.40 → show top-2 emotions         │
          │  e.g.  "NEUTRAL / ANGRY  34% / 28%"            │
          │  Bounding box rendered in muted yellow         │
          └───────────────────────────────────────────────┘
                               │
                               ▼
          ┌───────────────────────────────────────┐
          │  OVERLAY: label · confidence · bars    │
          └───────────────────────────────────────┘
```

### Key Design Choices

**Grayscale single-channel input.** FER2013 is grayscale by nature. Processing in grayscale halves memory, speeds training, and avoids the model learning spurious color correlations. The first `Conv1` layer is adapted to accept 1-channel input by averaging the pretrained ImageNet weights across the 3 input channels (`new_conv.weight = old_conv.weight.mean(dim=1, keepdim=True)`), which preserves the magnitude of learned features.

**Modified MaxPool.** The standard ResNet-18 uses a 3×3 maxpool with stride 2 after `conv1`, which aggressively downsamples early feature maps. For a 48×48 input this reduces spatial resolution too aggressively. `maxpool` is replaced with `MaxPool2d(kernel_size=2, stride=1, padding=0)` , a softer spatial reduction that retains more low-level detail. **Important:** the saved checkpoint `best_model.pth` was trained with this exact configuration. Do not swap it for `nn.Identity()` without retraining.

**Staged unfreezing.** Backbone frozen for the first 5 epochs while only the head trains. Then the full network unfreezes with CosineAnnealingLR. This prevents the large early gradients from the randomly-initialized head from destroying pretrained ImageNet features.

**Label smoothing + class weights.** FER2013 is severely imbalanced (`disgust`: 436 samples vs `happy`: 7,215). Hard one-hot labels combined with this imbalance cause the model to ignore minority classes. Label smoothing (ε=0.1) distributes probability mass across non-target classes, and per-class weights inversely proportional to class frequency are folded into the loss.

**Uncertainty visualization.** The model is a closed-set classifier , it always outputs a distribution over 7 classes even when the input is ambiguous. When `max(softmax) < 0.40`, the top-2 emotions are shown together (e.g. `NEUTRAL / ANGRY  34% / 28%`) and the bounding box turns muted yellow. This is a direct response to the known neutral/angry confusion in FER2013 and reflects the literature on closed-set classifier limitations.

---

## 📁 Project Structure

```
Live-FER/
│
├── configs/
│   └── config.yaml              # Single source of truth for all hyperparameters
│
├── data/
│   └── fer2013/                 # Dataset root (not tracked by git)
│       ├── train/
│       │   ├── angry/           # ~3,995 images
│       │   ├── disgust/         # ~436 images  ← heavily underrepresented
│       │   ├── fear/            # ~4,097 images
│       │   ├── happy/           # ~7,215 images ← dominant class
│       │   ├── neutral/         # ~4,965 images
│       │   ├── sad/             # ~4,830 images
│       │   └── surprise/        # ~3,171 images
│       └── test/
│           └── (same structure)
│
├── src/
│   ├── data/
│   │   ├── dataset.py           # FER2013Dataset + automatic class weight computation
│   │   └── transforms.py        # Train / val / inference transform pipelines
│   │
│   ├── model/
│   │   ├── resnet_fer.py        # ResNet-18 adapted for 1-ch, 48×48 FER
│   │   └── cam.py               # Grad-CAM extractor + heatmap overlay
│   │
│   ├── training/
│   │   ├── trainer.py           # Training loop, TensorBoard, early stopping
│   │   └── losses.py            # WeightedCrossEntropy + LabelSmoothingLoss
│   │
│   ├── inference/
│   │   ├── predictor.py         # Single-image predictor + sliding-window smoother
│   │   └── webcam.py            # Live webcam loop with Haar Cascade + overlay
│   │
│   └── utils/
│       ├── metrics.py           # AverageMeter, accuracy(), evaluate_model()
│       └── visualization.py     # Confusion matrix plot, per-class bar overlay
│
├── scripts/
│   ├── train.py                 # Training entry point
│   ├── evaluate.py              # Test-set evaluation + confusion matrix export
│   └── run_webcam.py            # Local live inference entry point
│
├── webapp/
│   ├── backend/
│   │   ├── app.py               # FastAPI WebSocket server (Hugging Face Spaces)
│   │   └── main.py              # FastAPI WebSocket server (Railway / local)
│   └── frontend/
│       ├── src/
│       │   ├── App.js           # React app , webcam capture, WebSocket client, overlay
│       │   └── index.js
│       ├── .env                 # ws://localhost:8000/ws
│       ├── .env.production      # wss://para0107-live-fer-backend.hf.space/ws
│       ├── package.json
│       └── vercel.json          # Vercel deployment config
│
├── checkpoints/                 # Saved .pth files (not tracked)
├── logs/                        # TensorBoard runs + exported figures
├── Dockerfile                   # For Hugging Face Spaces (port 7860)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1 , Environment

```bash
git clone https://github.com/para0107/Live-Facial-Emotion-Recognition
cd Live-FER

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2 , GPU check (recommended)

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

Training on CPU is possible but will take significantly longer (~8–12 hours vs ~45 minutes on a mid-range GPU).

---

## 📦 Dataset

Download **FER2013** from Kaggle:
🔗 [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

Place the extracted `train/` and `test/` folders inside `data/fer2013/`.

### Dataset statistics

| Class     | Train  | Test  | % of Train | Auto Weight |
|-----------|--------|-------|------------|-------------|
| angry     | 3,995  | 958   | 13.9%      | 1.74        |
| disgust   | 436    | 111   | 1.5%       | **15.94**   |
| fear      | 4,097  | 1,024 | 14.3%      | 1.70        |
| happy     | 7,215  | 1,774 | 25.1%      | 0.97        |
| neutral   | 4,965  | 1,233 | 17.3%      | 1.40        |
| sad       | 4,830  | 1,247 | 16.8%      | 1.44        |
| surprise  | 3,171  | 831   | 11.0%      | 2.19        |
| **Total** | **28,709** | **7,178** | , | , |

Class weights are computed automatically in `dataset.py`: `w_c = N_total / (C × N_c)`

---

## 🏋️ Training

```bash
python scripts/train.py
```

All hyperparameters live in `configs/config.yaml`.

### Hyperparameter reference

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Backbone | ResNet-18 | Good capacity/size ratio for 48×48 images |
| Pretrained | ImageNet | Exploits general visual feature hierarchy |
| Input channels | 1 | FER2013 is grayscale |
| Input size | 48 × 48 | Native FER2013 resolution |
| Freeze epochs | 5 | Head stabilizes before backbone unfreezes |
| Epochs | 50 | With early stopping patience=10 |
| Batch size | 64 | Stable gradients, fits ~4 GB VRAM |
| Learning rate | 2e-4 | Conservative; preserves pretrained weights |
| Weight decay | 1e-4 | L2 regularization |
| Optimizer | Adam | Adaptive LR, robust to sparse gradients |
| Scheduler | CosineAnnealingLR | Smooth decay to eta_min=1e-6 |
| Dropout | 0.5 / 0.25 | Two-stage in classifier head |
| Label smoothing | 0.1 | Prevents overconfident predictions |
| Random erasing | p=0.3 | Forces holistic facial feature use |
| Early stopping | patience=10 | |
| Uncertainty threshold | 0.40 | Below this → show top-2 emotions |

### Monitor with TensorBoard

```bash
tensorboard --logdir logs/
# Open http://localhost:6006
```

### Staged training

```
Epochs 1–5:   Backbone FROZEN  →  head-only training
              Large gradients from random init stay contained

Epoch 6+:     Full network UNFROZEN
              Fine-tunes with low LR decaying via cosine schedule
```

---

## 📊 Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
```

Outputs per-class precision, recall, F1 and saves `logs/confusion_matrix.png`.

**Most common confusions on FER2013:**
- `fear` ↔ `sad` (both involve downturned features)
- `disgust` ↔ `angry` (both involve brow lowering)
- `neutral` ↔ `angry` (subtle activation; motivates the uncertainty display)
- `surprise` ↔ `fear` (both involve widened eyes)

These reflect genuine perceptual ambiguity , human accuracy on FER2013 is estimated at ~65%.

---

## 🎥 Real-Time Inference (Local Webcam)

```bash
python scripts/run_webcam.py --checkpoint checkpoints/best_model.pth
```

### Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Reset temporal smoothing buffer |

### Display elements

- Bounding box colored by dominant emotion (muted yellow when uncertain)
- Label + confidence above the box; dual label when `confidence < 0.40`
- 7-class probability bar chart positioned beside the face box
- Face count bottom-left

### Temporal smoothing

Raw per-frame softmax vectors are averaged over a sliding window:

```
p̄_t = (1/N) Σ_{i=0}^{N-1} p_{t-i}     N = 10
```

Eliminates jitter from micro-expressions and brief detection instabilities without learnable parameters. At 30 fps this introduces ~333 ms of latency.

---

## 🌐 Web App

A full-stack web application is included in `webapp/`, allowing browser-based real-time inference over a WebSocket connection.

### Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18, Canvas API |
| Backend | FastAPI, WebSocket |
| Deployment (backend) | Hugging Face Spaces (port 7860) |
| Deployment (frontend) | Vercel |

### How it works

1. The React frontend captures webcam frames at 640×480 via `getUserMedia`
2. Frames are JPEG-encoded (quality 0.6) and sent over WebSocket as base64
3. A **ping-pong lock** prevents frame flooding: the next frame is only sent after the server has replied, naturally adapting to server load
4. The FastAPI backend runs Haar Cascade detection and `EmotionPredictor.predict_smoothed()` on each frame
5. Results (bounding boxes, per-class probabilities, uncertainty flag) are returned as JSON
6. The frontend renders bounding boxes and probability bars on a `<canvas>` overlay

### Running locally

**Backend:**
```bash
cd webapp/backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd webapp/frontend
npm install
npm start
# Connects to ws://localhost:8000/ws by default
```

### Uncertainty in the web app

The backend applies the same `uncertainty_threshold = 0.40` as the local webcam script. When `top_conf < 0.40`, the response includes both `emotion` + `secondary_emotion` and sets `is_uncertain: true`. The frontend renders a muted yellow box and a split label such as `NEUTRAL/ANGRY 34%/28%`.

---

## 🔬 Grad-CAM Visualization

```python
from src.model.cam import CAMExtractor
from src.model.resnet_fer import ResNetFER

model = ResNetFER(pretrained=False)
# load checkpoint...

extractor = CAMExtractor(model)
cam, predicted_class = extractor.generate_cam(input_tensor, target_class=3)  # 3 = happy
overlay = extractor.overlay_cam(original_image, cam)
```

The hook is registered on `model.features.layer4`. Expected behavior: the model attends to mouth/cheeks for `happy`, brow region for `angry`, eye region for `fear`/`surprise`. Diffuse or non-facial attention maps indicate the model is learning dataset artifacts.

---

## 📈 Results

| Configuration | Test Accuracy | Notes |
|---------------|---------------|-------|
| Random baseline | 14.3% | Uniform over 7 classes |
| ResNet-18, no pretrain, scratch | ~52% | Overfits quickly |
| ResNet-18, pretrained, full fine-tune | ~65% | Near human-level |
| + Label smoothing ε=0.1 | ~67% | Better calibration |
| + Class-weighted loss | ~67–68% | Disgust F1 improves significantly |
| + Random erasing p=0.3 | ~68% | Holistic features |
| **+ Staged freeze/unfreeze (final)** | **68.95%** | **Saved checkpoint** |

*Human accuracy on FER2013 ≈ 65%. Results vary across random seeds.*

---

## 📚 References

1. **Zhang et al. (2024).** *Open-Set Facial Expression Recognition.* AAAI 2024. `arXiv:2401.12507`
2. **Schroff et al. (2015).** *FaceNet: A Unified Embedding for Face Recognition and Clustering.* CVPR 2015. `arXiv:1503.03832`
3. **Dewi et al. (2024).** *Real-Time Facial Expression Recognition: Advances, Challenges, and Future Directions.* Vietnam Journal of Computer Science.
4. **He et al. (2016).** *Deep Residual Learning for Image Recognition.* CVPR 2016.
5. **Goodfellow et al. (2013).** *Challenges in Representation Learning* (FER2013 dataset).
6. **Selvaraju et al. (2017).** *Grad-CAM: Visual Explanations from Deep Networks.* ICCV 2017.
7. **Viola & Jones (2001).** *Rapid Object Detection using a Boosted Cascade of Simple Features.* CVPR 2001.

---

<div align="center">
Built as a Bachelor's thesis project in Computer Science.<br/>
Grounded in peer-reviewed FER literature · Every design decision justified.
</div>
