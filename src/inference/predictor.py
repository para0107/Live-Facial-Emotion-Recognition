import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from PIL import Image

from src.data.transforms import get_inference_transforms
from src.model.resnet_fer import ResNetFER

CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


class EmotionPredictor:
    def __init__(self, checkpoint_path, device=None, smoothing_window=10):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = get_inference_transforms(image_size=48)
        self.smoothing_window = smoothing_window
        self._prob_buffer = deque(maxlen=smoothing_window)

        self.model = ResNetFER(num_classes=len(CLASSES), pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, face_crop):
        if isinstance(face_crop, np.ndarray):
            face_crop = Image.fromarray(face_crop).convert('L')

        tensor = self.transform(face_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        return probs

    def predict_smoothed(self, face_crop):
        probs = self.predict(face_crop)
        self._prob_buffer.append(probs)
        smoothed = np.mean(self._prob_buffer, axis=0)
        return smoothed

    def reset_buffer(self):
        self._prob_buffer.clear()

    @property
    def class_names(self):
        return CLASSES
