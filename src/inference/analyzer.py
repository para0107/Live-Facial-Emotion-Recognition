from dataclasses import dataclass

import cv2
import numpy as np

from src.inference.predictor import EmotionPredictor
from src.inference.tracker import Box, FaceTracker

HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CONFIDENCE_DECIMALS = 4


@dataclass
class FaceAnalysis:
    face_id: int
    box: Box
    probs: np.ndarray
    emotion: str
    confidence: float
    is_uncertain: bool
    secondary_emotion: str | None
    secondary_confidence: float | None


class FaceAnalyzer:
    def __init__(
        self,
        predictor: EmotionPredictor,
        face_detector: cv2.CascadeClassifier,
        tracker: FaceTracker,
        uncertainty_threshold: float,
        scale_factor: float,
        min_neighbors: int,
        min_size: list[int],
    ) -> None:
        self._predictor = predictor
        self._face_detector = face_detector
        self._tracker = tracker
        self._uncertainty_threshold = uncertainty_threshold
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = min_size

    @property
    def class_names(self) -> list[str]:
        return self._predictor.class_names

    @property
    def device(self) -> str:
        return self._predictor.device

    def analyze_frame(self, gray_frame: np.ndarray) -> list[FaceAnalysis]:
        boxes = self._face_detector.detectMultiScale(
            gray_frame,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=tuple(self._min_size),
        )
        tracks = self._tracker.update(boxes)

        analyses: list[FaceAnalysis] = []
        for track in tracks:
            x, y, w, h = track.box
            track.probs_history.append(
                self._predictor.predict(gray_frame[y : y + h, x : x + w])
            )
            probs = np.mean(track.probs_history, axis=0)

            sorted_idx = np.argsort(probs)[::-1]
            top_idx = int(sorted_idx[0])
            top_conf = round(float(probs[top_idx]), CONFIDENCE_DECIMALS)
            is_uncertain = top_conf < self._uncertainty_threshold

            secondary_emotion = None
            secondary_confidence = None
            if is_uncertain:
                second_idx = int(sorted_idx[1])
                secondary_emotion = self.class_names[second_idx]
                secondary_confidence = round(
                    float(probs[second_idx]), CONFIDENCE_DECIMALS
                )

            analyses.append(
                FaceAnalysis(
                    face_id=track.face_id,
                    box=track.box,
                    probs=probs,
                    emotion=self.class_names[top_idx],
                    confidence=top_conf,
                    is_uncertain=is_uncertain,
                    secondary_emotion=secondary_emotion,
                    secondary_confidence=secondary_confidence,
                )
            )
        return analyses

    def reset(self) -> None:
        self._tracker.reset()


def setup_face_analyzer(checkpoint_path: str, inference_config: dict) -> FaceAnalyzer:
    """Build the predictor, Haar detector and tracker from the inference config."""
    return FaceAnalyzer(
        predictor=EmotionPredictor(checkpoint_path=checkpoint_path),
        face_detector=cv2.CascadeClassifier(HAAR_CASCADE_PATH),
        tracker=FaceTracker(
            inference_config["face_iou_threshold"],
            inference_config["face_max_missed_frames"],
            inference_config["smoothing_window"],
        ),
        uncertainty_threshold=inference_config["uncertainty_threshold"],
        scale_factor=inference_config["face_scale_factor"],
        min_neighbors=inference_config["face_min_neighbors"],
        min_size=inference_config["face_min_size"],
    )
