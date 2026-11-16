import numpy as np
from src.inference.analyzer import FaceAnalyzer
from src.inference.predictor import CLASSES
from src.inference.tracker import FaceTracker

FRAME_SIZE = 200
LEFT_BOX = (10, 10, 40, 40)
RIGHT_BOX = (120, 10, 40, 40)
LEFT_PIXEL = 50
RIGHT_PIXEL = 150
CONFIDENT_PROBS = np.array([0.9, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
SECOND_PROBS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
UNCERTAIN_PROBS = np.array([0.3, 0.0, 0.0, 0.0, 0.25, 0.25, 0.2])


class FakePredictor:
    class_names = CLASSES

    def __init__(self, probs_by_pixel: dict[int, np.ndarray]) -> None:
        self._probs_by_pixel = probs_by_pixel
        self.next_override: np.ndarray | None = None

    def predict(self, crop: np.ndarray) -> np.ndarray:
        if self.next_override is not None:
            return self.next_override
        return self._probs_by_pixel[int(crop.mean())]


class FakeDetector:
    def __init__(self, boxes: list[tuple[int, int, int, int]]) -> None:
        self.boxes = boxes

    def detectMultiScale(self, gray, scaleFactor, minNeighbors, minSize):
        return np.array(self.boxes)


def make_frame() -> np.ndarray:
    frame = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
    for x, y, w, h in (LEFT_BOX, RIGHT_BOX):
        frame[y : y + h, x : x + w] = LEFT_PIXEL if x == LEFT_BOX[0] else RIGHT_PIXEL
    return frame


def make_analyzer(
    predictor: FakePredictor, detector: FakeDetector, threshold: float = 0.4
) -> FaceAnalyzer:
    return FaceAnalyzer(
        predictor=predictor,
        face_detector=detector,
        tracker=FaceTracker(0.3, 5, 4),
        uncertainty_threshold=threshold,
        scale_factor=1.1,
        min_neighbors=5,
        min_size=[30, 30],
    )


def test_faces_are_not_averaged_together() -> None:
    predictor = FakePredictor({LEFT_PIXEL: CONFIDENT_PROBS, RIGHT_PIXEL: SECOND_PROBS})
    analyzer = make_analyzer(predictor, FakeDetector([LEFT_BOX, RIGHT_BOX]))
    for _ in range(3):
        left, right = analyzer.analyze_frame(make_frame())
    assert left.face_id != right.face_id
    assert np.allclose(left.probs, CONFIDENT_PROBS)
    assert np.allclose(right.probs, SECOND_PROBS)
    assert left.emotion == "angry"
    assert right.emotion == "surprise"


def test_smoothing_averages_one_face_over_frames() -> None:
    predictor = FakePredictor({LEFT_PIXEL: CONFIDENT_PROBS})
    analyzer = make_analyzer(predictor, FakeDetector([LEFT_BOX]))
    analyzer.analyze_frame(make_frame())
    predictor.next_override = SECOND_PROBS
    face = analyzer.analyze_frame(make_frame())[0]
    assert np.allclose(face.probs, (CONFIDENT_PROBS + SECOND_PROBS) / 2)


def test_uncertain_face_gets_secondary_fields() -> None:
    predictor = FakePredictor({LEFT_PIXEL: UNCERTAIN_PROBS})
    analyzer = make_analyzer(predictor, FakeDetector([LEFT_BOX]))
    face = analyzer.analyze_frame(make_frame())[0]
    assert face.is_uncertain
    assert face.emotion == "angry"
    assert face.secondary_emotion in ("neutral", "sad")
    assert face.secondary_confidence == 0.25


def test_confident_face_has_no_secondary_fields() -> None:
    predictor = FakePredictor({LEFT_PIXEL: CONFIDENT_PROBS})
    analyzer = make_analyzer(predictor, FakeDetector([LEFT_BOX]))
    face = analyzer.analyze_frame(make_frame())[0]
    assert not face.is_uncertain
    assert face.secondary_emotion is None
    assert face.secondary_confidence is None
