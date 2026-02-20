import cv2
import numpy as np

from src.inference.predictor import EmotionPredictor
from src.utils.visualization import draw_emotion_bar

HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

EMOTION_COLORS = {
    'angry':    (0,   0,   220),
    'disgust':  (0,   140, 0),
    'fear':     (180, 0,   180),
    'happy':    (0,   200, 200),
    'neutral':  (160, 160, 160),
    'sad':      (200, 100, 0),
    'surprise': (0,   180, 255),
}

# Color used when the model is uncertain (max prob < uncertainty_threshold)
UNCERTAIN_COLOR = (200, 200, 50)   # muted yellow — visually distinct from all emotion colors

UNCERTAINTY_THRESHOLD = 0.40       # can also be driven from config


class WebcamFER:
    def __init__(self, checkpoint_path, smoothing_window=10, scale_factor=1.1,
                 min_neighbors=5, min_size=(30, 30), confidence_threshold=0.3,
                 uncertainty_threshold=UNCERTAINTY_THRESHOLD):
        self.predictor = EmotionPredictor(
            checkpoint_path=checkpoint_path,
            smoothing_window=smoothing_window
        )
        self.face_detector = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold

    def _detect_faces(self, gray_frame):
        return self.face_detector.detectMultiScale(
            gray_frame,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )

    def _build_label(self, probs, class_names):
        """
        Return (label_text, box_color, is_uncertain).

        Uncertain  → max(prob) < uncertainty_threshold
                     Label shows top-2: "NEUTRAL / ANGRY  34% / 28%"
                     Box drawn in UNCERTAIN_COLOR.

        Confident  → single label: "HAPPY  87%"
                     Box drawn in the emotion's own color.
        """
        sorted_idx = np.argsort(probs)[::-1]   # descending
        top_idx   = int(sorted_idx[0])
        top_class = class_names[top_idx]
        top_conf  = float(probs[top_idx])

        is_uncertain = top_conf < self.uncertainty_threshold

        if is_uncertain:
            second_idx   = int(sorted_idx[1])
            second_class = class_names[second_idx]
            second_conf  = float(probs[second_idx])
            label = (
                f'{top_class.upper()} / {second_class.upper()}'
                f'  {top_conf:.0%} / {second_conf:.0%}'
            )
            color = UNCERTAIN_COLOR
        else:
            label = f'{top_class.upper()}  {top_conf:.0%}'
            color = EMOTION_COLORS.get(top_class, (255, 255, 255))

        return label, color, is_uncertain

    def _draw_label(self, frame, label, color, x, y, w, h):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.60
        thickness  = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        label_y = y - 10 if y - 10 > th + 4 else y + h + th + 10

        # Filled background pill for readability
        cv2.rectangle(frame,
                      (x, label_y - th - 4),
                      (x + tw + 8, label_y + baseline),
                      color, -1)
        cv2.putText(frame, label, (x + 4, label_y),
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    def _process_frame(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._detect_faces(gray)
        fh, fw = frame.shape[:2]

        for (x, y, w, h) in faces:
            face_crop = gray[y:y + h, x:x + w]
            probs     = self.predictor.predict_smoothed(face_crop)

            label, color, is_uncertain = self._build_label(probs, self.predictor.class_names)

            # Bounding box — thinner for uncertain, normal for confident
            box_thickness = 1 if is_uncertain else 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, box_thickness)

            self._draw_label(frame, label, color, x, y, w, h)

            # Probability bars — right of face box if room, else left, else top-left
            bar_panel_w = 220
            bar_x = x + w + 10
            if bar_x + bar_panel_w > fw:
                bar_x = x - bar_panel_w - 10
            if bar_x < 0:
                bar_x = 10
            bar_y = max(y, 10)

            draw_emotion_bar(frame, probs, self.predictor.class_names,
                             x=bar_x, y=bar_y, bar_width=140, bar_height=16)

        cv2.putText(frame, f'Faces: {len(faces)}', (10, fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError('Cannot open webcam. Check device connection.')

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f'Webcam FER running at {actual_w}x{actual_h}. '
            f'Uncertainty threshold: {self.uncertainty_threshold:.0%}. '
            f'Press Q to quit, R to reset buffer.'
        )

        cv2.namedWindow('Facial Emotion Recognition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Facial Emotion Recognition', actual_w * 2, actual_h * 2)

        while True:
            ret, frame = cap.read()
            if not ret:
                print('Failed to grab frame.')
                break

            frame = self._process_frame(frame)
            cv2.imshow('Facial Emotion Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.predictor.reset_buffer()
                print('Buffer reset.')

        cap.release()
        cv2.destroyAllWindows()