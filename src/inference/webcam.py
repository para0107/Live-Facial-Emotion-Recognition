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


class WebcamFER:
    def __init__(self, checkpoint_path, smoothing_window=10, scale_factor=1.1,
                 min_neighbors=5, min_size=(30, 30), confidence_threshold=0.3):
        self.predictor = EmotionPredictor(
            checkpoint_path=checkpoint_path,
            smoothing_window=smoothing_window
        )
        self.face_detector = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.confidence_threshold = confidence_threshold

    def _detect_faces(self, gray_frame):
        return self.face_detector.detectMultiScale(
            gray_frame,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )

    def _process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._detect_faces(gray)

        for (x, y, w, h) in faces:
            face_crop = gray[y:y + h, x:x + w]
            probs = self.predictor.predict_smoothed(face_crop)
            top_idx = int(np.argmax(probs))
            top_class = self.predictor.class_names[top_idx]
            top_conf = probs[top_idx]

            color = EMOTION_COLORS.get(top_class, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            label = f'{top_class} {top_conf:.0%}'
            label_y = y - 10 if y - 10 > 10 else y + h + 20
            cv2.putText(frame, label, (x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            draw_emotion_bar(frame, probs, self.predictor.class_names,
                             x=frame.shape[1] - 230, y=y)

        cv2.putText(frame, f'Faces: {len(faces)}', (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError('Cannot open webcam. Check device connection.')

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print('Webcam FER running. Press Q to quit.')

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

        cap.release()
        cv2.destroyAllWindows()
