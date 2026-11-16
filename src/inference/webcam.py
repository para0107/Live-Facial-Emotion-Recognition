import cv2

from src.inference.analyzer import setup_face_analyzer
from src.utils.visualization import draw_emotion_bar

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


class WebcamFER:
    def __init__(self, checkpoint_path, inference_config):
        self.analyzer = setup_face_analyzer(checkpoint_path, inference_config)
        self.uncertainty_threshold = inference_config['uncertainty_threshold']

    def _build_label(self, face):
        """
        Return (label_text, box_color) for a FaceAnalysis.

        Uncertain  → Label shows top-2: "NEUTRAL / ANGRY  34% / 28%"
                     Box drawn in UNCERTAIN_COLOR.

        Confident  → single label: "HAPPY  87%"
                     Box drawn in the emotion's own color.
        """
        if face.is_uncertain:
            label = (
                f'{face.emotion.upper()} / {face.secondary_emotion.upper()}'
                f'  {face.confidence:.0%} / {face.secondary_confidence:.0%}'
            )
            color = UNCERTAIN_COLOR
        else:
            label = f'{face.emotion.upper()}  {face.confidence:.0%}'
            color = EMOTION_COLORS.get(face.emotion, (255, 255, 255))

        return label, color

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
        faces = self.analyzer.analyze_frame(gray)
        fh, fw = frame.shape[:2]

        for face in faces:
            x, y, w, h = face.box
            label, color = self._build_label(face)

            # Bounding box — thinner for uncertain, normal for confident
            box_thickness = 1 if face.is_uncertain else 2
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

            draw_emotion_bar(frame, face.probs, self.analyzer.class_names,
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
                self.analyzer.reset()
                print('Buffer reset.')

        cap.release()
        cv2.destroyAllWindows()