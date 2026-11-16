from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

Box = tuple[int, int, int, int]


@dataclass
class TrackedFace:
    face_id: int
    box: Box
    probs_history: deque = field(default_factory=deque)
    missed_frames: int = 0


def compute_iou(box_a: Box, box_b: Box) -> float:
    """Intersection over union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    inter_w = min(ax + aw, bx + bw) - max(ax, bx)
    inter_h = min(ay + ah, by + bh) - max(ay, by)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    return intersection / union


class FaceTracker:
    def __init__(
        self, iou_threshold: float, max_missed_frames: int, smoothing_window: int
    ) -> None:
        self._iou_threshold = iou_threshold
        self._max_missed_frames = max_missed_frames
        self._smoothing_window = smoothing_window
        self._tracks: list[TrackedFace] = []
        self._next_face_id = 0

    def update(self, boxes: Sequence[Sequence[int]]) -> list[TrackedFace]:
        """Match boxes to existing tracks by IoU; return tracks in box order."""
        new_boxes: list[Box] = [tuple(int(v) for v in box) for box in boxes]
        pairs = sorted(
            (
                (compute_iou(track.box, box), track_idx, box_idx)
                for track_idx, track in enumerate(self._tracks)
                for box_idx, box in enumerate(new_boxes)
            ),
            reverse=True,
        )
        matched_by_box: dict[int, TrackedFace] = {}
        matched_track_idxs: set[int] = set()
        for iou, track_idx, box_idx in pairs:
            if iou < self._iou_threshold:
                break
            if track_idx in matched_track_idxs or box_idx in matched_by_box:
                continue
            matched_track_idxs.add(track_idx)
            matched_by_box[box_idx] = self._tracks[track_idx]

        ordered: list[TrackedFace] = []
        for box_idx, box in enumerate(new_boxes):
            track = matched_by_box.get(box_idx)
            if track is None:
                track = TrackedFace(
                    face_id=self._next_face_id,
                    box=box,
                    probs_history=deque(maxlen=self._smoothing_window),
                )
                self._next_face_id += 1
            track.box = box
            track.missed_frames = 0
            ordered.append(track)

        kept_old: list[TrackedFace] = []
        for track_idx, track in enumerate(self._tracks):
            if track_idx in matched_track_idxs:
                continue
            track.missed_frames += 1
            if track.missed_frames <= self._max_missed_frames:
                kept_old.append(track)

        self._tracks = ordered + kept_old
        return ordered

    def reset(self) -> None:
        self._tracks = []
        self._next_face_id = 0
