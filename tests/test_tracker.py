from src.inference.tracker import FaceTracker, compute_iou

BOX_A = (10, 10, 50, 50)
BOX_A_MOVED = (12, 11, 50, 50)
BOX_B = (200, 100, 50, 50)
MAX_MISSED = 2


def make_tracker() -> FaceTracker:
    return FaceTracker(
        iou_threshold=0.3, max_missed_frames=MAX_MISSED, smoothing_window=5
    )


def test_iou_of_disjoint_and_identical_boxes() -> None:
    assert compute_iou(BOX_A, BOX_B) == 0.0
    assert compute_iou(BOX_A, BOX_A) == 1.0


def test_moving_box_keeps_id() -> None:
    tracker = make_tracker()
    first = tracker.update([BOX_A])[0]
    second = tracker.update([BOX_A_MOVED])[0]
    assert first.face_id == second.face_id == 0
    assert second.box == BOX_A_MOVED


def test_two_faces_keep_ids_when_order_swaps() -> None:
    tracker = make_tracker()
    a, b = tracker.update([BOX_A, BOX_B])
    swapped_b, swapped_a = tracker.update([BOX_B, BOX_A_MOVED])
    assert a.face_id != b.face_id
    assert swapped_a.face_id == a.face_id
    assert swapped_b.face_id == b.face_id


def test_face_missing_too_long_gets_new_id() -> None:
    tracker = make_tracker()
    first_id = tracker.update([BOX_A])[0].face_id
    for _ in range(MAX_MISSED + 1):
        tracker.update([])
    assert tracker.update([BOX_A])[0].face_id != first_id


def test_face_missing_briefly_keeps_id() -> None:
    tracker = make_tracker()
    first_id = tracker.update([BOX_A])[0].face_id
    for _ in range(MAX_MISSED):
        tracker.update([])
    assert tracker.update([BOX_A])[0].face_id == first_id
