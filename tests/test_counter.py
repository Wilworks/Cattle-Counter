"""
test_counter.py - Unit tests for the counting logic.

Tests the CattleCounter's virtual line crossing detection,
anti-double-count protection, and direction gating.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import supervision as sv
from src.counting.counter import CattleCounter


def make_detections(tracker_ids, bboxes):
    """Helper to create supervision Detections with tracker IDs."""
    xyxy = np.array(bboxes, dtype=np.float32)
    detections = sv.Detections(xyxy=xyxy)
    detections.tracker_id = np.array(tracker_ids, dtype=int)
    detections.confidence = np.ones(len(tracker_ids), dtype=np.float32)
    return detections


def test_basic_counting():
    """Test basic single cattle crossing."""
    counter = CattleCounter(line_y=0.5, direction="top_to_bottom", min_crossing_distance=0)
    frame_h = 100

    # Cattle above line (y=20)
    det = make_detections([1], [[40, 15, 60, 25]])
    result = counter.update(det, frame_h)
    assert result["total_count"] == 0, "Should not count before crossing"

    # Cattle crosses line (y=55)
    det = make_detections([1], [[40, 50, 60, 60]])
    result = counter.update(det, frame_h)
    assert result["total_count"] == 1, "Should count after crossing"
    print("[PASS] test_basic_counting")


def test_no_double_count():
    """Test that the same ID is never counted twice (ID Lock)."""
    counter = CattleCounter(line_y=0.5, direction="top_to_bottom", min_crossing_distance=0)
    frame_h = 100

    # First crossing
    det = make_detections([1], [[40, 15, 60, 25]])
    counter.update(det, frame_h)
    det = make_detections([1], [[40, 50, 60, 60]])
    counter.update(det, frame_h)
    assert counter.total_count == 1

    # Step back above line
    det = make_detections([1], [[40, 15, 60, 25]])
    counter.update(det, frame_h)

    # Cross again
    det = make_detections([1], [[40, 50, 60, 60]])
    result = counter.update(det, frame_h)
    assert result["total_count"] == 1, "Should NOT double-count"
    print("[PASS] test_no_double_count")


def test_direction_gate():
    """Test that only exit-direction crossings are counted."""
    counter = CattleCounter(line_y=0.5, direction="top_to_bottom", min_crossing_distance=0)
    frame_h = 100

    # Cattle starts below line (wrong direction)
    det = make_detections([1], [[40, 55, 60, 65]])
    counter.update(det, frame_h)

    # Moves above line (bottom_to_top - wrong direction)
    det = make_detections([1], [[40, 15, 60, 25]])
    result = counter.update(det, frame_h)
    assert result["total_count"] == 0, "Should not count wrong direction"
    print("[PASS] test_direction_gate")


def test_multiple_cattle():
    """Test counting multiple cattle crossing simultaneously."""
    counter = CattleCounter(line_y=0.5, direction="top_to_bottom", min_crossing_distance=0)
    frame_h = 100

    # Three cattle above line
    det = make_detections([1, 2, 3], [
        [10, 15, 30, 25],
        [40, 15, 60, 25],
        [70, 15, 90, 25],
    ])
    counter.update(det, frame_h)

    # All three cross
    det = make_detections([1, 2, 3], [
        [10, 55, 30, 65],
        [40, 55, 60, 65],
        [70, 55, 90, 65],
    ])
    result = counter.update(det, frame_h)
    assert result["total_count"] == 3, f"Expected 3, got {result['total_count']}"
    print("[PASS] test_multiple_cattle")


def test_reset():
    """Test counter reset clears all state."""
    counter = CattleCounter(line_y=0.5, direction="top_to_bottom", min_crossing_distance=0)
    frame_h = 100

    det = make_detections([1], [[40, 15, 60, 25]])
    counter.update(det, frame_h)
    det = make_detections([1], [[40, 55, 60, 65]])
    counter.update(det, frame_h)
    assert counter.total_count == 1

    counter.reset()
    assert counter.total_count == 0
    assert len(counter.counted_ids) == 0
    print("[PASS] test_reset")


def test_bottom_to_top_direction():
    """Test counting in reverse direction."""
    counter = CattleCounter(line_y=0.5, direction="bottom_to_top", min_crossing_distance=0)
    frame_h = 100

    # Cattle below line
    det = make_detections([1], [[40, 55, 60, 65]])
    counter.update(det, frame_h)

    # Crosses upward
    det = make_detections([1], [[40, 15, 60, 25]])
    result = counter.update(det, frame_h)
    assert result["total_count"] == 1, "Should count bottom-to-top crossing"
    print("[PASS] test_bottom_to_top_direction")


if __name__ == "__main__":
    print("\n=== Cattle Counter Unit Tests ===\n")
    test_basic_counting()
    test_no_double_count()
    test_direction_gate()
    test_multiple_cattle()
    test_reset()
    test_bottom_to_top_direction()
    print("\n=== All tests passed! ===\n")