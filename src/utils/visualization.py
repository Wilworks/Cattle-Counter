"""
visualization.py - Draw bounding boxes, IDs, counting line, and count overlay.

Handles all visual annotations on video frames for the live display.
"""

import cv2
import numpy as np


# Color palette (BGR)
COLORS = {
    "box": (0, 255, 200),          # Bright cyan-green
    "box_counted": (0, 200, 0),    # Green (already counted)
    "line": (0, 100, 255),         # Orange
    "text_bg": (30, 30, 30),       # Dark background
    "text": (255, 255, 255),       # White
    "count_bg": (0, 120, 255),     # Orange
    "id_text": (255, 255, 0),      # Yellow
}


def draw_counting_line(frame: np.ndarray, line_y: int, color: tuple = None) -> np.ndarray:
    """Draw the virtual counting line across the frame.

    Args:
        frame: BGR image as numpy array.
        line_y: Y-coordinate of the counting line in pixels.
        color: Optional BGR color tuple.

    Returns:
        Annotated frame.
    """
    color = color or COLORS["line"]
    h, w = frame.shape[:2]
    cv2.line(frame, (0, line_y), (w, line_y), color, 2, cv2.LINE_AA)

    # Label
    cv2.putText(
        frame, "COUNTING LINE", (10, line_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
    )
    return frame


def draw_detections(
    frame: np.ndarray,
    detections,
    counted_ids: set,
) -> np.ndarray:
    """Draw bounding boxes and track IDs on the frame.

    Args:
        frame: BGR image as numpy array.
        detections: supervision.Detections with tracker_id populated.
        counted_ids: Set of track IDs that have been counted.

    Returns:
        Annotated frame.
    """
    if detections.tracker_id is None:
        return frame

    for i, tracker_id in enumerate(detections.tracker_id):
        tracker_id = int(tracker_id)
        bbox = detections.xyxy[i].astype(int)
        x1, y1, x2, y2 = bbox

        # Color based on counted status
        is_counted = tracker_id in counted_ids
        color = COLORS["box_counted"] if is_counted else COLORS["box"]

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw ID label
        label = f"ID:{tracker_id}"
        if is_counted:
            label += " [OK]"

        # Confidence if available
        if detections.confidence is not None:
            conf = detections.confidence[i]
            label += f" {conf:.2f}"

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), COLORS["text_bg"], -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["id_text"], 1, cv2.LINE_AA)

        # Draw centroid
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 4, color, -1)

    return frame


def draw_count_overlay(
    frame: np.ndarray,
    total_count: int,
    active_tracks: int = 0,
    fps: float = 0.0,
) -> np.ndarray:
    """Draw the count display overlay in the top-right corner.

    Args:
        frame: BGR image as numpy array.
        total_count: Current total cattle count.
        active_tracks: Number of currently active tracks.
        fps: Current processing FPS.

    Returns:
        Annotated frame.
    """
    h, w = frame.shape[:2]

    # Main count box
    count_text = f"COUNT: {total_count}"
    (tw, th), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)

    box_x = w - tw - 30
    box_y = 10
    box_w = tw + 20
    box_h = th + 20

    # Background with slight transparency effect
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h),
                  COLORS["count_bg"], -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Count text
    cv2.putText(frame, count_text, (box_x + 10, box_y + th + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS["text"], 3, cv2.LINE_AA)

    # Info bar below count
    info_y = box_y + box_h + 25
    info_text = f"Active: {active_tracks}"
    cv2.putText(frame, info_text, (box_x + 10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    if fps > 0:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (box_x + 10, info_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def draw_new_count_flash(frame: np.ndarray, new_ids: list) -> np.ndarray:
    """Draw a brief flash notification when new cattle are counted.

    Args:
        frame: BGR image as numpy array.
        new_ids: List of newly counted track IDs this frame.

    Returns:
        Annotated frame.
    """
    if not new_ids:
        return frame

    h, w = frame.shape[:2]
    text = f"+{len(new_ids)} COUNTED"

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cx = (w - tw) // 2
    cy = h // 2

    # Flash background
    cv2.rectangle(frame, (cx - 10, cy - th - 10), (cx + tw + 10, cy + 10),
                  (0, 180, 0), -1)
    cv2.putText(frame, text, (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    return frame