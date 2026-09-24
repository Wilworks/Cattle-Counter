"""
counter.py - Virtual line counting with 3-layer anti-double-count protection.

Implements the core counting algorithm:
1. ID Lock - counted IDs are permanently locked
2. Direction Gate - only exit-direction crossings count
3. Track Buffer - ByteTrack maintains IDs across brief disappearances
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import supervision as sv
from rich.console import Console

console = Console()


class CattleCounter:
    """Virtual line counter with anti-double-count protection.

    Counts cattle crossing a configurable virtual line in a specified
    direction. Each tracked animal is counted exactly once, regardless
    of step-backs or re-entries.
    """

    def __init__(
        self,
        line_y: float = 0.5,
        direction: str = "top_to_bottom",
        min_crossing_distance: int = 10,
    ):
        """Initialize the counter.

        Args:
            line_y: Normalized Y-position of the counting line (0.0=top, 1.0=bottom).
            direction: Exit direction - "top_to_bottom" or "bottom_to_top".
            min_crossing_distance: Minimum pixels past line before counting (hysteresis).
        """
        self.line_y_normalized = line_y
        self.direction = direction
        self.min_crossing_distance = min_crossing_distance

        # Core state
        self.total_count = 0
        self.counted_ids = set()
        self.centroid_history = defaultdict(list)

        # Session tracking
        self.frame_count = 0

    def get_line_y_pixels(self, frame_height: int) -> int:
        """Convert normalized line position to pixel coordinate.

        Args:
            frame_height: Height of the video frame in pixels.

        Returns:
            Y-coordinate of the counting line in pixels.
        """
        return int(self.line_y_normalized * frame_height)

    def update(self, detections: sv.Detections, frame_height: int) -> dict:
        """Process tracked detections and update count.

        Args:
            detections: Tracked detections with tracker_id populated.
            frame_height: Height of the current frame in pixels.

        Returns:
            Dictionary with:
            - total_count: Current total count
            - new_counts: List of track IDs counted this frame
            - active_tracks: Number of tracks being monitored
        """
        self.frame_count += 1
        line_y = self.get_line_y_pixels(frame_height)
        new_counts = []

        if detections.tracker_id is None:
            return {
                "total_count": self.total_count,
                "new_counts": [],
                "active_tracks": 0,
            }

        for i, tracker_id in enumerate(detections.tracker_id):
            tracker_id = int(tracker_id)

            # Calculate centroid
            bbox = detections.xyxy[i]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0

            # Store centroid history
            self.centroid_history[tracker_id].append(cy)

            # Keep only last 30 centroids to save memory
            if len(self.centroid_history[tracker_id]) > 30:
                self.centroid_history[tracker_id] = self.centroid_history[tracker_id][-30:]

            # Need at least 2 points to detect crossing
            if len(self.centroid_history[tracker_id]) < 2:
                continue

            # Layer 1: ID Lock - skip if already counted
            if tracker_id in self.counted_ids:
                continue

            # Get previous and current positions
            prev_cy = self.centroid_history[tracker_id][-2]
            curr_cy = self.centroid_history[tracker_id][-1]

            # Layer 2: Direction Gate - check crossing direction
            crossed = False
            if self.direction == "top_to_bottom":
                # Crossed if was above line and now below (with hysteresis)
                if prev_cy < line_y and curr_cy >= line_y + self.min_crossing_distance:
                    crossed = True
            elif self.direction == "bottom_to_top":
                # Crossed if was below line and now above (with hysteresis)
                if prev_cy > line_y and curr_cy <= line_y - self.min_crossing_distance:
                    crossed = True

            if crossed:
                # Count it and lock the ID
                self.counted_ids.add(tracker_id)
                self.total_count += 1
                new_counts.append(tracker_id)

        return {
            "total_count": self.total_count,
            "new_counts": new_counts,
            "active_tracks": len(detections.tracker_id),
        }

    def reset(self) -> None:
        """Reset the counter for a new counting session."""
        self.total_count = 0
        self.counted_ids.clear()
        self.centroid_history.clear()
        self.frame_count = 0
        console.print("[yellow]Counter reset[/yellow]")

    def get_status(self) -> dict:
        """Get current counter status."""
        return {
            "total_count": self.total_count,
            "counted_ids": len(self.counted_ids),
            "tracked_animals": len(self.centroid_history),
            "frames_processed": self.frame_count,
        }