"""
tracker.py - ByteTrack multi-object tracking integration.

Wraps the supervision library's ByteTrack implementation to assign
persistent unique IDs to detected cattle across video frames.
"""

import numpy as np
import supervision as sv
from rich.console import Console
from rich.panel import Panel

console = Console()


class CattleTracker:
    """ByteTrack-based multi-object tracker for cattle.

    Uses motion-based association (no appearance features) to maintain
    persistent IDs across frames. Ideal for cattle which are visually
    similar to each other.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.5,
        lost_track_buffer: int = 60,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ):
        """Initialize the ByteTrack tracker.

        Args:
            track_activation_threshold: Minimum confidence to start a new track.
            lost_track_buffer: Number of frames to keep a lost track alive.
                              Critical for step-back protection (~2s at 30fps).
            minimum_matching_threshold: IoU threshold for matching detections to tracks.
            frame_rate: Expected FPS of the video source.
        """
        console.print(Panel(
            f"[bold bright_white]ByteTrack Tracker[/bold bright_white]\n"
            f"[dim]Buffer: {lost_track_buffer} frames | Match: {minimum_matching_threshold}[/dim]",
            border_style="bright_yellow",
            title="[bright_yellow]Tracker[/bright_yellow]",
        ))

        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

        self.frame_count = 0
        console.print("[bold black on bright_green] READY [/bold black on bright_green] "
                      "Tracker initialized")

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker with new frame detections.

        Args:
            detections: supervision.Detections object from the detector.

        Returns:
            supervision.Detections with tracker_id field populated.
        """
        self.frame_count += 1
        tracked = self.tracker.update_with_detections(detections)
        return tracked

    def reset(self) -> None:
        """Reset the tracker state for a new counting session."""
        self.tracker.reset()
        self.frame_count = 0
        console.print("[yellow]Tracker reset[/yellow]")

    @property
    def active_tracks(self) -> int:
        """Return the number of currently active tracks."""
        return len(self.tracker.tracked_tracks)