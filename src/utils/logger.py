"""
logger.py - Session logging for cattle counting results.

Logs per-session and per-event counting data to CSV files
for validation and reporting.
"""

import os
import csv
from datetime import datetime
from pathlib import Path

from rich.console import Console

console = Console()


class CountLogger:
    """Logs cattle counting events and session summaries to CSV files.

    Creates two log files per session:
    - events.csv: Per-crossing events with timestamps and track IDs
    - summary.csv: Appended session summary with total counts
    """

    def __init__(self, output_dir: str = "output"):
        """Initialize the logger.

        Args:
            output_dir: Directory to write log files.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.events_file = os.path.join(output_dir, f"events_{self.session_id}.csv")
        self.summary_file = os.path.join(output_dir, "session_summary.csv")

        # Initialize events CSV
        with open(self.events_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "frame_number", "track_id", "running_count"])

        self.event_count = 0
        console.print(f"[dim]Logging to: {self.events_file}[/dim]")

    def log_crossing(self, frame_number: int, track_id: int, running_count: int) -> None:
        """Log a single cattle crossing event.

        Args:
            frame_number: Current frame number.
            track_id: Track ID of the cattle that crossed.
            running_count: Running total count after this crossing.
        """
        timestamp = datetime.now().isoformat()
        with open(self.events_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, frame_number, track_id, running_count])
        self.event_count += 1

    def log_session_summary(self, total_count: int, total_frames: int,
                            avg_fps: float, source: str = "") -> None:
        """Log a session summary.

        Args:
            total_count: Final cattle count for the session.
            total_frames: Total frames processed.
            avg_fps: Average processing FPS.
            source: Video source path or description.
        """
        timestamp = datetime.now().isoformat()
        file_exists = os.path.exists(self.summary_file)

        with open(self.summary_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["session_id", "timestamp", "source", "total_count",
                                 "total_frames", "avg_fps", "events_file"])
            writer.writerow([self.session_id, timestamp, source, total_count,
                             total_frames, f"{avg_fps:.1f}", self.events_file])

        console.print(f"\n[bold green]Session logged: {total_count} cattle counted[/bold green]")