"""
pipeline.py - End-to-end cattle counting pipeline.

Orchestrates: Camera Capture -> Detection -> Tracking -> Counting -> Display/Log

This is the main entry point for running the cattle counter on
a live camera feed or recorded video.
"""

import os
import sys
import time

import cv2
import yaml
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich import box

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.detection.detector import CattleDetector
from src.tracking.tracker import CattleTracker
from src.counting.counter import CattleCounter
from src.utils.visualization import (
    draw_counting_line,
    draw_detections,
    draw_count_overlay,
    draw_new_count_flash,
)
from src.utils.logger import CountLogger

console = Console()


class CattleCountingPipeline:
    """End-to-end pipeline: Capture -> Detect -> Track -> Count -> Display.

    Integrates all system components into a single runnable pipeline
    that processes video frames in real-time.
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize the pipeline from a configuration file.

        Args:
            config_path: Path to the YAML configuration file.
        """
        console.print(Panel(
            "[bold bright_white]Cattle Gate Counter[/bold bright_white]\n"
            "[bold bright_cyan]Brute-Force Bovine Intelligence[/bold bright_cyan]\n"
            "[dim]Real-time Detection + Tracking + Counting[/dim]",
            border_style="bright_cyan",
            title="[bold bright_cyan]PIPELINE[/bold bright_cyan]",
            box=box.DOUBLE_EDGE,
        ))

        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        det_cfg = self.config.get("detection", {})
        trk_cfg = self.config.get("tracking", {})
        cnt_cfg = self.config.get("counting", {})
        out_cfg = self.config.get("output", {})

        self.detector = CattleDetector(
            weights_path=det_cfg.get("weights", "models/weights/best.pt"),
            conf_thresh=det_cfg.get("conf_thresh", 0.5),
            iou_thresh=det_cfg.get("iou_thresh", 0.45),
            imgsz=det_cfg.get("imgsz", 640),
            device=det_cfg.get("device", "0"),
        )

        self.tracker = CattleTracker(
            track_activation_threshold=trk_cfg.get("track_high_thresh", 0.5),
            lost_track_buffer=trk_cfg.get("track_buffer", 60),
            minimum_matching_threshold=trk_cfg.get("match_thresh", 0.8),
            frame_rate=self.config.get("source", {}).get("fps", 30),
        )

        self.counter = CattleCounter(
            line_y=cnt_cfg.get("line_y", 0.5),
            direction=cnt_cfg.get("direction", "top_to_bottom"),
            min_crossing_distance=cnt_cfg.get("min_crossing_distance", 10),
        )

        # Output settings
        self.display = out_cfg.get("display", True)
        self.save_video = out_cfg.get("save_video", False)
        self.save_path = out_cfg.get("save_path", "output/")

        # Logger
        self.logger = None
        if out_cfg.get("log_counts", True):
            self.logger = CountLogger(self.save_path)

        console.print("\n[bold black on bright_green] READY [/bold black on bright_green] "
                      "All components initialized\n")

    def run(self, source: str = None) -> dict:
        """Run the counting pipeline on a video source.

        Args:
            source: Video source (file path, RTSP URL, or USB index).
                   Overrides config if provided.

        Returns:
            Dictionary with session results.
        """
        # Determine source
        if source is None:
            src_cfg = self.config.get("source", {})
            source = src_cfg.get("path", "")

        if not source:
            console.print("[red]No video source specified. Set in config or pass as argument.[/red]")
            return {"error": "No source"}

        # Parse USB camera index
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        console.print(f"[bold]Opening source: {source}[/bold]")
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            console.print(f"[red]Failed to open video source: {source}[/red]")
            return {"error": f"Cannot open {source}"}

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        console.print(f"[dim]Resolution: {frame_width}x{frame_height} @ {source_fps:.1f} FPS[/dim]")

        # Video writer (optional)
        writer = None
        if self.save_video:
            os.makedirs(self.save_path, exist_ok=True)
            out_path = os.path.join(self.save_path, "annotated_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, source_fps, (frame_width, frame_height))
            console.print(f"[dim]Saving annotated video to: {out_path}[/dim]")

        # Processing loop
        frame_count = 0
        start_time = time.time()
        fps_calc = 0.0

        console.print("\n[bold black on bright_magenta] RUNNING [/bold black on bright_magenta] "
                      "Press 'q' to stop, 'r' to reset counter\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                frame_start = time.time()

                # Stage 1: Detection
                detections = self.detector.detect_supervision(frame)

                # Stage 2: Tracking
                tracked = self.tracker.update(detections)

                # Stage 3: Counting
                count_result = self.counter.update(tracked, frame_height)

                # Log new crossings
                if self.logger and count_result["new_counts"]:
                    for track_id in count_result["new_counts"]:
                        self.logger.log_crossing(
                            frame_count, track_id, count_result["total_count"]
                        )

                # Stage 4: Visualization
                if self.display or writer:
                    line_y = self.counter.get_line_y_pixels(frame_height)
                    frame = draw_counting_line(frame, line_y)
                    frame = draw_detections(frame, tracked, self.counter.counted_ids)
                    frame = draw_count_overlay(
                        frame, count_result["total_count"],
                        count_result["active_tracks"], fps_calc
                    )
                    if count_result["new_counts"]:
                        frame = draw_new_count_flash(frame, count_result["new_counts"])

                    if writer:
                        writer.write(frame)

                    if self.display:
                        cv2.imshow("Cattle Counter", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            console.print("\n[yellow]Stopped by user[/yellow]")
                            break
                        elif key == ord("r"):
                            self.counter.reset()
                            self.tracker.reset()
                            console.print("[yellow]Counter and tracker reset[/yellow]")

                # Calculate FPS
                frame_time = time.time() - frame_start
                fps_calc = 1.0 / max(frame_time, 1e-6)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
        finally:
            cap.release()
            if writer:
                writer.release()
            if self.display:
                cv2.destroyAllWindows()

        # Session summary
        elapsed = time.time() - start_time
        avg_fps = frame_count / max(elapsed, 1e-6)

        if self.logger:
            self.logger.log_session_summary(
                self.counter.total_count, frame_count, avg_fps, str(source)
            )

        # Final report
        console.print(Panel(
            f"[bold bright_white]Session Complete[/bold bright_white]\n\n"
            f"  [bold cyan]Total Count:[/bold cyan]    [bold bright_green]{self.counter.total_count}[/bold bright_green]\n"
            f"  [bold cyan]Frames:[/bold cyan]         {frame_count}\n"
            f"  [bold cyan]Duration:[/bold cyan]       {elapsed:.1f}s\n"
            f"  [bold cyan]Avg FPS:[/bold cyan]        {avg_fps:.1f}\n"
            f"  [bold cyan]Unique IDs:[/bold cyan]     {len(self.counter.counted_ids)}",
            border_style="bright_green",
            title="[bold bright_green]RESULTS[/bold bright_green]",
            box=box.DOUBLE_EDGE,
        ))

        return {
            "total_count": self.counter.total_count,
            "frames_processed": frame_count,
            "duration_seconds": elapsed,
            "avg_fps": avg_fps,
        }


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "config/settings.yaml"
    source = sys.argv[2] if len(sys.argv) > 2 else None

    pipeline = CattleCountingPipeline(config)
    pipeline.run(source)