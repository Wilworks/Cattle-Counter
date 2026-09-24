"""
detector.py - YOLOv8 inference wrapper for cattle detection.

Provides a clean interface for loading a fine-tuned YOLOv8 model
and running detection on individual frames or video streams.
"""

import numpy as np
from ultralytics import YOLO
from rich.console import Console
from rich.panel import Panel

console = Console()


class CattleDetector:
    """Wrapper around YOLOv8 for cattle detection inference.

    Loads a fine-tuned YOLOv8 model and provides methods for
    running detection on frames with configurable thresholds.
    """

    def __init__(
        self,
        weights_path: str = "models/weights/best.pt",
        conf_thresh: float = 0.5,
        iou_thresh: float = 0.45,
        imgsz: int = 640,
        device: str = "0",
    ):
        """Initialize the cattle detector.

        Args:
            weights_path: Path to the fine-tuned YOLO model weights.
            conf_thresh: Minimum confidence threshold for detections.
            iou_thresh: IoU threshold for Non-Maximum Suppression.
            imgsz: Input image size for inference.
            device: Device to run inference on ("0" for GPU, "cpu" for CPU).
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self.device = device

        console.print(Panel(
            f"[bold bright_white]Loading YOLOv8 Model[/bold bright_white]\n"
            f"[dim]Weights: {weights_path}[/dim]\n"
            f"[dim]Device: {device} | Conf: {conf_thresh} | IoU: {iou_thresh}[/dim]",
            border_style="bright_cyan",
            title="[bright_cyan]Detector[/bright_cyan]",
        ))

        self.model = YOLO(weights_path)
        console.print("[bold black on bright_green] READY [/bold black on bright_green] "
                      "Model loaded successfully")

    def detect(self, frame: np.ndarray) -> list:
        """Run detection on a single frame.

        Args:
            frame: BGR image as a numpy array (from OpenCV).

        Returns:
            List of detections, each as a dict with keys:
            - bbox: [x1, y1, x2, y2] pixel coordinates
            - confidence: float
            - class_id: int
        """
        results = self.model(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                detections.append({
                    "bbox": bbox,
                    "confidence": conf,
                    "class_id": cls_id,
                })

        return detections

    def detect_supervision(self, frame: np.ndarray):
        """Run detection and return supervision-compatible Detections object.

        This is the preferred method when integrating with ByteTrack
        via the supervision library.

        Args:
            frame: BGR image as a numpy array (from OpenCV).

        Returns:
            supervision.Detections object.
        """
        import supervision as sv

        results = self.model(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        return sv.Detections.from_ultralytics(results)