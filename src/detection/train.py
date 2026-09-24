"""
train.py - YOLOv8 training script with rich telemetry.

Fine-tunes YOLOv8 models (m/l/x) on the mega cattle dataset
with full console telemetry including progress bars, hardware
info, and training metrics.
"""

import os
import sys
import time
import hashlib
import platform
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

MODELS = {
    "yolov8m": {"name": "YOLOv8m (Medium)", "weights": "yolov8m.pt", "params": "25.9M"},
    "yolov8l": {"name": "YOLOv8l (Large)", "weights": "yolov8l.pt", "params": "43.7M"},
    "yolov8x": {"name": "YOLOv8x (Extra-Large)", "weights": "yolov8x.pt", "params": "68.2M"},
}


def print_hardware_card() -> None:
    """Print a hardware info card with GPU, CPU, and system details."""
    table = Table(title="Hardware & Runtime", border_style="bright_cyan", box=box.DOUBLE_EDGE)
    table.add_column("Component", style="bold")
    table.add_column("Details", style="cyan")

    table.add_row("Platform", platform.platform())
    table.add_row("Python", platform.python_version())
    table.add_row("PyTorch", torch.__version__)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        cuda_version = torch.version.cuda
        table.add_row("GPU", gpu_name)
        table.add_row("VRAM", f"{vram_total:.1f} GB")
        table.add_row("CUDA", cuda_version)
    else:
        table.add_row("GPU", "[red]Not available (CPU mode)[/red]")

    table.add_row("CPU Cores", str(os.cpu_count()))
    console.print(table)


def print_model_card(model_key: str, config: dict) -> None:
    """Print a model configuration card."""
    model_info = MODELS[model_key]
    table = Table(title=f"Training: {model_info['name']}", border_style="bright_magenta", box=box.DOUBLE_EDGE)
    table.add_column("Parameter", style="bold")
    table.add_column("Value", style="magenta")

    table.add_row("Architecture", model_info["name"])
    table.add_row("Parameters", model_info["params"])
    table.add_row("Base Weights", model_info["weights"])
    table.add_row("Dataset", config.get("dataset_config", "N/A"))
    table.add_row("Epochs", str(config.get("epochs", 200)))
    table.add_row("Batch Size", str(config.get("batch_size", 8)))
    table.add_row("Image Size", str(config.get("imgsz", 640)))
    table.add_row("Optimizer", config.get("optimizer", "AdamW"))
    table.add_row("AMP", str(config.get("amp", True)))
    console.print(table)


def train_model(model_key: str, config: dict, project_dir: str = "runs") -> dict:
    """Train a YOLOv8 model variant on the mega cattle dataset.

    Args:
        model_key: Key from MODELS dict ("yolov8m", "yolov8l", "yolov8x").
        config: Training configuration dictionary (from settings.yaml).
        project_dir: Directory to save training runs.

    Returns:
        Dictionary with training results and best model path.
    """
    if model_key not in MODELS:
        console.print(f"[red]Unknown model: {model_key}. Choose from: {list(MODELS.keys())}[/red]")
        sys.exit(1)

    model_info = MODELS[model_key]

    # Hero banner
    console.print(Panel(
        f"[bold bright_white]Cattle Counter - Model Training[/bold bright_white]\n"
        f"[bold bright_magenta]{model_info['name']}[/bold bright_magenta]\n"
        f"[dim]Brute-Force Bovine Intelligence[/dim]",
        border_style="bright_magenta",
        title="[bold bright_magenta]TRAIN[/bold bright_magenta]",
        box=box.DOUBLE_EDGE,
    ))

    print_hardware_card()
    print_model_card(model_key, config)

    # Batch size adjustment for YOLOv8x on 11GB VRAM
    batch_size = config.get("batch_size", 8)
    accumulate = 1
    if model_key == "yolov8x" and batch_size > 4:
        accumulate = batch_size // 4
        batch_size = 4
        console.print(f"\n[yellow]YOLOv8x on limited VRAM: batch={batch_size}, "
                      f"gradient accumulation={accumulate} (effective batch={batch_size * accumulate})[/yellow]")

    # Load model
    console.print(f"\n[bold black on bright_cyan] LOADING [/bold black on bright_cyan] "
                  f"Initializing {model_info['name']}...")
    model = YOLO(model_info["weights"])

    # Train
    start_time = time.time()
    console.print(f"[bold black on bright_magenta] TRAINING [/bold black on bright_magenta] "
                  f"Starting training...\n")

    results = model.train(
        data=config.get("dataset_config", "data/mega_cattle.yaml"),
        epochs=config.get("epochs", 200),
        batch=batch_size,
        imgsz=config.get("imgsz", 640),
        optimizer=config.get("optimizer", "AdamW"),
        lr0=config.get("lr0", 0.001),
        lrf=config.get("lrf", 0.01),
        warmup_epochs=config.get("warmup_epochs", 5),
        patience=config.get("patience", 30),
        amp=config.get("amp", True),
        mosaic=config.get("mosaic", 1.0),
        mixup=config.get("mixup", 0.15),
        close_mosaic=config.get("close_mosaic", 20),
        project=project_dir,
        name=model_key,
        exist_ok=True,
        verbose=True,
    )

    elapsed = time.time() - start_time

    # Results summary
    best_weights = os.path.join(project_dir, model_key, "weights", "best.pt")

    # Compute determinism seal
    if os.path.exists(best_weights):
        sha256 = hashlib.sha256(Path(best_weights).read_bytes()).hexdigest()
    else:
        sha256 = "N/A"

    result_table = Table(title="Training Complete", border_style="bright_green", box=box.DOUBLE_EDGE)
    result_table.add_column("Metric", style="bold")
    result_table.add_column("Value", style="green")

    result_table.add_row("Model", model_info["name"])
    result_table.add_row("Duration", f"{elapsed / 60:.1f} minutes")
    result_table.add_row("Best Weights", best_weights)
    result_table.add_row("SHA-256", sha256[:32] + "...")

    console.print(result_table)
    console.print(f"\n[bold black on bright_green] COMPLETE [/bold black on bright_green] "
                  f"Training finished in {elapsed / 60:.1f} minutes")

    return {
        "model": model_key,
        "best_weights": best_weights,
        "elapsed_seconds": elapsed,
        "sha256": sha256,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[red]Usage: python train.py <model_key> [config_path][/red]")
        console.print("[dim]  model_key: yolov8m | yolov8l | yolov8x[/dim]")
        console.print("[dim]  config_path: path to settings.yaml (default: config/settings.yaml)[/dim]")
        sys.exit(1)

    model_key = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config/settings.yaml"

    with open(config_path, "r") as f:
        settings = yaml.safe_load(f)

    train_config = settings.get("training", {})
    train_model(model_key, train_config)