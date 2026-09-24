"""
run_training.py - Comparative training runner for YOLOv8m/l/x.

Trains all three model variants on the mega-dataset and produces
a comparison report for model selection.
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.detection.train import train_model, print_hardware_card, MODELS

console = Console()


def run_comparative_training(config_path: str = "config/settings.yaml",
                              models_to_train: list = None) -> dict:
    """Train multiple YOLOv8 variants and produce a comparison.

    Args:
        config_path: Path to settings YAML.
        models_to_train: List of model keys to train. Defaults to all three.

    Returns:
        Dictionary mapping model keys to their training results.
    """
    if models_to_train is None:
        models_to_train = ["yolov8m", "yolov8l", "yolov8x"]

    console.print(Panel(
        "[bold bright_white]Comparative Model Training[/bold bright_white]\n"
        "[bold bright_magenta]Brute-Force Bovine Intelligence[/bold bright_magenta]\n"
        f"[dim]Models: {', '.join(models_to_train)}[/dim]",
        border_style="bright_magenta",
        title="[bold bright_magenta]TRAIN ALL[/bold bright_magenta]",
        box=box.DOUBLE_EDGE,
    ))

    print_hardware_card()

    with open(config_path, "r") as f:
        settings = yaml.safe_load(f)
    train_config = settings.get("training", {})

    all_results = {}
    total_start = time.time()

    for i, model_key in enumerate(models_to_train, 1):
        console.print(f"\n[bold]{'='*60}[/bold]")
        console.print(f"[bold bright_cyan]Training {i}/{len(models_to_train)}: "
                      f"{MODELS[model_key]['name']}[/bold bright_cyan]")
        console.print(f"[bold]{'='*60}[/bold]\n")

        result = train_model(model_key, train_config)
        all_results[model_key] = result

    total_elapsed = time.time() - total_start

    # Comparison report
    console.print(Panel(
        "[bold bright_white]Training Comparison Report[/bold bright_white]",
        border_style="bright_green",
        title="[bold bright_green]COMPARE[/bold bright_green]",
        box=box.DOUBLE_EDGE,
    ))

    table = Table(title="Model Comparison", border_style="bright_green", box=box.DOUBLE_EDGE)
    table.add_column("Model", style="bold")
    table.add_column("Parameters", style="cyan")
    table.add_column("Training Time", justify="right", style="magenta")
    table.add_column("Best Weights", style="green")

    for key, result in all_results.items():
        info = MODELS[key]
        elapsed_min = result["elapsed_seconds"] / 60
        table.add_row(
            info["name"],
            info["params"],
            f"{elapsed_min:.1f} min",
            result["best_weights"],
        )

    console.print(table)
    console.print(f"\n[bold]Total training time: {total_elapsed / 60:.1f} minutes[/bold]")
    console.print("\n[dim]Next: Run evaluation on test videos to compare counting accuracy.[/dim]")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Comparative YOLOv8 training")
    parser.add_argument("--config", default="config/settings.yaml", help="Config file path")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=["yolov8m", "yolov8l", "yolov8x"],
                        help="Specific models to train")
    args = parser.parse_args()

    run_comparative_training(args.config, args.models)