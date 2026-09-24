"""
merge_datasets.py — Merge all converted datasets into the unified mega-dataset.

Copies images and labels from individual source directories into the
final mega_dataset structure with train/val/test splits.
"""

import os
import shutil
import random
from pathlib import Path
from typing import Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

console = Console()

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def collect_paired_files(images_dir: str, labels_dir: str) -> list:
    """Collect image-label pairs from a source directory.

    Only includes images that have a corresponding label file.

    Args:
        images_dir: Directory containing images.
        labels_dir: Directory containing YOLO txt labels.

    Returns:
        List of (image_path, label_path) tuples.
    """
    pairs = []
    for img_path in Path(images_dir).iterdir():
        if img_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        label_path = Path(labels_dir) / f"{img_path.stem}.txt"
        if label_path.exists():
            pairs.append((str(img_path), str(label_path)))
    return pairs


def merge_and_split(
    sources: list,
    output_dir: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.15,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> dict:
    """Merge multiple dataset sources and split into train/val/test.

    Args:
        sources: List of dicts with keys 'name', 'images_dir', 'labels_dir'.
        output_dir: Root of the mega_dataset directory.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        seed: Random seed for reproducible splits.

    Returns:
        Stats dictionary with per-source and per-split counts.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    console.print(Panel(
        "[bold bright_white]Dataset Merger & Splitter[/bold bright_white]\n"
        f"[dim]Split: {train_ratio:.0%} train / {val_ratio:.0%} val / {test_ratio:.0%} test[/dim]\n"
        f"[dim]Seed: {seed}[/dim]",
        border_style="bright_green",
        title="🔀 Merge",
    ))

    # Create output directories
    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # Collect all pairs from all sources
    all_pairs = []
    source_stats = {}

    for source in sources:
        name = source["name"]
        pairs = collect_paired_files(source["images_dir"], source["labels_dir"])
        source_stats[name] = len(pairs)
        all_pairs.extend(pairs)
        console.print(f"  [cyan]{name}[/cyan]: {len(pairs)} image-label pairs")

    console.print(f"\n[bold]Total pairs collected: {len(all_pairs)}[/bold]")

    if not all_pairs:
        console.print("[red]✖ No image-label pairs found![/red]")
        return {"total": 0}

    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_pairs)

    n_total = len(all_pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    split_assignments = {
        "train": all_pairs[:n_train],
        "val": all_pairs[n_train:n_train + n_val],
        "test": all_pairs[n_train + n_val:],
    }

    split_counts = {}

    # Copy files to mega_dataset
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold bright_green]Merging →[/bold bright_green]"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("Merge", total=n_total)

        for split_name, pairs in split_assignments.items():
            split_counts[split_name] = len(pairs)
            for img_src, lbl_src in pairs:
                img_ext = Path(img_src).suffix
                stem = Path(img_src).stem

                # Avoid name collisions with a hash suffix
                unique_name = f"{stem}_{hash(img_src) % 100000:05d}"

                img_dst = os.path.join(output_dir, "images", split_name, f"{unique_name}{img_ext}")
                lbl_dst = os.path.join(output_dir, "labels", split_name, f"{unique_name}.txt")

                shutil.copy2(img_src, img_dst)
                shutil.copy2(lbl_src, lbl_dst)

                progress.advance(task)

    # Report
    table = Table(title="Mega Dataset Summary", border_style="bright_green")
    table.add_column("Split", style="bold")
    table.add_column("Images", justify="right", style="cyan")
    table.add_column("Percentage", justify="right")

    for split_name in splits:
        count = split_counts.get(split_name, 0)
        pct = f"{count / n_total * 100:.1f}%"
        table.add_row(split_name.capitalize(), str(count), pct)

    table.add_row("Total", str(n_total), "100%", style="bold bright_white")
    console.print(table)

    return {
        "total": n_total,
        "splits": split_counts,
        "sources": source_stats,
    }


if __name__ == "__main__":
    console.print("[dim]Import this module and call merge_and_split() with your sources.[/dim]")
