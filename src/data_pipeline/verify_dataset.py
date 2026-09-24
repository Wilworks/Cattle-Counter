"""
verify_dataset.py - Verify the mega-dataset integrity and produce a rich stats report.

Checks for missing labels, corrupt images, annotation format validity,
and generates a comprehensive dataset quality report.
"""

import os
from pathlib import Path
from collections import Counter

from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

console = Console()

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def verify_dataset(dataset_dir: str) -> dict:
    """Verify the mega-dataset and produce a quality report.

    Args:
        dataset_dir: Root directory of the mega_dataset (contains images/ and labels/).

    Returns:
        Dictionary with verification stats and any issues found.
    """
    console.print(Panel(
        "[bold bright_white]Dataset Verification[/bold bright_white]\n"
        f"[dim]{dataset_dir}[/dim]",
        border_style="bright_cyan",
        title="[bright_cyan]Verify[/bright_cyan]",
    ))

    splits = ["train", "val", "test"]
    stats = {
        "total_images": 0,
        "total_labels": 0,
        "total_boxes": 0,
        "missing_labels": [],
        "orphan_labels": [],
        "corrupt_images": [],
        "invalid_labels": [],
        "out_of_range_boxes": [],
        "class_distribution": Counter(),
        "image_sizes": [],
        "per_split": {},
    }

    for split in splits:
        images_dir = os.path.join(dataset_dir, "images", split)
        labels_dir = os.path.join(dataset_dir, "labels", split)

        if not os.path.isdir(images_dir):
            console.print(f"[yellow]Warning: Missing split directory: {images_dir}[/yellow]")
            continue

        image_files = {
            f.stem: f for f in Path(images_dir).iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        }
        label_files = {
            f.stem: f for f in Path(labels_dir).iterdir()
            if f.suffix == ".txt"
        } if os.path.isdir(labels_dir) else {}

        split_stats = {"images": len(image_files), "labels": len(label_files), "boxes": 0, "issues": 0}

        for stem in image_files:
            if stem not in label_files:
                stats["missing_labels"].append(f"{split}/{stem}")
                split_stats["issues"] += 1

        for stem in label_files:
            if stem not in image_files:
                stats["orphan_labels"].append(f"{split}/{stem}")
                split_stats["issues"] += 1

        with Progress(
            SpinnerColumn(), TextColumn(f"[bold cyan]Verifying {split}[/bold cyan]"),
            BarColumn(), MofNCompleteColumn(),
        ) as progress:
            task = progress.add_task(split, total=len(image_files))

            for stem, img_path in image_files.items():
                try:
                    img = Image.open(img_path)
                    img.verify()
                    img = Image.open(img_path)
                    w, h = img.size
                    stats["image_sizes"].append((w, h))
                except Exception:
                    stats["corrupt_images"].append(f"{split}/{img_path.name}")
                    split_stats["issues"] += 1
                    progress.advance(task)
                    continue

                label_path = label_files.get(stem)
                if label_path and label_path.exists():
                    try:
                        lines = label_path.read_text().strip().split("\n")
                        for line_num, line in enumerate(lines, 1):
                            if not line.strip():
                                continue
                            parts = line.strip().split()
                            if len(parts) != 5:
                                stats["invalid_labels"].append(f"{split}/{stem}.txt:L{line_num}")
                                split_stats["issues"] += 1
                                continue
                            cls_id = int(parts[0])
                            values = [float(v) for v in parts[1:]]
                            stats["class_distribution"][cls_id] += 1
                            split_stats["boxes"] += 1
                            if any(v < 0.0 or v > 1.0 for v in values):
                                stats["out_of_range_boxes"].append(f"{split}/{stem}.txt:L{line_num}")
                    except Exception as e:
                        stats["invalid_labels"].append(f"{split}/{stem}.txt ({e})")
                        split_stats["issues"] += 1

                progress.advance(task)

        stats["total_images"] += split_stats["images"]
        stats["total_labels"] += split_stats["labels"]
        stats["total_boxes"] += split_stats["boxes"]
        stats["per_split"][split] = split_stats

    _print_report(stats)
    return stats


def _print_report(stats: dict) -> None:
    """Print a rich verification report."""
    table = Table(title="Dataset Splits", border_style="bright_cyan")
    table.add_column("Split", style="bold")
    table.add_column("Images", justify="right", style="cyan")
    table.add_column("Labels", justify="right", style="green")
    table.add_column("Boxes", justify="right", style="magenta")
    table.add_column("Issues", justify="right", style="red")

    for split, s in stats["per_split"].items():
        table.add_row(split.capitalize(), str(s["images"]), str(s["labels"]),
                      str(s["boxes"]), str(s["issues"]) if s["issues"] > 0 else "[green]0[/green]")

    table.add_row("Total", str(stats["total_images"]), str(stats["total_labels"]),
                  str(stats["total_boxes"]), "", style="bold bright_white")
    console.print(table)

    if stats["class_distribution"]:
        cls_table = Table(title="Class Distribution", border_style="bright_yellow")
        cls_table.add_column("Class ID", style="bold")
        cls_table.add_column("Count", justify="right", style="cyan")
        for cls_id, count in sorted(stats["class_distribution"].items()):
            cls_table.add_row(str(cls_id), str(count))
        console.print(cls_table)

    if stats["image_sizes"]:
        widths = [s[0] for s in stats["image_sizes"]]
        heights = [s[1] for s in stats["image_sizes"]]
        console.print(f"\n[bold]Image Sizes:[/bold]")
        console.print(f"  Width:  min={min(widths)}, max={max(widths)}, avg={sum(widths)//len(widths)}")
        console.print(f"  Height: min={min(heights)}, max={max(heights)}, avg={sum(heights)//len(heights)}")

    total_issues = (len(stats["missing_labels"]) + len(stats["orphan_labels"]) +
                    len(stats["corrupt_images"]) + len(stats["invalid_labels"]) +
                    len(stats["out_of_range_boxes"]))

    if total_issues == 0:
        console.print("\n[bold black on bright_green] PASS [/bold black on bright_green] "
                      "Dataset verification passed - no issues found")
    else:
        console.print(f"\n[bold black on bright_red] {total_issues} ISSUES [/bold black on bright_red]")
        if stats["missing_labels"]:
            console.print(f"  [red]Missing labels: {len(stats['missing_labels'])}[/red]")
        if stats["orphan_labels"]:
            console.print(f"  [red]Orphan labels: {len(stats['orphan_labels'])}[/red]")
        if stats["corrupt_images"]:
            console.print(f"  [red]Corrupt images: {len(stats['corrupt_images'])}[/red]")
        if stats["invalid_labels"]:
            console.print(f"  [red]Invalid labels: {len(stats['invalid_labels'])}[/red]")
        if stats["out_of_range_boxes"]:
            console.print(f"  [yellow]Out-of-range boxes: {len(stats['out_of_range_boxes'])}[/yellow]")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        console.print("[red]Usage: python verify_dataset.py <mega_dataset_dir>[/red]")
        sys.exit(1)
    verify_dataset(sys.argv[1])