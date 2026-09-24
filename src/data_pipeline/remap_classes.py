"""
remap_classes.py — Remap all class IDs to a single class (0: cattle).

Handles YOLO txt files where datasets may have multiple classes
(cow, bull, calf, etc.) and unifies them to class 0.
"""

import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

console = Console()


def remap_labels(labels_dir: str, target_class: int = 0) -> dict:
    """Remap all class IDs in YOLO label files to a single target class.

    Args:
        labels_dir: Directory containing YOLO txt label files.
        target_class: The class ID to remap everything to (default 0).

    Returns:
        Dictionary with stats: total files processed, total boxes remapped,
        and original class distribution before remapping.
    """
    label_files = list(Path(labels_dir).glob("*.txt"))
    stats = {
        "files_processed": 0,
        "boxes_remapped": 0,
        "original_classes": {},
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold bright_yellow]Remapping classes →[/bold bright_yellow]"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("Remap", total=len(label_files))

        for label_file in label_files:
            lines = label_file.read_text().strip().split("\n")
            new_lines = []
            remapped = False

            for line in lines:
                if not line.strip():
                    continue

                parts = line.strip().split()
                if len(parts) >= 5:
                    original_class = int(parts[0])

                    # Track original class distribution
                    stats["original_classes"][original_class] = (
                        stats["original_classes"].get(original_class, 0) + 1
                    )

                    if original_class != target_class:
                        parts[0] = str(target_class)
                        remapped = True
                        stats["boxes_remapped"] += 1

                    new_lines.append(" ".join(parts))

            if remapped:
                label_file.write_text("\n".join(new_lines) + "\n")

            stats["files_processed"] += 1
            progress.advance(task)

    return stats


if __name__ == "__main__":
    import sys

    console.print(Panel(
        "[bold bright_white]Class Remapper[/bold bright_white]\n"
        "[dim]All classes → 0 (cattle)[/dim]",
        border_style="bright_yellow",
        title="🏷️ Remap",
    ))

    if len(sys.argv) < 2:
        console.print("[red]Usage: python remap_classes.py <labels_directory>[/red]")
        sys.exit(1)

    labels_dir = sys.argv[1]
    if not os.path.isdir(labels_dir):
        console.print(f"[red]Directory not found: {labels_dir}[/red]")
        sys.exit(1)

    stats = remap_labels(labels_dir)
    console.print(f"\n[bold green]✔ Processed {stats['files_processed']} files[/bold green]")
    console.print(f"[bold green]✔ Remapped {stats['boxes_remapped']} bounding boxes to class 0[/bold green]")
    if stats["original_classes"]:
        console.print("\n[bold]Original class distribution:[/bold]")
        for cls_id, count in sorted(stats["original_classes"].items()):
            console.print(f"  Class {cls_id}: {count} boxes")
