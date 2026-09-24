"""
deduplicate.py — Remove duplicate and near-duplicate images using perceptual hashing.

Uses pHash (perceptual hash) to detect visually similar images across
merged datasets, preventing the model from training on duplicates.
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

from PIL import Image
import imagehash
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

console = Console()

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def compute_hashes(images_dir: str, hash_size: int = 16) -> dict:
    """Compute perceptual hashes for all images in a directory.

    Args:
        images_dir: Directory containing images.
        hash_size: Size of the hash (higher = more sensitive). Default 16.

    Returns:
        Dictionary mapping image path to its perceptual hash.
    """
    image_files = [
        f for f in Path(images_dir).iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    hashes = {}
    errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold bright_cyan]Hashing images →[/bold bright_cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("Hash", total=len(image_files))

        for img_path in image_files:
            try:
                img = Image.open(img_path)
                h = imagehash.phash(img, hash_size=hash_size)
                hashes[str(img_path)] = h
            except Exception:
                errors += 1

            progress.advance(task)

    if errors > 0:
        console.print(f"[yellow]⚠ {errors} images could not be hashed (corrupt/unreadable)[/yellow]")

    return hashes


def find_duplicates(hashes: dict, threshold: int = 6) -> list:
    """Find duplicate image groups based on hash similarity.

    Args:
        hashes: Dictionary mapping image paths to perceptual hashes.
        threshold: Maximum hamming distance to consider as duplicate.
                   0 = exact match, 6 = visually very similar.

    Returns:
        List of duplicate groups, where each group is a list of image paths.
    """
    paths = list(hashes.keys())
    hash_values = [hashes[p] for p in paths]
    visited = set()
    duplicate_groups = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold bright_magenta]Finding duplicates →[/bold bright_magenta]"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("Compare", total=len(paths))

        for i in range(len(paths)):
            if i in visited:
                progress.advance(task)
                continue

            group = [paths[i]]
            for j in range(i + 1, len(paths)):
                if j in visited:
                    continue
                distance = hash_values[i] - hash_values[j]
                if distance <= threshold:
                    group.append(paths[j])
                    visited.add(j)

            if len(group) > 1:
                duplicate_groups.append(group)
                visited.add(i)

            progress.advance(task)

    return duplicate_groups


def remove_duplicates(
    images_dir: str,
    labels_dir: str,
    threshold: int = 6,
    hash_size: int = 16,
    dry_run: bool = False,
) -> dict:
    """Find and remove duplicate images and their corresponding labels.

    Keeps the first image in each duplicate group, removes the rest.

    Args:
        images_dir: Directory containing images.
        labels_dir: Directory containing YOLO label files.
        threshold: Hamming distance threshold for duplicate detection.
        hash_size: Perceptual hash size.
        dry_run: If True, only report duplicates without deleting.

    Returns:
        Stats dictionary with counts of duplicates found and removed.
    """
    console.print(Panel(
        f"[bold bright_white]Deduplication[/bold bright_white]\n"
        f"[dim]Images: {images_dir}[/dim]\n"
        f"[dim]Threshold: {threshold} | Hash size: {hash_size}[/dim]",
        border_style="bright_magenta",
        title="🔍 Dedup",
    ))

    hashes = compute_hashes(images_dir, hash_size)
    duplicate_groups = find_duplicates(hashes, threshold)

    total_duplicates = sum(len(g) - 1 for g in duplicate_groups)
    stats = {
        "total_images": len(hashes),
        "duplicate_groups": len(duplicate_groups),
        "duplicates_to_remove": total_duplicates,
        "removed": 0,
    }

    if not duplicate_groups:
        console.print("[bold green]✔ No duplicates found![/bold green]")
        return stats

    # Report
    table = Table(title="Duplicate Groups Found", border_style="bright_magenta")
    table.add_column("Group", style="bold")
    table.add_column("Keep", style="green")
    table.add_column("Remove", style="red")

    for i, group in enumerate(duplicate_groups[:20]):  # Show first 20 groups
        keep = Path(group[0]).name
        remove = ", ".join(Path(p).name for p in group[1:])
        table.add_row(str(i + 1), keep, remove)

    if len(duplicate_groups) > 20:
        table.add_row("...", f"+ {len(duplicate_groups) - 20} more groups", "")

    console.print(table)

    if dry_run:
        console.print(f"\n[yellow]DRY RUN: Would remove {total_duplicates} duplicate images[/yellow]")
        return stats

    # Remove duplicates (keep first in each group)
    for group in duplicate_groups:
        for dup_path in group[1:]:  # Skip first (keep it)
            # Remove image
            if os.path.exists(dup_path):
                os.remove(dup_path)

            # Remove corresponding label
            label_path = os.path.join(
                labels_dir,
                Path(dup_path).stem + ".txt"
            )
            if os.path.exists(label_path):
                os.remove(label_path)

            stats["removed"] += 1

    console.print(f"\n[bold green]✔ Removed {stats['removed']} duplicate images + labels[/bold green]")
    return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        console.print("[red]Usage: python deduplicate.py <images_dir> <labels_dir> [--dry-run][/red]")
        sys.exit(1)

    dry = "--dry-run" in sys.argv
    stats = remove_duplicates(sys.argv[1], sys.argv[2], dry_run=dry)

    console.print(f"\n[bold]Summary: {stats['total_images']} images scanned, "
                  f"{stats['duplicate_groups']} duplicate groups, "
                  f"{stats['removed']} removed[/bold]")
