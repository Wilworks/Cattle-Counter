"""
build_mega_dataset.py - Orchestrates the entire data pipeline.

Downloads, converts, remaps, deduplicates, merges, and verifies
all cattle datasets into the unified mega-dataset.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rich.console import Console
from rich.panel import Panel
from rich import box

from src.data_pipeline.convert_formats import batch_convert_voc, coco_to_yolo, openimages_to_yolo
from src.data_pipeline.remap_classes import remap_labels
from src.data_pipeline.deduplicate import remove_duplicates
from src.data_pipeline.merge_datasets import merge_and_split
from src.data_pipeline.verify_dataset import verify_dataset

console = Console()


def build_mega_dataset():
    """Run the complete data pipeline to build the mega-dataset.

    This script expects datasets to be manually downloaded into
    data/sources/<source_name>/ with their original formats.
    It then converts, remaps, deduplicates, merges, and verifies.
    """
    console.print(Panel(
        "[bold bright_white]Mega Dataset Builder[/bold bright_white]\n"
        "[bold bright_cyan]Brute-Force Bovine Intelligence[/bold bright_cyan]\n"
        "[dim]Convert -> Remap -> Dedup -> Merge -> Verify[/dim]",
        border_style="bright_cyan",
        title="[bold bright_cyan]BUILD[/bold bright_cyan]",
        box=box.DOUBLE_EDGE,
    ))

    sources_dir = os.path.join(PROJECT_ROOT, "data", "sources")
    mega_dir = os.path.join(PROJECT_ROOT, "data", "mega_dataset")

    # Check for source datasets
    available_sources = []
    for source_name in os.listdir(sources_dir):
        source_path = os.path.join(sources_dir, source_name)
        if os.path.isdir(source_path) and source_name != ".gitkeep":
            available_sources.append(source_name)

    if not available_sources:
        console.print("[yellow]No datasets found in data/sources/[/yellow]")
        console.print("[dim]Download datasets into data/sources/<name>/ first.[/dim]")
        console.print("\n[bold]Expected structure per source:[/bold]")
        console.print("  data/sources/<name>/images/  (image files)")
        console.print("  data/sources/<name>/labels/  (annotation files)")
        console.print("\n[bold]Supported annotation formats:[/bold]")
        console.print("  - YOLO txt (ready to use)")
        console.print("  - Pascal VOC XML (auto-converted)")
        console.print("  - COCO JSON (auto-converted)")
        return

    console.print(f"\n[bold]Found {len(available_sources)} source datasets:[/bold]")
    for name in available_sources:
        console.print(f"  [cyan]{name}[/cyan]")

    # Step 1: Convert formats (if needed)
    console.print("\n[bold bright_magenta]Step 1: Format Conversion[/bold bright_magenta]")
    for source_name in available_sources:
        source_path = os.path.join(sources_dir, source_name)
        images_dir = os.path.join(source_path, "images")
        labels_dir = os.path.join(source_path, "labels")

        if not os.path.isdir(images_dir):
            console.print(f"  [yellow]Skipping {source_name}: no images/ directory[/yellow]")
            continue

        os.makedirs(labels_dir, exist_ok=True)

        # Check for VOC XML files
        xml_dir = os.path.join(source_path, "annotations_xml")
        if os.path.isdir(xml_dir):
            console.print(f"  [cyan]Converting VOC XML for {source_name}...[/cyan]")
            batch_convert_voc(xml_dir, labels_dir)

        # Check for COCO JSON
        coco_json = os.path.join(source_path, "annotations.json")
        if os.path.exists(coco_json):
            console.print(f"  [cyan]Converting COCO JSON for {source_name}...[/cyan]")
            coco_to_yolo(coco_json, images_dir, labels_dir)

    # Step 2: Remap classes
    console.print("\n[bold bright_yellow]Step 2: Class Remapping (all -> 0: cattle)[/bold bright_yellow]")
    for source_name in available_sources:
        labels_dir = os.path.join(sources_dir, source_name, "labels")
        if os.path.isdir(labels_dir):
            stats = remap_labels(labels_dir)
            console.print(f"  [cyan]{source_name}[/cyan]: {stats['boxes_remapped']} boxes remapped")

    # Step 3: Merge and split
    console.print("\n[bold bright_green]Step 3: Merge & Split[/bold bright_green]")
    merge_sources = []
    for source_name in available_sources:
        source_path = os.path.join(sources_dir, source_name)
        images_dir = os.path.join(source_path, "images")
        labels_dir = os.path.join(source_path, "labels")
        if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
            merge_sources.append({
                "name": source_name,
                "images_dir": images_dir,
                "labels_dir": labels_dir,
            })

    if merge_sources:
        merge_and_split(merge_sources, mega_dir)

    # Step 4: Deduplicate
    console.print("\n[bold bright_magenta]Step 4: Deduplication[/bold bright_magenta]")
    for split in ["train", "val", "test"]:
        images_dir = os.path.join(mega_dir, "images", split)
        labels_dir = os.path.join(mega_dir, "labels", split)
        if os.path.isdir(images_dir):
            remove_duplicates(images_dir, labels_dir)

    # Step 5: Verify
    console.print("\n[bold bright_cyan]Step 5: Verification[/bold bright_cyan]")
    verify_dataset(mega_dir)

    console.print(Panel(
        "[bold black on bright_green] COMPLETE [/bold black on bright_green]\n"
        "[bold]Mega-dataset is ready for training![/bold]",
        border_style="bright_green",
    ))


if __name__ == "__main__":
    build_mega_dataset()