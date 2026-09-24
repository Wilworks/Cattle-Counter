"""
convert_formats.py — Unified annotation format converter.

Converts Pascal VOC (XML), COCO (JSON), and Open Images (CSV)
annotation formats into YOLO txt format for the mega-dataset pipeline.
"""

import json
import os
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

console = Console()


def voc_to_yolo(xml_path: str, output_dir: str, class_id: int = 0) -> Optional[str]:
    """Convert a single Pascal VOC XML annotation to YOLO txt format.

    Args:
        xml_path: Path to the VOC XML annotation file.
        output_dir: Directory to write the YOLO txt file.
        class_id: Class ID to assign (default 0 for cattle).

    Returns:
        Path to the created YOLO txt file, or None if no objects found.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        return None

    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    if img_w == 0 or img_h == 0:
        return None

    lines = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Convert to YOLO normalized format
        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        # Clamp to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    if not lines:
        return None

    stem = Path(xml_path).stem
    out_path = os.path.join(output_dir, f"{stem}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return out_path


def coco_to_yolo(coco_json_path: str, images_dir: str, output_dir: str,
                 target_categories: Optional[list] = None, class_id: int = 0) -> int:
    """Convert COCO JSON annotations to YOLO txt format.

    Filters for specific category names (e.g., ['cow', 'cattle']) and remaps
    all of them to a single class ID.

    Args:
        coco_json_path: Path to the COCO annotations JSON file.
        images_dir: Directory containing the source images.
        output_dir: Directory to write YOLO txt files.
        target_categories: List of COCO category names to include (e.g., ['cow']).
        class_id: Class ID to assign (default 0 for cattle).

    Returns:
        Number of annotation files created.
    """
    if target_categories is None:
        target_categories = ["cow", "cattle", "bull", "calf", "bovine"]

    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    # Map category IDs to names, filter for cattle-related
    cat_name_map = {c["id"]: c["name"] for c in coco["categories"]}
    target_cat_ids = {
        cid for cid, name in cat_name_map.items()
        if name.lower() in [t.lower() for t in target_categories]
    }

    if not target_cat_ids:
        console.print("[yellow]⚠ No matching categories found in COCO JSON[/yellow]")
        return 0

    # Build image ID to dimensions map
    img_map = {img["id"]: img for img in coco["images"]}

    # Group annotations by image ID
    img_annotations = {}
    for ann in coco["annotations"]:
        if ann["category_id"] in target_cat_ids:
            img_id = ann["image_id"]
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            img_annotations[img_id].append(ann)

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Converting COCO →[/bold cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("COCO", total=len(img_annotations))

        for img_id, anns in img_annotations.items():
            img_info = img_map[img_id]
            img_w = img_info["width"]
            img_h = img_info["height"]

            lines = []
            for ann in anns:
                x, y, w, h = ann["bbox"]  # COCO: [x_min, y_min, width, height]
                x_center = (x + w / 2.0) / img_w
                y_center = (y + h / 2.0) / img_h
                norm_w = w / img_w
                norm_h = h / img_h

                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                norm_w = max(0.0, min(1.0, norm_w))
                norm_h = max(0.0, min(1.0, norm_h))

                lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

            if lines:
                stem = Path(img_info["file_name"]).stem
                out_path = os.path.join(output_dir, f"{stem}.txt")
                with open(out_path, "w") as f:
                    f.write("\n".join(lines) + "\n")
                count += 1

            progress.advance(task)

    return count


def openimages_to_yolo(csv_path: str, output_dir: str, class_id: int = 0) -> int:
    """Convert Open Images CSV annotations to YOLO txt format.

    Expects the standard Open Images bbox CSV format with columns:
    ImageID, Source, LabelName, Confidence, XMin, XMax, YMin, YMax, ...

    Coordinates in Open Images are already normalized [0, 1].

    Args:
        csv_path: Path to the Open Images annotations CSV.
        output_dir: Directory to write YOLO txt files.
        class_id: Class ID to assign (default 0 for cattle).

    Returns:
        Number of annotation files created.
    """
    # Group rows by ImageID
    img_annotations = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["ImageID"]
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            img_annotations[img_id].append(row)

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Converting Open Images →[/bold green]"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("OpenImages", total=len(img_annotations))

        for img_id, rows in img_annotations.items():
            lines = []
            for row in rows:
                # Open Images: XMin, XMax, YMin, YMax are already normalized
                xmin = float(row["XMin"])
                xmax = float(row["XMax"])
                ymin = float(row["YMin"])
                ymax = float(row["YMax"])

                x_center = (xmin + xmax) / 2.0
                y_center = (ymin + ymax) / 2.0
                width = xmax - xmin
                height = ymax - ymin

                lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            if lines:
                out_path = os.path.join(output_dir, f"{img_id}.txt")
                with open(out_path, "w") as f:
                    f.write("\n".join(lines) + "\n")
                count += 1

            progress.advance(task)

    return count


def batch_convert_voc(xml_dir: str, output_dir: str, class_id: int = 0) -> int:
    """Convert all Pascal VOC XML files in a directory to YOLO format.

    Args:
        xml_dir: Directory containing VOC XML annotation files.
        output_dir: Directory to write YOLO txt files.
        class_id: Class ID to assign (default 0 for cattle).

    Returns:
        Number of annotation files created.
    """
    os.makedirs(output_dir, exist_ok=True)
    xml_files = list(Path(xml_dir).glob("*.xml"))
    count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold magenta]Converting VOC →[/bold magenta]"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("VOC", total=len(xml_files))

        for xml_file in xml_files:
            result = voc_to_yolo(str(xml_file), output_dir, class_id)
            if result:
                count += 1
            progress.advance(task)

    return count


if __name__ == "__main__":
    console.print(Panel(
        "[bold bright_white]Annotation Format Converter[/bold bright_white]\n"
        "[dim]VOC XML | COCO JSON | Open Images CSV → YOLO txt[/dim]",
        border_style="bright_cyan",
        title="🔄 Convert",
    ))
    console.print("[dim]Import this module and call the conversion functions directly.[/dim]")
