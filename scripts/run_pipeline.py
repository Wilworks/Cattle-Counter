"""
run_pipeline.py - Master bootstrap runner for the cattle counting pipeline.

Single entry point that handles environment checks and launches
the end-to-end counting pipeline.
"""

import os
import sys
import argparse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rich.console import Console
from rich.panel import Panel
from rich import box

console = Console()


def check_environment() -> bool:
    """Verify all required dependencies are available."""
    checks = []
    try:
        import torch
        gpu = torch.cuda.is_available()
        checks.append(("PyTorch", torch.__version__, True))
        checks.append(("CUDA Available", str(gpu), gpu))
        if gpu:
            checks.append(("GPU", torch.cuda.get_device_name(0), True))
    except ImportError:
        checks.append(("PyTorch", "NOT INSTALLED", False))

    try:
        import ultralytics
        checks.append(("Ultralytics", ultralytics.__version__, True))
    except ImportError:
        checks.append(("Ultralytics", "NOT INSTALLED", False))

    try:
        import supervision
        checks.append(("Supervision", supervision.__version__, True))
    except ImportError:
        checks.append(("Supervision", "NOT INSTALLED", False))

    try:
        import cv2
        checks.append(("OpenCV", cv2.__version__, True))
    except ImportError:
        checks.append(("OpenCV", "NOT INSTALLED", False))

    # Print checks
    all_pass = all(c[2] for c in checks)
    for name, version, ok in checks:
        status = "[bold black on bright_green] OK [/bold black on bright_green]" if ok \
            else "[bold black on bright_red] FAIL [/bold black on bright_red]"
        console.print(f"  {status} {name}: {version}")

    return all_pass


def main():
    """Main entry point for the cattle counting pipeline."""
    parser = argparse.ArgumentParser(description="Cattle Gate Counter - Real-time counting pipeline")
    parser.add_argument("--config", type=str, default="config/settings.yaml",
                        help="Path to configuration file")
    parser.add_argument("--source", type=str, default=None,
                        help="Video source (file path, RTSP URL, or USB camera index)")
    parser.add_argument("--check", action="store_true",
                        help="Only run environment checks, don't start pipeline")
    args = parser.parse_args()

    # Hero banner
    console.print(Panel(
        "[bold bright_white]Cattle Gate Counter[/bold bright_white]\n"
        "[bold bright_cyan]Brute-Force Bovine Intelligence[/bold bright_cyan]",
        border_style="bright_cyan",
        title="[bold bright_cyan]RUN[/bold bright_cyan]",
        box=box.DOUBLE_EDGE,
    ))

    # Environment check
    console.print("\n[bold]Environment Check:[/bold]")
    env_ok = check_environment()

    if args.check:
        if env_ok:
            console.print("\n[bold green]All checks passed.[/bold green]")
        else:
            console.print("\n[bold red]Some checks failed. Install missing dependencies.[/bold red]")
        return

    if not env_ok:
        console.print("\n[bold red]Environment check failed. Run: pip install -r requirements.txt[/bold red]")
        sys.exit(1)

    # Launch pipeline
    from src.pipeline.pipeline import CattleCountingPipeline
    pipeline = CattleCountingPipeline(args.config)
    results = pipeline.run(args.source)

    return results


if __name__ == "__main__":
    main()