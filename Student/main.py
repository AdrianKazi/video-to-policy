"""Minimal entry point for Student pipelines."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

from Autoencoder.cli import main as autoencoder_main  # noqa: E402
from Sequential.cli import main as sequential_main  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python main.py [autoencoder|sequential] ...")

    target = sys.argv[1].lower()
    argv = sys.argv[2:]

    if target in {"autoencoder", "ae"}:
        autoencoder_main(argv)
    elif target in {"sequential", "seq"}:
        sequential_main(argv)
    else:
        raise SystemExit(f"Unknown pipeline: {target}. Use 'autoencoder' or 'sequential'.")
