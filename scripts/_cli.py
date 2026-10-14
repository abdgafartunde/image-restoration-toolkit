"""Shared command-line handling for the scripts in this directory."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_FIGURES = Path(__file__).resolve().parent.parent / "figures"


def output_dir(description: str) -> Path:
    """Parse ``--outdir`` (default: ``figures/`` in the repository) and create it.

    Building the default from ``__file__`` lets the scripts run from any
    working directory.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_FIGURES,
        help="directory for output figures (default: figures/ in the repository)",
    )
    outdir = parser.parse_args().outdir
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir
