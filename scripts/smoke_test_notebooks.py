"""Execute every tracked notebook in memory against the synthetic default data."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    notebook_paths = sorted((root / "notebooks").glob("*.ipynb"))
    if not notebook_paths:
        print("No notebooks found.")
        return 1
    for path in notebook_paths:
        with path.open(encoding="utf-8") as handle:
            notebook = nbformat.read(handle, as_version=4)
        NotebookClient(notebook, timeout=180, resources={"metadata": {"path": str(root)}}).execute()
        print(f"Passed: {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
