"""Fail if staged or tracked files look like private inputs, outputs, or credentials."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


BLOCKED_PATHS = (
    re.compile(r"(^|/)garmin_data_download(/|$)", re.I),
    re.compile(r"(^|/)daylio_export_.*\.csv$", re.I),
    re.compile(r"(^|/)config\.local\.toml$", re.I),
    re.compile(r"(^|/)(data/derived|reports/private|notebooks/private)(/|$)", re.I),
)
SECRET_PATTERNS = (
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)


def _git_paths(staged: bool) -> list[str]:
    command = ["git", "diff", "--cached", "--name-only"] if staged else ["git", "ls-files"]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    paths = [line for line in output.splitlines() if line]
    if not staged:
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        paths.extend(line for line in untracked.splitlines() if line)
    return sorted(set(paths))


def verify(staged: bool = False) -> list[str]:
    errors: list[str] = []
    for relative_name in _git_paths(staged):
        if any(pattern.search(relative_name) for pattern in BLOCKED_PATHS):
            errors.append(f"Private path is tracked or staged: {relative_name}")
            continue
        path = Path(relative_name)
        if not path.is_file() or path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"Non-text file requires manual review: {relative_name}")
            continue
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            errors.append(f"Possible credential found in: {relative_name}")
        if path.suffix == ".ipynb":
            notebook = json.loads(text)
            dirty_cells = [
                cell
                for cell in notebook.get("cells", [])
                if cell.get("cell_type") == "code" and (cell.get("outputs") or cell.get("execution_count") is not None)
            ]
            if dirty_cells:
                errors.append(f"Notebook output must be stripped: {relative_name}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", action="store_true", help="Inspect staged rather than tracked paths.")
    arguments = parser.parse_args()
    errors = verify(staged=arguments.staged)
    if errors:
        print("Repository safety check failed:", *errors, sep="\n- ")
        return 1
    print("Repository safety check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
