"""Guards that keep public outputs free of personal source fields."""

from __future__ import annotations

import re

import pandas as pd


PROHIBITED_COLUMN_PATTERNS = (
    r"(^|_)(note|title|name|uuid|identifier|email|profile|device|latitude|longitude)(_|$)",
    r"(^|_)(coordinates?|location)(_|$)",
)


def assert_public_frame_safe(frame: pd.DataFrame) -> None:
    """Raise if a derived table contains a prohibited field name."""
    unsafe = [
        column
        for column in frame.columns
        if any(re.search(pattern, column, flags=re.IGNORECASE) for pattern in PROHIBITED_COLUMN_PATTERNS)
    ]
    if unsafe:
        raise ValueError(f"Derived table contains prohibited columns: {unsafe}")
