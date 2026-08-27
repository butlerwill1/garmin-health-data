from __future__ import annotations

import pandas as pd
import pytest

from garmin_daylio_analysis.privacy import assert_public_frame_safe


@pytest.mark.parametrize("column", ["note", "activity_name", "device_id", "latitude"])
def test_privacy_guard_rejects_prohibited_columns(column):
    with pytest.raises(ValueError, match="prohibited"):
        assert_public_frame_safe(pd.DataFrame({column: ["x"]}))
