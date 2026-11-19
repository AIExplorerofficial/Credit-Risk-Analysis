"""Test fixtures select only real observations in the notebook's saved data splits."""
from __future__ import annotations

import pytest

from backend.analytics import examples_data


@pytest.fixture(scope="session")
def actual_cal_b_application() -> dict:
    rows = examples_data(1)
    if not rows:
        pytest.fail("The notebook cal_B split contains no complete applicant rows")
    return rows[0]["applicant"]
