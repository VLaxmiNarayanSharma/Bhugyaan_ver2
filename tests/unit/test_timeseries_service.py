import pytest
from app.services.timeseries_service import step_windows, resolve_method_default


# ✅ Test valid yearly window
def test_step_windows_yearly():
    result = step_windows(2020, 2022, "yearly")

    assert len(result) == 3
    assert result[0]["label"] == "2020"
    assert result[-1]["label"] == "2022"


# ✅ Test valid quarterly window
def test_step_windows_quarterly():
    result = step_windows(2021, 2021, "quarterly")

    assert len(result) == 4
    assert result[0]["label"] == "2021-Q1"


# ❌ Invalid interval
def test_step_windows_invalid_interval():
    with pytest.raises(ValueError):
        step_windows(2020, 2022, "monthly")


# ❌ Invalid year range
def test_step_windows_invalid_year():
    with pytest.raises(ValueError):
        step_windows(2025, 2020, "yearly")


# ✅ resolve method
def test_resolve_method_default():
    assert resolve_method_default("rf.pkl") == "rf"
    assert resolve_method_default(" svm ") == "svm"