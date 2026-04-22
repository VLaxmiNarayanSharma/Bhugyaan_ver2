import numpy as np
from app.services.change_detection_service import compute_change_stats_from_codes


def test_compute_change_stats_basic():
    before = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    after = np.array([[1, 3], [3, 2]], dtype=np.uint8)

    valid_mask = np.array([[True, True], [True, True]])

    result = compute_change_stats_from_codes(before, after, valid_mask)

    assert "transitions" in result
    assert "area_by_transition_key" in result
    assert isinstance(result["transitions"], list)


def test_compute_change_no_change():
    before = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    after = np.array([[1, 1], [1, 1]], dtype=np.uint8)

    valid_mask = np.ones_like(before, dtype=bool)

    result = compute_change_stats_from_codes(before, after, valid_mask)

    assert result["transitions"] == []