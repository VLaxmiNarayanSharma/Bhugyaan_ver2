import pytest
from app.services.classification_service import (
    classify_ee_image_for_year,
    classify_ee_image_for_date
)

def test_classify_ee_image_for_year_invalid_geom():
    result, err = classify_ee_image_for_year(
        geom=None,
        year=2023,
        method="rf",
        location_name="test"
    )

    assert result is None
    assert err is not None