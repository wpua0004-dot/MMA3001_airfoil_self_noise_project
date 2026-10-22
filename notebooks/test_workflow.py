"""Workflow tests: check that training, validating and predicting work together."""

import numpy as np
import pytest

from airfoil_utils import (
    FEATURE_COLUMNS,
    load_airfoil_data,
)
from workflow import (
    FLAG_COLUMN,
    PREDICTION_COLUMN,
    predict_new_data,
    train_reference_model,
)


@pytest.fixture(scope="module")
def trained():
    """Train the final model once and share it between the tests (saves time)."""
    return train_reference_model()

def test_input_outside_training_range_is_flagged_but_still_predicted(trained):
    """Extrapolation is flagged, not blocked (the limitation stated in the report)."""
    model, reference, _ = trained
    new = load_airfoil_data().iloc[:2][FEATURE_COLUMNS].copy()
    # far above training range
    new.iloc[1, new.columns.get_loc("Frequency_Hz")] = 500000.0   
    result = predict_new_data(model, new, reference)
    assert list(result[FLAG_COLUMN]) == [False, True]
    assert result[PREDICTION_COLUMN].notna().all()


def test_invalid_input_stops_with_a_clear_error(trained):
    """Bad input is rejected by the validation before any prediction is made."""
    model, reference, _ = trained
    bad = load_airfoil_data().iloc[:2][FEATURE_COLUMNS].copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        predict_new_data(model, bad, reference)