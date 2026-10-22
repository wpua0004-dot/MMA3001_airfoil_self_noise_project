"""Tests for the functions in airfoil_utils.py."""

import numpy as np
import pandas as pd
import pytest

from airfoil_utils import (
    FEATURE_COLUMNS,
    GROUP_COLUMNS,
    build_models,
    make_final_model,
    mlp_parameters,
    regression_metrics,
    split_by_group,
    validate_input_data,
    load_airfoil_data,
)


# ---------------------------------------------------------------------------
# Helpers that build small test data
# ---------------------------------------------------------------------------
def good_inputs(n=3):
    """A small valid table of inputs (n rows, the five required columns)."""
    return pd.DataFrame(
        {
            "Frequency_Hz": [800.0, 1000.0, 1250.0][:n],
            "Angle_of_attack_deg": [0.0, 5.4, 12.6][:n],
            "Chord_length_m": [0.3048, 0.1524, 0.0254][:n],
            "Free_stream_velocity_m/s": [71.3, 55.5, 39.6][:n],
            "Suction_side_displacement_thickness_m": [0.0027, 0.0035, 0.0100][:n],
        }
    )

# ---------------------------------------------------------------------------
# validate_input_data (how invalid, missing or unsupported inputs are handled)
# ---------------------------------------------------------------------------
def test_validate_accepts_good_data():
    """Check that the good_inputs pass all the validation criterita."""
    assert validate_input_data(good_inputs()) is True


def test_validate_check_1_rejects_missing_column():
    """Check the result if dropping a column."""
    with pytest.raises(ValueError, match="Check 1"):
        validate_input_data(good_inputs().drop(columns="Chord_length_m"))


def test_validate_check_2_rejects_text():
    """Check the result if value is not a number."""
    bad = good_inputs().astype(object)
    bad.iloc[0, 0] = "loud"
    with pytest.raises(ValueError, match="Check 2"):
        validate_input_data(bad)


def test_validate_check_3_rejects_missing_values():
    """Check the result if value is NaN."""
    bad = good_inputs()
    bad.iloc[1, 2] = np.nan
    with pytest.raises(ValueError, match="Check 3"):
        validate_input_data(bad)


def test_validate_check_4_rejects_duplicated_rows():
    """Check the result if there's duplicated rows."""
    bad = pd.concat([good_inputs(), good_inputs().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Check 4"):
        validate_input_data(bad)


def test_validate_check_5_rejects_negative_values():
    """Check the result if a chord length is negative."""
    bad = good_inputs()
    bad.loc[0, "Chord_length_m"] = -1.0
    with pytest.raises(ValueError, match="Check 5"):
        validate_input_data(bad)

# ---------------------------------------------------------------------------
# split_by_group (no leakage between train and test)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def airfoil_df():
    """Load the NASA airfoil data once and share it with the tests in this file."""
    return load_airfoil_data()

def test_split_uses_every_row_once_and_is_about_80_20(airfoil_df):
    """The split loses no rows, repeats no rows, and gives a test set of about 20%."""
    train_val, test = split_by_group(airfoil_df)
    assert len(train_val) + len(test) == len(airfoil_df)
    assert set(train_val.index).isdisjoint(test.index)
    assert 0.15 < len(test) / len(airfoil_df) < 0.25

# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def test_build_models_names_and_fresh_objects():
    """Ensures no reusing fitted models at a new call."""
    models = build_models(20)
    assert list(models) == ["Mean baseline", "Linear regression", "Scaled MLP"]
    assert "Unscaled MLP" in build_models(20, include_unscaled=True)
    # a new call must give new objects, so a fitted model is never reused by mistake
    assert build_models(20)["Scaled MLP"] is not models["Scaled MLP"]

# ---------------------------------------------------------------------------
# Evaluation and Cost of the model (numbers quoted in the report)
# ---------------------------------------------------------------------------
def test_metrics_known_values():
    """Checked by hand: errors are -1, 0, +1 so MAE = 2/3 and RMSE = sqrt(2/3)."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 4.0])
    scores = regression_metrics(y_true, y_pred)
    assert scores["MAE"] == pytest.approx(2 / 3)
    assert scores["RMSE"] == pytest.approx(np.sqrt(2 / 3))
    assert scores["R2"] == pytest.approx(1 - 2 / 2)   
    
def test_mlp_parameters_match_a_really_fitted_model():
    """The hand formula must agree with the weights scikit-learn actually stores."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.uniform(1, 10, size=(60, 5)), columns=FEATURE_COLUMNS)
    y = X.sum(axis=1)
    fitted = make_final_model(7).fit(X, y)
    mlp = fitted.steps[-1][1]
    counted = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
    assert counted == mlp_parameters(7)
