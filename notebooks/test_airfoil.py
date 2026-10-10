import numpy as np
import pytest
from airfoil_utils import regression_metrics, validate_input_data, make_mlp

def test_metrics():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.0, 2.0])
    m = regression_metrics(y_true, y_pred)
    assert m["MAE"] == 0.0

def test_validation():
    with pytest.raises(ValueError):
        validate_input_data(np.array([[1.0, np.nan]]))