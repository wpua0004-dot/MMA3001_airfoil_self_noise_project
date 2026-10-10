"""Utility functions for airfoil self-noise project."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

def validate_input_data(X):
    """Ensures no NaN values exist in input features."""
    if np.isnan(X).any():
        raise ValueError("Input data contains NaN values.")
    return True

def regression_metrics(y_true, y_pred):
    """Computes MAE, RMSE, and R2 score."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

def make_mlp(random_state=42, hidden_layer_sizes=(20,), alpha=0.001,
             early_stopping=True):
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=0.01,
        max_iter=2500,
        early_stopping=early_stopping,
        validation_fraction=0.15,
        n_iter_no_change=35,
        tol=1e-5,
        random_state=random_state,
    )