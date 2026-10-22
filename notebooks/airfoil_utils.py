"""Utility functions for the airfoil self-noise project (MMA3001).

Constants defined in this file:

1. FEATURE_COLUMNS
2. TARGET_COLUMN
3. ALL_COLUMNS
4. GROUP_COLUMNS
5. RANDOM_STATE
6. FINAL_WIDTH
7. LINEAR_FLOPS
8. DATA_PATH

Ten functions in this file:

1. load_airfoil_data()
2. validate_input_data()
3. flag_out_of_range()
4. split_by_group()
5. make_mlp()
6. make_final_model()
7. build_models()
8. regression_metrics()
9. mlp_parameters()
10. mlp_flops()

"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Constants (names and units follow Tables 2.1 and 2.2 of the report)
# ----------------------------------------------------------------------------
# Input features
FEATURE_COLUMNS = [
    "Frequency_Hz",
    "Angle_of_attack_deg",
    "Chord_length_m",
    "Free_stream_velocity_m/s",
    "Suction_side_displacement_thickness_m",
]
# Output feature
TARGET_COLUMN = "Scaled_sound_pressure_level_dB"
# Joint input and output feature
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# Split identity
GROUP_COLUMNS = [
    "Angle_of_attack_deg",
    "Chord_length_m",
    "Free_stream_velocity_m/s",
    "Suction_side_displacement_thickness_m",
]

# Random seed
RANDOM_STATE = 42
# Hidden layer width chosen in Section 6 of the report
FINAL_WIDTH = 50
# 5 multiplications + 5 additions (4 sums + 1 intercept)          
LINEAR_FLOPS = 10         

# Define file path for reference data (NASA NACA0012 dataset)
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "airfoil_self_noise.dat"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_airfoil_data(path=DATA_PATH):
    """Load the NASA airfoil self-noise data file.

    Parameters
    ----------
    path : str or Path
        Whitespace-separated file with six columns and no header.

    Returns
    -------
    pandas.DataFrame
        Six columns named as in ``ALL_COLUMNS``.
    """
    return pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLUMNS)


def validate_input_data(df):
    """Check the data with the same checks as Section 2 of the report.

    The checks are:

    1. Row counts: the five input columns must exist and there must be rows.
    2. Data type: the five input values must be numbers (floats).
    3. Missing values: there must be none.
    4. Duplicated rows: there must be none.
    5. Valid ranges: frequency, chord length, free-stream velocity, suction side
       displacement thickness (and the sound pressure level, if it is in the
       data) must be at least zero. 

    Parameters
    ----------
    df : pandas.DataFrame
        Data with the columns in ``FEATURE_COLUMNS`` (the output column
        ``TARGET_COLUMN`` is optional).

    Returns
    -------
    bool
        ``True`` when every check passes.

    Raises
    ------
    ValueError
        If any check fails. The message says which check failed.
    """
    # Check 1: the input columns exist and there is at least one row
    missing_columns = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Check 1 failed: missing column(s) {missing_columns}.")
    if len(df) == 0:
        raise ValueError("Check 1 failed: the data has no rows.")

    # Look at the five inputs, plus the output if it is in the data
    columns = FEATURE_COLUMNS + ([TARGET_COLUMN] if TARGET_COLUMN in df.columns else [])
    data = df[columns]

    # Check 2: every value must be a number (text would break the checks)
    for column in columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"Check 2 failed: the values must be numbers, but {column} contains text.")

    # Check 3: missing values
    if data.isnull().sum().sum() > 0:
        raise ValueError("Check 3 failed: the data contains missing (NaN) values.")

    # Check 4: duplicated rows
    n_duplicates = data.duplicated().sum()
    if n_duplicates > 0:
        raise ValueError(f"Check 4 failed: the data contains {n_duplicates} duplicated row(s).")

    # Check 5: valid ranges (minimum at least zero; angle of attack is not checked)
    for column in columns:
        if column != "Angle_of_attack_deg" and data[column].min() < 0:
            raise ValueError(f"Check 5 failed: {column} contains negative values.")

    return True


def flag_out_of_range(X_new, X_reference):
    """Flag rows whose inputs lie outside the range seen in training.

    The model only covers the input ranges it was trained on, so a prediction
    for a flagged row is an extrapolation and should be treated with caution.

    Parameters
    ----------
    X_new : pandas.DataFrame
        Rows to check (columns ``FEATURE_COLUMNS``).
    X_reference : pandas.DataFrame
        Training inputs that define the valid range (same columns).

    Returns
    -------
    pandas.Series
        ``True`` for each row of ``X_new`` where at least one input is below
        the training minimum or above the training maximum.
    """
    low = X_reference[FEATURE_COLUMNS].min()
    high = X_reference[FEATURE_COLUMNS].max()
    outside = (X_new[FEATURE_COLUMNS] < low) | (X_new[FEATURE_COLUMNS] > high)
    return outside.any(axis=1)


def split_by_group(df, test_size=0.2, random_state=RANDOM_STATE):
    """Split the data into (train + validation) and test sets, by airfoil setup.

    Whole setups go to one side only, so near-identical rows never appear in
    both sets. With the default settings this is the 80 : 20 split used in the
    report; the 80% part is later divided into train and validation.

    Parameters
    ----------
    df : pandas.DataFrame
        Data with the columns in ``ALL_COLUMNS``.
    test_size : float
        Fraction of setups assigned to the test set.
    random_state : int
        Seed of the random split.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(train_val, test)``.
    """
    groups, _ = pd.factorize(df[GROUP_COLUMNS].apply(tuple, axis=1))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_val_idx], df.iloc[test_idx]


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def make_mlp(random_state=RANDOM_STATE, hidden_layer_sizes=(FINAL_WIDTH,), alpha=0.001,
             early_stopping=True):
    """Create the MLP regressor used in the report (Code 3.2).

    One hidden layer with ReLU activation, trained with the Adam optimiser and
    an L2 penalty (``alpha``). With early stopping, a random 15% of the rows is
    set aside and training stops when its error has not improved for 35 rounds.

    Parameters
    ----------
    random_state : int
        Seed for the weight initialisation and the early-stopping hold-out.
    hidden_layer_sizes : tuple of int
        Number of neurons in each hidden layer. Default ``(50,)``.
    alpha : float
        L2 penalty that discourages very large weights.
    early_stopping : bool
        Stop training when the hold-out error stops improving.

    Returns
    -------
    sklearn.neural_network.MLPRegressor
        An unfitted regressor.
    """
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


def make_final_model(width=FINAL_WIDTH):
    """Create the selected model: ``StandardScaler`` followed by the MLP (Code 3.1).

    Parameters
    ----------
    width : int
        Hidden layer width. Default 50, the value chosen in Section 6.

    Returns
    -------
    sklearn.pipeline.Pipeline
        An unfitted scaled MLP pipeline.
    """
    return make_pipeline(StandardScaler(), make_mlp(hidden_layer_sizes=(width,)))


def build_models(width, include_unscaled=False):
    """Return a fresh dictionary of the models compared in the report.

    A new set of objects is created on every call, so a model fitted earlier is
    never reused by mistake.

    Parameters
    ----------
    width : int
        Hidden layer width of the MLP models.
    include_unscaled : bool
        Also include the MLP without the scaler (used only in Section 4).

    Returns
    -------
    dict
        Model name -> unfitted model. The names are ``"Mean baseline"``,
        ``"Linear regression"``, optionally ``"Unscaled MLP"``, and
        ``"Scaled MLP"``.
    """
    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear regression": LinearRegression(),
    }
    if include_unscaled:
        models["Unscaled MLP"] = make_mlp(hidden_layer_sizes=(width,))
    models["Scaled MLP"] = make_final_model(width)
    return models


# ----------------------------------------------------------------------------
# Evaluation and cost
# ----------------------------------------------------------------------------
def regression_metrics(y_true, y_pred):
    """Compute the three accuracy metrics used in the report.

    Parameters
    ----------
    y_true : array-like
        Measured values (dB).
    y_pred : array-like
        Predicted values (dB).

    Returns
    -------
    dict
        ``MAE`` (mean absolute error, dB), ``RMSE`` (root mean square error,
        dB) and ``R2`` (coefficient of determination; 1 is a perfect fit and 0
        is no better than predicting the mean).
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def mlp_parameters(width, n_inputs=5):
    """Count the weights and biases of a one-hidden-layer MLP with one output.

    Input to hidden layer: ``n_inputs * width`` weights and ``width`` biases.
    Hidden layer to output: ``width`` weights and 1 bias.
    
    Parameters
    ----------
    width : int
        Hidden layer width of the MLP models.
    n_inputs : int
        Number of input features.        
    """
    return n_inputs * width + width + (width + 1)


def mlp_flops(width, n_inputs=5):
    """Hand count of arithmetic operations for ONE prediction of the MLP.

    Only multiplications and additions are counted (the activation function and
    the scaling step are not).

    * Hidden layer: per neuron (width) ``n_inputs`` multiplications and ``n_inputs``
      additions (``n_inputs - 1`` to add the products, 1 for the bias).
    * Output layer: ``width`` multiplications and ``width`` additions.

    Parameters
    ----------
    width : int
        Hidden layer width of the MLP models.
    n_inputs : int
        Number of input features.    
    """
    return 2 * n_inputs * width + 2 * width
