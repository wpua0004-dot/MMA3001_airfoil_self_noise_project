"""Predict airfoil self-noise for new data with the final model.

Three functions in this file:

1. train_reference_model
2. read_user_data: reads 
3. predict_new_data

"""

from pathlib import Path

import pandas as pd

from airfoil_utils import (
    DATA_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    flag_out_of_range,
    load_airfoil_data,
    make_final_model,
    regression_metrics,
    split_by_group,
    validate_input_data,
)

PREDICTION_COLUMN = "Predicted_SPL_dB"
FLAG_COLUMN = "Outside_training_range"


def train_reference_model(data_path=DATA_PATH):
    """Train the final model on the reference data and check it on the test set.

    Parameters
    ----------
    data_path : str or Path
        The NASA airfoil data file.

    Returns
    -------
    model : sklearn.pipeline.Pipeline
        The fitted scaled MLP (width 50), trained on train + validation (80%).
    reference_inputs : pandas.DataFrame
        The training inputs, used to decide what counts as "inside the range".
    test_metrics : dict
        MAE, RMSE and R2 on the untouched 20% test set.
    """
    # data checks
    df = load_airfoil_data(data_path)
    validate_input_data(df[FEATURE_COLUMNS])

    # data training
    train_val, test = split_by_group(df)
    model = make_final_model()
    model.fit(train_val[FEATURE_COLUMNS], train_val[TARGET_COLUMN])

    test_metrics = regression_metrics(test[TARGET_COLUMN], model.predict(test[FEATURE_COLUMNS]))
    return model, train_val[FEATURE_COLUMNS], test_metrics


def read_user_data(path):
    """Read the user's CSV file and check that the five input columns exist.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If one or more required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing these columns: {missing}")
    return df


def predict_new_data(model, new_df, reference_inputs):
    """Validate the user's rows and predict the sound pressure level.

    Parameters
    ----------
    model : fitted pipeline
        Output of :func:`train_reference_model`.
    new_df : pandas.DataFrame
        User data containing at least the five input columns.
    reference_inputs : pandas.DataFrame
        Training inputs that define the valid range.

    Returns
    -------
    pandas.DataFrame
        A copy of ``new_df`` with two extra columns: ``Predicted_SPL_dB`` and
        ``Outside_training_range``.

    Raises
    ------
    ValueError
        If the input data fail :func:`airfoil_utils.validate_input_data`.
    """
    # data checks
    X_new = new_df[FEATURE_COLUMNS]
    validate_input_data(X_new)

    result = new_df.copy()
    result[PREDICTION_COLUMN] = model.predict(X_new)
    result[FLAG_COLUMN] = flag_out_of_range(X_new, reference_inputs).values
    return result
