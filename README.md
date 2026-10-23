# Airfoil Self-Noise Prediction (MMA3001)

[Uploading 2493e632-29cb-4e69-992e-3e637616f58a.png…]()


**Goal:** Predicts the scaled sound pressure level (dB) of a NACA 0012 airfoil from five
wind-tunnel inputs, and tests whether a neural network regression is worth using over a
mean baseline and linear regression for higher accuracy.

**Result:** neural network regression with scaled MLP (one hidden layer, 50 neurons) reaches a test RMSE of
3.96 dB and R² of 0.68 on airfoil setups it has never seen, accuracy of 20% higher than linear regression.

## Contents
- [Results](#results)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [How to use each part](#how-to-use-each-part)
- [Method in brief](#method-in-brief)
- [Limitations](#limitations)
- [Data source and license](#data-source-and-license)

## Results

Held-out test set (20% of airfoil setups, never used for tuning):

| Model | MAE (dB) | RMSE (dB) | R² |
| --- | --- | --- | --- |
| Mean baseline | 5.68 | 7.15 | -0.05 |
| Linear regression | 3.69 | 4.98 | 0.49 |
| **Scaled MLP (width 50)** | **3.04** | **3.96** | **0.68** |

## Quick start

**Requirements:** Python 3.10 or later (developed on 3.12).

```
git clone <your-repo-url>
cd MMA3001_Airfoil_self_noise
pip install -r requirements.txt
```

Then, from the `notebooks/` folder:

```
cd notebooks
jupyter notebook
```

Open `airfoil_analysis.ipynb` to reproduce the analysis, or `workflow.ipynb` to
predict for your own data. Run the tests from the project root with
`pytest notebooks`.

### Packages

| Package | Used for |
| --- | --- |
| numpy, pandas | data handling |
| scikit-learn | models, group splitting, metrics |
| matplotlib, seaborn | figures |
| jupyter | running the notebooks |
| pytest | running the tests |
| pdoc (optional) | regenerating the HTML docs |

## Project structure

```
data/                  NASA dataset and an example input file
notebooks/
  airfoil_analysis.ipynb   full analysis (produces the report's figures and tables)
  airfoil_utils.py         shared functions (data checks, split, models, metrics, cost)
  workflow.py              train the final model and predict for new data
  workflow.ipynb           user-friendly notebook version of workflow.py
  test_airfoil.py          tests for airfoil_utils.py
  test_workflow.py         tests for workflow.py
results/               figures used in the report and a sample predictions file
docs/                  generated HTML documentation
```

## How to use each part

### 1. Analysis: `notebooks/airfoil_analysis.ipynb`
Reproduces everything in the report: data checks, the four-method comparison,
10-fold group cross-validation, the width sweep and the final test.
Run all cells from top to bottom. Figures are saved to `results/figures/`.

### 2. Shared code: `notebooks/airfoil_utils.py`
The building blocks used by everything else. Not run directly; imported by the
notebooks and tests. See the generated docs in `docs/` for every function.

### 3. Predicting for your own data: `notebooks/workflow.ipynb`
Open the notebook and point it at a CSV with these columns:
`Frequency_Hz`, `Angle_of_attack_deg`, `Chord_length_m`,
`Free_stream_velocity_m/s`, `Suction_side_displacement_thickness_m`.
An example is in `data/example_input.csv`. The output adds `Predicted_SPL_dB`
and `Outside_training_range`.

The same thing from Python (run from `notebooks/`):

```python
from workflow import train_reference_model, read_user_data, predict_new_data

model, reference, _ = train_reference_model()
new = read_user_data("../data/example_input.csv")
result = predict_new_data(model, new, reference)
```

Invalid input (missing columns, text, NaN or infinite values, negative
frequency, chord length, velocity or thickness) raises a clear error. Inputs
outside the training range are still predicted but flagged.

### 4. Tests: `pytest notebooks`
Run from the project root. Covers the data checks, the group split (no setup in
both train and test), model building, the parameter count formula, and the
workflow's extrapolation flag and error handling.

### 5. Documentation: `docs/`
Open `docs/index.html` in a browser. To regenerate after editing docstrings:

```
cd notebooks
pdoc airfoil_utils workflow -o ../docs
```

## Method in brief

- **Data:** 1,503 NASA wind-tunnel measurements (UCI Airfoil Self-Noise).
- **Group-aware splitting:** rows from the same airfoil setup are strongly
  related, so whole setups stay on one side of every split (train, validation,
  test and every cross-validation fold).
- **Models compared:** mean baseline, linear regression, unscaled MLP, scaled MLP.
- **Tuning:** hidden layer width chosen by group cross-validation; 50 was chosen
  over 100 for similar accuracy at about half the cost.
- **Cost:** parameters, FLOPs, training time and prediction time are reported.

## Limitations

- Trained on one airfoil shape (NACA 0012) only.
- Predictions outside the training range are extrapolations and are flagged.
- The MLP performed badly on one cross-validation fold, so it may be unreliable
  for setups unlike those seen in training.
- Only four methods were compared; other models (e.g. decision trees, SVR) were not studied.

## Data source and license

T. Brooks, D. Pope and M. Marcolini, "Airfoil Self-Noise," UCI Machine Learning
Repository, 1989. https://doi.org/10.24432/C5VW2C

Released under the license in [LICENSE](LICENSE).

_The content is AI-generated, investigated by author._
