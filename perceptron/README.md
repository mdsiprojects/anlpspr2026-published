# Perceptron Learning Through Manufacturing Quality Control

This beginner-friendly example shows how a perceptron learns a binary decision boundary. It classifies inspected manufactured parts as **ACCEPT** (`1`) or **REJECT** (`0`) from two measurements supplied directly by an inspection process.

The dataset is realistic synthetic teaching data. It was not collected from a factory and must not be used to make real production or safety decisions.

## Features and labels

The feature order is always:

```text
[dimension_error_mm, surface_defects]
```

| Feature | Meaning | Unit |
|---|---|---|
| `dimension_error_mm` | Absolute deviation from the target dimension | Millimetres |
| `surface_defects` | Number of visible surface defects | Count |

The labels are:

- `ACCEPT = 1`
- `REJECT = 0`

Larger dimension errors and more surface defects should generally push a part toward rejection. The perceptron discovers this direction by learning its weights from labelled observations.

## Training data

| Part | Dimension error (mm) | Surface defects | Label |
|---|---:|---:|---|
| P01 | 0.10 | 0 | ACCEPT (`1`) |
| P02 | 0.15 | 1 | ACCEPT (`1`) |
| P03 | 0.20 | 0 | ACCEPT (`1`) |
| P04 | 0.25 | 1 | ACCEPT (`1`) |
| P05 | 0.30 | 0 | ACCEPT (`1`) |
| P06 | 0.05 | 1 | ACCEPT (`1`) |
| P07 | 0.55 | 0 | REJECT (`0`) |
| P08 | 0.45 | 2 | REJECT (`0`) |
| P09 | 0.20 | 3 | REJECT (`0`) |
| P10 | 0.70 | 1 | REJECT (`0`) |
| P11 | 0.35 | 2 | REJECT (`0`) |
| P12 | 0.60 | 2 | REJECT (`0`) |

The observations are linearly separable, so a single perceptron can learn a boundary that classifies every training part correctly.

## Unseen test parts

These parts are not used to update the weights:

| Part | Dimension error (mm) | Surface defects | Expected label |
|---|---:|---:|---|
| T01 | 0.12 | 0 | ACCEPT (`1`) |
| T02 | 0.28 | 1 | ACCEPT (`1`) |
| T03 | 0.50 | 1 | REJECT (`0`) |
| T04 | 0.10 | 3 | REJECT (`0`) |

## Learning sequence

Work through the notebooks in order:

| Notebook | Purpose |
|---|---|
| `01_perceptron_from_scratch.ipynb` | Build the weighted sum, step activation function, and update rule using standard Python |
| `02_perceptron_with_numpy.ipynb` | Express the same calculation with NumPy arrays and dot products |
| `03_visualising_the_perceptron.ipynb` | Plot the data, learned regions, boundary evolution, errors, predictions, and tolerance-band limitation |

The standalone scripts provide the same first two implementations:

| Script | Purpose |
|---|---|
| `basic_perceptron_example.py` | Dependency-free implementation with explicit arithmetic |
| `basic_perceptron_example_v2.py` | NumPy implementation with vector operations |

## Environment setup

Use the course virtual environment from the repository root.

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
source venv/bin/activate
python -m pip install -r requirements.txt
```

No dependency beyond the existing course requirements is needed.

## Run the scripts

From the repository root:

```powershell
python .\perceptron\basic_perceptron_example.py
python .\perceptron\basic_perceptron_example_v2.py
```

Both scripts use random seed `42`, learning rate `0.1`, and at most 100 epochs. They stop early when an epoch has zero errors.

## Run the notebooks

Open the `.ipynb` files in VS Code or Jupyter, select the repository `venv` as the Python kernel, and choose **Run All**. If the environment is not listed as a kernel, register it once:

```powershell
python -m ipykernel install --user --name anlpspr2026 --display-name "ANLP Spring 2026"
```

The notebooks are stored without execution output so every student starts from a clean, reproducible state.

## How the perceptron works

For one inspected part, the perceptron calculates:

The **step activation function** converts the numerical score into a binary
class: scores at or above zero map to `1`, while scores below zero map to `0`.

```text
score = w_error * dimension_error_mm + w_defects * surface_defects + bias
# Step activation function: scores >= 0 map to 1; scores < 0 map to 0.
prediction = 1 if score >= 0 else 0
```

It compares the prediction with the known label:

```text
error = target - prediction
```

When the prediction is wrong, it updates each weight:

```text
weight = weight + learning_rate * error * feature
bias = bias + learning_rate * error
```

One pass through all training observations is an **epoch**. Training stops when a complete epoch has no errors.

## Interpreting the learned model

The two learned feature weights should be negative:

- A larger dimension error lowers the score and pushes the prediction toward `REJECT`.
- More surface defects lower the score and push the prediction toward `REJECT`.
- The bias shifts the location of the acceptance boundary.

The visualisation notebook shows the same relationship geometrically. The learned line divides the two-dimensional measurement space into acceptance and rejection regions.

## Limitations

- The data is deliberately small, synthetic, and linearly separable.
- A perceptron can learn only one linear boundary.
- Real inspection decisions use calibrated equipment, domain tolerances, uncertainty analysis, and safety controls.
- Raw part length illustrates a representation limitation: acceptable values may occupy a middle tolerance band, while both undersized and oversized values are rejected. A single threshold cannot isolate that middle interval.
- Converting raw length to `dimension_error_mm = abs(measured_length - target_length)` creates the one-sided measurement used in this lesson. This transformation is introduced only after the basic perceptron is understood.

## Learning objectives

After completing the examples, students should be able to:

1. Calculate a perceptron's weighted sum and binary prediction.
2. Explain the roles of weights, bias, learning rate, and epochs.
3. Apply the online perceptron update rule after a mistake.
4. Interpret a learned linear decision boundary.
5. Explain why feature representation and linear separability matter.
