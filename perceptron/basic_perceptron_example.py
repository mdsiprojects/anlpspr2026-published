"""Train a perceptron to accept or reject manufactured parts.

Inputs are measurements supplied by an inspection process:
1. Dimension error in millimetres.
2. Number of visible surface defects.

Labels: REJECT = 0, ACCEPT = 1.
The dataset is synthetic and intended only for teaching.
"""

import random

SEED = 42
LEARNING_RATE = 0.1
MAX_EPOCHS = 100

TRAINING_DATA = [
    ("P01", [0.10, 0], 1),
    ("P02", [0.15, 1], 1),
    ("P03", [0.20, 0], 1),
    ("P04", [0.25, 1], 1),
    ("P05", [0.30, 0], 1),
    ("P06", [0.05, 1], 1),
    ("P07", [0.55, 0], 0),
    ("P08", [0.45, 2], 0),
    ("P09", [0.20, 3], 0),
    ("P10", [0.70, 1], 0),
    ("P11", [0.35, 2], 0),
    ("P12", [0.60, 2], 0),
]

TEST_CASES = [
    ("T01", [0.12, 0], 1),
    ("T02", [0.28, 1], 1),
    ("T03", [0.50, 1], 0),
    ("T04", [0.10, 3], 0),
]


def predict(features, weights, bias):
    """Return the binary prediction for one inspected part."""
    weighted_sum = (
        weights[0] * features[0]
        + weights[1] * features[1]
        + bias
    )
    return 1 if weighted_sum >= 0 else 0


random.seed(SEED)
weights = [random.uniform(-1, 1), random.uniform(-1, 1)]
bias = random.uniform(-1, 1)

print("Synthetic manufacturing quality-control example")
print("Labels: ACCEPT = 1, REJECT = 0")
print("Feature order: [dimension_error_mm, surface_defects]")
print(f"Starting weights: {[round(weight, 3) for weight in weights]}")
print(f"Starting bias: {bias:.3f}\n")

history = []
for epoch in range(MAX_EPOCHS):
    errors = 0
    for part_id, features, target in TRAINING_DATA:
        prediction = predict(features, weights, bias)
        error = target - prediction

        if error != 0:
            errors += 1
            weights[0] += LEARNING_RATE * error * features[0]
            weights[1] += LEARNING_RATE * error * features[1]
            bias += LEARNING_RATE * error

    history.append(errors)
    print(
        f"Epoch {epoch + 1:3d} | Errors: {errors} | "
        f"Weights: [{weights[0]:.3f}, {weights[1]:.3f}] | Bias: {bias:.3f}"
    )
    if errors == 0:
        print(f"\nTraining complete after {epoch + 1} epoch(s).")
        break
else:
    raise RuntimeError("The perceptron did not converge within 100 epochs.")

print("\nTesting unseen parts")
for part_id, features, expected in TEST_CASES:
    prediction = predict(features, weights, bias)
    label = "ACCEPT" if prediction == 1 else "REJECT"
    expected_label = "ACCEPT" if expected == 1 else "REJECT"
    print(
        f"{part_id}: dimension error={features[0]:.2f} mm, "
        f"surface defects={features[1]} -> {label} "
        f"(expected {expected_label})"
    )

print("\nLearned parameter interpretation")
print(f"Dimension-error weight: {weights[0]:+.3f}")
print(f"Surface-defect weight: {weights[1]:+.3f}")
print(f"Bias: {bias:+.3f}")
print("Negative feature weights mean larger measurements push toward REJECT.")
