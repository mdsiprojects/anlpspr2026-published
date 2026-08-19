"""Train a NumPy perceptron for synthetic manufacturing quality control."""

import numpy as np

SEED = 42
LEARNING_RATE = 0.1
MAX_EPOCHS = 100

TRAINING_DATA = [
    ("P01", np.array([0.10, 0.0]), 1),
    ("P02", np.array([0.15, 1.0]), 1),
    ("P03", np.array([0.20, 0.0]), 1),
    ("P04", np.array([0.25, 1.0]), 1),
    ("P05", np.array([0.30, 0.0]), 1),
    ("P06", np.array([0.05, 1.0]), 1),
    ("P07", np.array([0.55, 0.0]), 0),
    ("P08", np.array([0.45, 2.0]), 0),
    ("P09", np.array([0.20, 3.0]), 0),
    ("P10", np.array([0.70, 1.0]), 0),
    ("P11", np.array([0.35, 2.0]), 0),
    ("P12", np.array([0.60, 2.0]), 0),
]

TEST_CASES = [
    ("T01", np.array([0.12, 0.0]), 1),
    ("T02", np.array([0.28, 1.0]), 1),
    ("T03", np.array([0.50, 1.0]), 0),
    ("T04", np.array([0.10, 3.0]), 0),
]


def predict(features, weights, bias):
    """Return the binary prediction for one inspected part."""
    return 1 if np.dot(weights, features) + bias >= 0 else 0


np.random.seed(SEED)
weights = np.random.uniform(-1, 1, size=2)
bias = float(np.random.uniform(-1, 1))

print("Synthetic manufacturing quality-control example (NumPy)")
print("Labels: ACCEPT = 1, REJECT = 0")
print("Feature order: [dimension_error_mm, surface_defects]")
print(f"Starting weights: {np.round(weights, 3)}")
print(f"Starting bias: {bias:.3f}\n")

history = []
for epoch in range(MAX_EPOCHS):
    errors = 0
    for part_id, features, target in TRAINING_DATA:
        prediction = predict(features, weights, bias)
        error = target - prediction
        if error != 0:
            errors += 1
            weights = weights + LEARNING_RATE * error * features
            bias = bias + LEARNING_RATE * error

    history.append(errors)
    print(
        f"Epoch {epoch + 1:3d} | Errors: {errors} | "
        f"Weights: {np.round(weights, 3)} | Bias: {bias:.3f}"
    )
    if errors == 0:
        print(f"\nTraining complete after {epoch + 1} epoch(s).")
        break
else:
    raise RuntimeError("The perceptron did not converge within 100 epochs.")

print("\nTesting unseen parts")
for part_id, features, expected in TEST_CASES:
    score = float(np.dot(weights, features) + bias)
    prediction = 1 if score >= 0 else 0
    label = "ACCEPT" if prediction == 1 else "REJECT"
    expected_label = "ACCEPT" if expected == 1 else "REJECT"
    print(
        f"{part_id}: dimension error={features[0]:.2f} mm, "
        f"surface defects={int(features[1])}, score={score:+.3f} "
        f"-> {label} (expected {expected_label})"
    )

print("\nThe dot product is the compact form of:")
print("weight_error * dimension_error_mm + weight_defects * surface_defects")
print(f"Learned weights: {np.round(weights, 3)}")
print(f"Bias: {bias:+.3f}")
