"""Train a NumPy perceptron for synthetic manufacturing quality control."""

import numpy as np

SEED = 42
LEARNING_RATE = 0.1
MAX_EPOCHS = 100

part_ids = np.array([
    "P01", "P02", "P03", "P04", "P05", "P06",
    "P07", "P08", "P09", "P10", "P11", "P12",
])

X = np.array([
    [0.10, 0.0],
    [0.15, 1.0],
    [0.20, 0.0],
    [0.25, 1.0],
    [0.30, 0.0],
    [0.05, 1.0],
    [0.55, 0.0],
    [0.45, 2.0],
    [0.20, 3.0],
    [0.70, 1.0],
    [0.35, 2.0],
    [0.60, 2.0],
])
y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

test_part_ids = np.array(["T01", "T02", "T03", "T04"])
X_test = np.array([
    [0.12, 0.0],
    [0.28, 1.0],
    [0.50, 1.0],
    [0.10, 3.0],
])
y_test = np.array([1, 1, 0, 0])


def predict(features, weights, bias):
    """Return the binary prediction for one inspected part."""
    # Step activation function: scores >= 0 map to 1; scores < 0 map to 0.
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
    scores = X @ weights + bias
    # Step activation function: scores >= 0 map to 1; scores < 0 map to 0.
    predictions = (scores >= 0).astype(int)
    errors = y - predictions
    error_count = np.sum(errors != 0)

    history.append(errors)
    print(
        f"Epoch {epoch + 1:3d} | Errors: {error_count} | "
        f"Weights: {np.round(weights, 3)} | Bias: {bias:.3f}"
    )
    if error_count == 0:
        print(f"\nTraining complete after {epoch + 1} epoch(s).")
        break

    # update weights and bias
    weights += LEARNING_RATE * (X.T @ errors)
    bias += LEARNING_RATE * np.sum(errors)
else:
    raise RuntimeError("The perceptron did not converge within 100 epochs.")

print("\nTesting unseen parts")
scores_test = X_test @ weights + bias
# Step activation function: scores >= 0 map to 1; scores < 0 map to 0.
predictions_test = (scores_test >= 0).astype(int)

for part_id, xi, yi, pi in zip(test_part_ids, X_test, y_test, predictions_test):

    label = "ACCEPT" if pi == 1 else "REJECT"
    expected_label = "ACCEPT" if yi == 1 else "REJECT"
    print(
        f"{part_id}: dimension error={xi[0]:.2f} mm, "
        f"surface defects={int(xi[1])}, score={(weights @ xi + bias):+.3f} "
        f"-> {label} (expected {expected_label})"
    )

print("\nThe dot product is the compact form of:")
print("weight_error * dimension_error_mm + weight_defects * surface_defects")
print(f"Learned weights: {np.round(weights, 3)}")
print(f"Bias: {bias:+.3f}")
