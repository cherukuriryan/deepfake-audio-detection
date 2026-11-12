import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# Load richer 130-feature dataset
# --------------------------------------------------

data = np.load(
    "data/richer_multidataset_features_16000.npz"
)

X = data["X"]
y = data["y"]
sources = data["source"]


print("Richer feature dataset loaded.")
print("X shape:", X.shape)

print("Total:", len(X))
print("Real:", np.sum(y == "real"))
print("Spoofed:", np.sum(y == "spoofed"))

print("DFADD:", np.sum(sources == "DFADD"))
print("ASVspoof:", np.sum(sources == "ASVspoof2019"))


# --------------------------------------------------
# Keep source + class combinations balanced
# --------------------------------------------------

stratify_groups = np.array(
    [
        f"{source}_{label}"
        for source, label in zip(
            sources,
            y
        )
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=stratify_groups
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Features per sample:", X.shape[1])


# --------------------------------------------------
# Extra Trees
# --------------------------------------------------

model = ExtraTreesClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)


print("\nTraining richer-feature Extra Trees...")

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\nRESULTS")
print("--------------------")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions,
        labels=[
            "real",
            "spoofed"
        ]
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=[
            "real",
            "spoofed"
        ],
        zero_division=0
    )
)


# --------------------------------------------------
# Save model
# --------------------------------------------------

joblib.dump(
    model,
    "models/deepfake_audio_richer_extra_trees.pkl"
)


print("\nSaved model:")
print(
    "models/deepfake_audio_richer_extra_trees.pkl"
)