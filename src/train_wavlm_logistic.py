import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# Load WavLM embeddings
# --------------------------------------------------

data = np.load(
    "data/wavlm_features_1000.npz"
)

X = data["X"]
y = data["y"]
sources = data["source"]

print("WavLM dataset loaded.")
print("Shape:", X.shape)
print("Real:", np.sum(y == "real"))
print("Spoofed:", np.sum(y == "spoofed"))


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


# --------------------------------------------------
# Logistic Regression
# --------------------------------------------------

model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


print("\nTraining WavLM Logistic Regression...")

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
    "models/deepfake_audio_wavlm_logistic.pkl"
)

print("\nSaved model:")
print(
    "models/deepfake_audio_wavlm_logistic.pkl"
)