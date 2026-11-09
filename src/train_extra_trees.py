import joblib
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# Load the same saved DFADD features used for the final MLP
data = np.load("data/final_mlp_features_20000.npz")

X = data["X"]
y = data["y"]

print("Dataset loaded.")
print("X shape:", X.shape)
print("y shape:", y.shape)


# Same 80/20 split style
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# Train Extra Trees model
model = ExtraTreesClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Extra Trees...")

model.fit(X_train, y_train)


# Evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nRESULTS")
print("--------------------")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=["real", "spoofed"]))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=["real", "spoofed"]
    )
)


# Save model
joblib.dump(
    model,
    "models/deepfake_audio_extra_trees_model.pkl"
)

print("\nSaved model:")
print("models/deepfake_audio_extra_trees_model.pkl")