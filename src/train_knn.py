import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load the same 20,000 DFADD MFCC feature set
data = np.load("data/final_mlp_features_20000.npz")

X = data["X"]
y = data["y"]

print("Dataset loaded.")
print("X shape:", X.shape)
print("y shape:", y.shape)


# Same 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# KNN needs feature scaling
model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        n_jobs=-1
    )
)

print("\nTraining KNN...")

model.fit(X_train, y_train)


# Evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nRESULTS")
print("--------------------")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=["real", "spoofed"]
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=["real", "spoofed"],
        zero_division=0
    )
)


# Save model
joblib.dump(
    model,
    "models/deepfake_audio_knn_model.pkl"
)

print("\nSaved model:")
print("models/deepfake_audio_knn_model.pkl")