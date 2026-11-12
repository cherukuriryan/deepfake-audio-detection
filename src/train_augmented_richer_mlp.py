import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# Load augmented richer-feature dataset
# --------------------------------------------------

data = np.load(
    "data/augmented_richer_features_8000.npz"
)

X = data["X"]
y = data["y"]
sources = data["source"]
augmented = data["augmented"]
group_ids = data["group_id"]


print("Augmented richer-feature dataset loaded.")
print("X shape:", X.shape)

print("Total rows:", len(X))
print("Real rows:", np.sum(y == "real"))
print("Spoofed rows:", np.sum(y == "spoofed"))

print(
    "Original rows:",
    np.sum(augmented == False)
)

print(
    "Augmented rows:",
    np.sum(augmented == True)
)

print(
    "Unique groups:",
    len(np.unique(group_ids))
)


# --------------------------------------------------
# Build one record per original audio group
# --------------------------------------------------

unique_groups = np.unique(
    group_ids
)

group_labels = []
group_sources = []


for group in unique_groups:

    first_index = np.where(
        group_ids == group
    )[0][0]

    group_labels.append(
        y[first_index]
    )

    group_sources.append(
        sources[first_index]
    )


group_labels = np.array(
    group_labels
)

group_sources = np.array(
    group_sources
)


# Keep source + class combinations balanced
group_stratify = np.array(
    [
        f"{source}_{label}"
        for source, label in zip(
            group_sources,
            group_labels
        )
    ]
)


# --------------------------------------------------
# Split GROUPS first
# --------------------------------------------------

train_groups, test_groups = train_test_split(
    unique_groups,
    test_size=0.20,
    random_state=42,
    stratify=group_stratify
)


# --------------------------------------------------
# Convert group split into row masks
# --------------------------------------------------

train_mask = np.isin(
    group_ids,
    train_groups
)

test_mask = np.isin(
    group_ids,
    test_groups
)


X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]


print("\nGROUP-AWARE SPLIT")
print("--------------------")

print(
    "Training groups:",
    len(train_groups)
)

print(
    "Testing groups:",
    len(test_groups)
)

print(
    "Training rows:",
    len(X_train)
)

print(
    "Testing rows:",
    len(X_test)
)


# Check for leakage
overlap = np.intersect1d(
    train_groups,
    test_groups
)

print(
    "Group overlap:",
    len(overlap)
)


# --------------------------------------------------
# Train same MLP architecture
# --------------------------------------------------

model = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42
    )
)


print("\nTraining augmented richer-feature MLP...")

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
    "models/deepfake_audio_augmented_richer_mlp.pkl"
)


print("\nSaved model:")
print(
    "models/deepfake_audio_augmented_richer_mlp.pkl"
)