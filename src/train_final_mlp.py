import io
import os
import joblib
import librosa
import numpy as np
import soundfile as sf

from datasets import load_dataset, Audio
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGET_PER_CLASS = 10000
TARGET_SAMPLE_RATE = 16000


dataset = load_dataset(
    "isjwdu/DFADD",
    split="train",
    streaming=True
)

dataset = dataset.cast_column("audio", Audio(decode=False))
dataset = dataset.shuffle(seed=42, buffer_size=10000)


def extract_features(sample):
    audio_bytes = sample["audio"]["bytes"]
    audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))

    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    audio_data = audio_data.astype(float)

    if sample_rate != TARGET_SAMPLE_RATE:
        audio_data = librosa.resample(
            audio_data,
            orig_sr=sample_rate,
            target_sr=TARGET_SAMPLE_RATE
        )
        sample_rate = TARGET_SAMPLE_RATE

    mfcc = librosa.feature.mfcc(
        y=audio_data,
        sr=sample_rate,
        n_mfcc=20
    )

    return np.mean(mfcc, axis=1)


X = []
y = []

real_count = 0
spoofed_count = 0


print("Collecting balanced DFADD samples...")


for sample in dataset:
    label = sample["label"]

    if label == "real" and real_count < TARGET_PER_CLASS:
        try:
            features = extract_features(sample)
        except Exception as e:
            print("Skipping bad real sample:", e)
            continue

        X.append(features)
        y.append(label)
        real_count += 1

    elif label == "spoofed" and spoofed_count < TARGET_PER_CLASS:
        try:
            features = extract_features(sample)
        except Exception as e:
            print("Skipping bad spoofed sample:", e)
            continue

        X.append(features)
        y.append(label)
        spoofed_count += 1

    total = real_count + spoofed_count

    if total > 0 and total % 500 == 0:
        print(
            f"Collected {total}/20000 "
            f"({real_count} real, {spoofed_count} spoofed)"
        )

    if (
        real_count >= TARGET_PER_CLASS
        and spoofed_count >= TARGET_PER_CLASS
    ):
        break


X = np.array(X)
y = np.array(y)


print("\nFinal dataset:")
print("Total samples:", len(y))
print("Real samples:", np.sum(y == "real"))
print("Spoofed samples:", np.sum(y == "spoofed"))
print("Feature shape:", X.shape)


os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)


np.savez(
    "data/final_mlp_features_20000.npz",
    X=X,
    y=y
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(y_train))
print("Testing samples:", len(y_test))


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


print("\nTraining final MLP model...")

model.fit(X_train, y_train)


predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)


print("\nModel accuracy:", accuracy)

print("\nClassification report:")
print(classification_report(y_test, predictions))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))


joblib.dump(
    model,
    "models/deepfake_audio_mlp_model.pkl"
)


print("\nFinal MLP model saved successfully.")
print("Saved to: models/deepfake_audio_mlp_model.pkl")
print("Classes:", model.classes_)