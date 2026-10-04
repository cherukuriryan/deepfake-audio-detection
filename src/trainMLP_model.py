import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset, Audio
import io
import soundfile as sf
import librosa
import joblib
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

    mfcc = librosa.feature.mfcc(
        y=audio_data.astype(float),
        sr=sample_rate,
        n_mfcc=20
    )

    return np.mean(mfcc, axis=1)


X = []
y = []

for sample in dataset.take(1000):
    features = extract_features(sample)

    X.append(features)
    y.append(sample["label"])

print("Samples collected:", len(X))
print("Labels:", set(y))
print("Real samples:", y.count("real"))
print("Spoofed samples:", y.count("spoofed"))

real_indices = [i for i, label in enumerate(y) if label == "real"]
spoofed_indices = [i for i, label in enumerate(y) if label == "spoofed"][:len(real_indices)]

balanced_indices = real_indices + spoofed_indices

X = np.array([X[i] for i in balanced_indices])
y = np.array([y[i] for i in balanced_indices])
print("Balanced samples:", len(y))
print("Balanced labels:", np.unique(y, return_counts=True))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
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

model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model accuracy:", accuracy)

print(classification_report(y_test, predictions))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(model, "models/deepfake_audio_mlp_model.pkl")