import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import load_dataset, Audio
import io
import soundfile as sf
import librosa

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
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model accuracy:", accuracy)