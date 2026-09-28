import io

import joblib
import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


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


features = []
labels = []

for sample in dataset.take(10000):
	features.append(extract_features(sample))
	labels.append(sample["label"])

print("Samples collected:", len(features))
print("Labels:", set(labels))
print("Real samples:", labels.count("real"))
print("Spoofed samples:", labels.count("spoofed"))

real_indices = [index for index, label in enumerate(labels) if label == "real"]
spoofed_indices = [index for index, label in enumerate(labels) if label == "spoofed"]
spoofed_indices = spoofed_indices[:len(real_indices)]
balanced_indices = real_indices + spoofed_indices

X = np.array([features[index] for index in balanced_indices])
y = np.array([labels[index] for index in balanced_indices])
print("Balanced samples:", len(y))
print("Balanced labels:", np.unique(y, return_counts=True))

X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=0.2,
	random_state=42,
	stratify=y
)

model = HistGradientBoostingClassifier(
	max_iter=100,
	learning_rate=0.1,
	random_state=42
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Model accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(model, "models/deepfake_audio_gb_model.pkl")
