import io
import json
import joblib
import librosa
import numpy as np

from datasets import load_dataset, Audio


# Load your trained MLP model
model = joblib.load("models/deepfake_audio_mlp_model.pkl")


# Load In-the-Wild dataset
dataset = load_dataset(
    "SpeechAntiSpoofingBenchmarks/InTheWild",
    split="test",
    streaming=True
)

dataset = dataset.cast_column("audio", Audio(decode=False))


# Get one sample
sample = next(iter(dataset))

audio_bytes = sample["audio"]["bytes"]

notes = sample["notes"]

if isinstance(notes, str):
    notes = json.loads(notes)


# Decode audio manually
audio_data, sample_rate = librosa.load(
    io.BytesIO(audio_bytes),
    sr=16000,
    mono=True
)


# Extract same 20 MFCC features used by our model
mfcc = librosa.feature.mfcc(
    y=audio_data,
    sr=sample_rate,
    n_mfcc=20
)

features = np.mean(mfcc, axis=1)


# Make prediction
prediction = model.predict([features])[0]
probabilities = model.predict_proba([features])[0]
confidence = np.max(probabilities) * 100


print("File:", sample["path"])
print("Speaker:", notes.get("speaker"))
print("True label:", notes.get("label"))
print("Model prediction:", prediction)
print(f"Confidence: {confidence:.2f}%")