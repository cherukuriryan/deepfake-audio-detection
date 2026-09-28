import joblib
import librosa
import numpy as np

model = joblib.load("models/deepfake_audio_model.pkl")


def extract_features(audio_path):
    audio_data, sample_rate = librosa.load(audio_path, sr=16000, mono=True)

    mfcc = librosa.feature.mfcc(
        y=audio_data,
        sr=sample_rate,
        n_mfcc=20
    )

    return np.mean(mfcc, axis=1)


def predict_audio(audio_path):
    features = extract_features(audio_path)

    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]
    confidence = max(probabilities)

    return prediction, confidence * 100