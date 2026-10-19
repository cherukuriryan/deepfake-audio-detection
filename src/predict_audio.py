import sys
import joblib
import librosa
import numpy as np

model = joblib.load("models/deepfake_audio_mlp_model.pkl")


def extract_features(audio_path):
    audio_data, sample_rate = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

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

    confidence = np.max(probabilities) * 100

    return prediction, confidence


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict_audio.py <audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]

    prediction, confidence = predict_audio(audio_path)

    print("Prediction:", prediction)
    print(f"Confidence: {confidence:.2f}%")