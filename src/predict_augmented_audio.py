import sys
import joblib
import librosa
import numpy as np


model = joblib.load(
    "models/deepfake_audio_augmented_richer_mlp.pkl"
)


def extract_features(audio_path):

    y, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    if len(y) < 2048:
        raise ValueError("Audio clip too short")

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    delta = librosa.feature.delta(mfcc)

    delta_mean = np.mean(delta, axis=1)
    delta_std = np.std(delta, axis=1)

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    delta2_mean = np.mean(delta2, axis=1)
    delta2_std = np.std(delta2, axis=1)

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    zero_crossing = librosa.feature.zero_crossing_rate(
        y
    )

    rms = librosa.feature.rms(
        y=y
    )

    spectral_features = np.array(
        [
            np.mean(centroid),
            np.std(centroid),

            np.mean(bandwidth),
            np.std(bandwidth),

            np.mean(rolloff),
            np.std(rolloff),

            np.mean(zero_crossing),
            np.std(zero_crossing),

            np.mean(rms),
            np.std(rms)
        ]
    )

    features = np.concatenate(
        [
            mfcc_mean,
            mfcc_std,
            delta_mean,
            delta_std,
            delta2_mean,
            delta2_std,
            spectral_features
        ]
    )

    return features


def predict_audio(audio_path):

    features = extract_features(
        audio_path
    )

    prediction = model.predict(
        [features]
    )[0]

    probabilities = model.predict_proba(
        [features]
    )[0]

    confidence = np.max(
        probabilities
    ) * 100

    return prediction, confidence


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "Usage: python "
            "src/predict_augmented_audio.py "
            "<audio_file>"
        )
        sys.exit(1)

    audio_path = sys.argv[1]

    try:

        prediction, confidence = predict_audio(
            audio_path
        )

        print("\nPrediction Results")
        print("--------------------")
        print("Prediction:", prediction)
        print(
            f"Confidence: {confidence:.2f}%"
        )

    except Exception as error:

        print(
            "Error:",
            error
        )