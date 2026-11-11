import io
import joblib
import librosa
import numpy as np

from datasets import load_dataset, Audio
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


TARGET_PER_CLASS = 100
TOTAL_TARGET = TARGET_PER_CLASS * 2


# Load richer-feature model
model = joblib.load(
    "models/deepfake_audio_richer_multidataset_mlp.pkl"
)


# Load In-the-Wild
dataset = load_dataset(
    "SpeechAntiSpoofingBenchmarks/InTheWild",
    split="test",
    streaming=True
)

dataset = dataset.cast_column(
    "audio",
    Audio(decode=False)
)

dataset = dataset.shuffle(
    seed=42,
    buffer_size=1000
)


# --------------------------------------------------
# SAME 130-feature extraction used for training
# --------------------------------------------------

def extract_features(audio_bytes):

    y, sr = librosa.load(
        io.BytesIO(audio_bytes),
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


true_labels = []
predicted_labels = []

real_count = 0
spoof_count = 0
skipped = 0


print(
    "Testing richer-feature MLP "
    "on In-the-Wild..."
)


for sample in dataset:

    true_label = sample["label"]

    if true_label == 0 and real_count >= TARGET_PER_CLASS:
        continue

    if true_label == 1 and spoof_count >= TARGET_PER_CLASS:
        continue

    try:

        audio_bytes = sample["audio"]["bytes"]

        if audio_bytes is None:
            raise ValueError("Audio bytes missing")

        features = extract_features(
            audio_bytes
        )

        prediction = model.predict(
            [features]
        )[0]

        if prediction in ["real", "bonafide"]:
            predicted_label = 0

        elif prediction in ["spoof", "spoofed"]:
            predicted_label = 1

        else:
            print(
                "Unknown prediction:",
                prediction
            )
            continue

        true_labels.append(
            true_label
        )

        predicted_labels.append(
            predicted_label
        )

        if true_label == 0:
            real_count += 1
        else:
            spoof_count += 1

        total = real_count + spoof_count

        if total % 10 == 0:
            print(
                f"Tested {total}/{TOTAL_TARGET} "
                f"(Real: {real_count}, "
                f"Spoof: {spoof_count})"
            )

        if (
            real_count >= TARGET_PER_CLASS
            and
            spoof_count >= TARGET_PER_CLASS
        ):
            break

    except Exception as error:

        skipped += 1

        if skipped <= 5:
            print(
                "Skipped sample:",
                error
            )


print("\nRESULTS")
print("--------------------")

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1]
    )
)

print("\nClassification Report:")

print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=[
            "bonafide",
            "spoof"
        ],
        zero_division=0
    )
)

print("\nSamples tested:", len(true_labels))
print("Skipped:", skipped)