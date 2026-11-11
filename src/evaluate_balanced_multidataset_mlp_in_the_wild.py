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


# Load balanced multi-dataset MLP
model = joblib.load(
    "models/deepfake_audio_balanced_multidataset_mlp.pkl"
)


# Load In-the-Wild test data
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


def extract_features(audio_bytes):
    audio_data, sample_rate = librosa.load(
        io.BytesIO(audio_bytes),
        sr=16000,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=audio_data,
        sr=sample_rate,
        n_mfcc=20
    )

    return np.mean(mfcc, axis=1)


true_labels = []
predicted_labels = []

real_count = 0
spoof_count = 0
skipped = 0


print(
    "Testing balanced multi-dataset MLP "
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
            and spoof_count >= TARGET_PER_CLASS
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