import librosa
import numpy as np

from sklearn.preprocessing import StandardScaler


AUDIO_PATH = "test_audio/real_test.wav"
DATA_PATH = "data/richer_multidataset_features_16000.npz"


# --------------------------------------------------
# Feature names
# --------------------------------------------------

feature_names = []

for i in range(1, 21):
    feature_names.append(f"MFCC {i} mean")

for i in range(1, 21):
    feature_names.append(f"MFCC {i} std")

for i in range(1, 21):
    feature_names.append(f"Delta MFCC {i} mean")

for i in range(1, 21):
    feature_names.append(f"Delta MFCC {i} std")

for i in range(1, 21):
    feature_names.append(f"Delta2 MFCC {i} mean")

for i in range(1, 21):
    feature_names.append(f"Delta2 MFCC {i} std")

feature_names += [
    "Spectral centroid mean",
    "Spectral centroid std",
    "Spectral bandwidth mean",
    "Spectral bandwidth std",
    "Spectral rolloff mean",
    "Spectral rolloff std",
    "Zero crossing rate mean",
    "Zero crossing rate std",
    "RMS energy mean",
    "RMS energy std"
]


# --------------------------------------------------
# Same 130-feature extraction
# --------------------------------------------------

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

    return np.concatenate(
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


# --------------------------------------------------
# Load training features
# --------------------------------------------------

data = np.load(DATA_PATH)

X = data["X"]
y = data["y"]
sources = data["source"]


print("Training dataset loaded.")
print("Shape:", X.shape)

print(
    "Real:",
    np.sum(y == "real")
)

print(
    "Spoofed:",
    np.sum(y == "spoofed")
)


# --------------------------------------------------
# Extract your recording
# --------------------------------------------------

target = extract_features(
    AUDIO_PATH
)

print("\nReal microphone recording:")
print("Feature count:", len(target))


# --------------------------------------------------
# Standardize everything
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

target_scaled = scaler.transform(
    [target]
)[0]


# --------------------------------------------------
# Class centroids
# --------------------------------------------------

real_mask = y == "real"
spoof_mask = y == "spoofed"

real_centroid = np.mean(
    X_scaled[real_mask],
    axis=0
)

spoof_centroid = np.mean(
    X_scaled[spoof_mask],
    axis=0
)


distance_real = np.linalg.norm(
    target_scaled - real_centroid
)

distance_spoof = np.linalg.norm(
    target_scaled - spoof_centroid
)


print("\nCLASS CENTROID DISTANCES")
print("-------------------------")

print(
    f"Distance to REAL centroid: "
    f"{distance_real:.3f}"
)

print(
    f"Distance to SPOOF centroid: "
    f"{distance_spoof:.3f}"
)


if distance_real < distance_spoof:
    print(
        "\nRecording is closer to the REAL "
        "training centroid."
    )
else:
    print(
        "\nRecording is closer to the SPOOF "
        "training centroid."
    )


# --------------------------------------------------
# Source + class centroids
# --------------------------------------------------

print("\nSOURCE + CLASS DISTANCES")
print("-------------------------")

for source_name in [
    "DFADD",
    "ASVspoof2019"
]:

    for label_name in [
        "real",
        "spoofed"
    ]:

        mask = (
            (sources == source_name)
            &
            (y == label_name)
        )

        centroid = np.mean(
            X_scaled[mask],
            axis=0
        )

        distance = np.linalg.norm(
            target_scaled - centroid
        )

        print(
            f"{source_name:14} "
            f"{label_name:8} "
            f"{distance:.3f}"
        )


# --------------------------------------------------
# Nearest training samples
# --------------------------------------------------

distances = np.linalg.norm(
    X_scaled - target_scaled,
    axis=1
)

nearest_indices = np.argsort(
    distances
)[:20]


print("\n20 NEAREST TRAINING SAMPLES")
print("----------------------------")

nearest_real = 0
nearest_spoof = 0


for rank, index in enumerate(
    nearest_indices,
    start=1
):

    label = y[index]
    source = sources[index]

    if label == "real":
        nearest_real += 1
    else:
        nearest_spoof += 1

    print(
        f"{rank:2}. "
        f"{source:14} "
        f"{label:8} "
        f"distance={distances[index]:.3f}"
    )


print("\nNearest-neighbor summary:")
print("Real:", nearest_real)
print("Spoofed:", nearest_spoof)


# --------------------------------------------------
# Features pushing toward spoof
# --------------------------------------------------

real_feature_distance = (
    target_scaled - real_centroid
) ** 2

spoof_feature_distance = (
    target_scaled - spoof_centroid
) ** 2


# Positive score =
# closer to spoof than real for that feature
spoof_score = (
    real_feature_distance
    - spoof_feature_distance
)

top_spoof_features = np.argsort(
    spoof_score
)[::-1]


print("\nTOP FEATURES MAKING IT LOOK SPOOF-LIKE")
print("---------------------------------------")

shown = 0

for index in top_spoof_features:

    if spoof_score[index] <= 0:
        continue

    print(
        f"{feature_names[index]:30} "
        f"score={spoof_score[index]:.3f}"
    )

    shown += 1

    if shown >= 15:
        break


# --------------------------------------------------
# Features pushing toward real
# --------------------------------------------------

top_real_features = np.argsort(
    spoof_score
)


print("\nTOP FEATURES MAKING IT LOOK REAL-LIKE")
print("--------------------------------------")

shown = 0

for index in top_real_features:

    if spoof_score[index] >= 0:
        continue

    print(
        f"{feature_names[index]:30} "
        f"score={spoof_score[index]:.3f}"
    )

    shown += 1

    if shown >= 15:
        break