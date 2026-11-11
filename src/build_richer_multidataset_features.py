import io
import librosa
import numpy as np

from datasets import load_dataset, Audio


TARGET_PER_CLASS_PER_DATASET = 4000
rng = np.random.default_rng(42)


# --------------------------------------------------
# Rich audio feature extraction
# --------------------------------------------------

def extract_features(audio_bytes):

    y, sr = librosa.load(
        io.BytesIO(audio_bytes),
        sr=16000,
        mono=True
    )

    # Prevent extremely short/broken clips
    if len(y) < 2048:
        raise ValueError("Audio clip too short")

    # -----------------------------
    # MFCC features
    # -----------------------------

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20
    )

    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )

    mfcc_std = np.std(
        mfcc,
        axis=1
    )


    # -----------------------------
    # Delta MFCC
    # -----------------------------

    delta = librosa.feature.delta(
        mfcc
    )

    delta_mean = np.mean(
        delta,
        axis=1
    )

    delta_std = np.std(
        delta,
        axis=1
    )


    # -----------------------------
    # Delta-delta MFCC
    # -----------------------------

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    delta2_mean = np.mean(
        delta2,
        axis=1
    )

    delta2_std = np.std(
        delta2,
        axis=1
    )


    # -----------------------------
    # Spectral features
    # -----------------------------

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


    # 120 MFCC-based features
    # + 10 spectral features
    # = 130 total features

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


# --------------------------------------------------
# Storage
# --------------------------------------------------

all_features = []
all_labels = []
all_sources = []


# --------------------------------------------------
# Collect DFADD
# --------------------------------------------------

print("\n================================")
print("COLLECTING DFADD")
print("================================")


dfadd = load_dataset(
    "isjwdu/DFADD",
    split="train",
    streaming=True
)

dfadd = dfadd.cast_column(
    "audio",
    Audio(decode=False)
)

dfadd = dfadd.shuffle(
    seed=42,
    buffer_size=10000
)


dfadd_real = 0
dfadd_spoof = 0
dfadd_skipped = 0


for sample in dfadd:

    label = sample["label"]

    if label == "real":

        if dfadd_real >= TARGET_PER_CLASS_PER_DATASET:
            continue

        converted_label = "real"

    elif label == "spoofed":

        if dfadd_spoof >= TARGET_PER_CLASS_PER_DATASET:
            continue

        converted_label = "spoofed"

    else:
        continue


    try:

        audio_bytes = sample["audio"]["bytes"]

        if audio_bytes is None:
            raise ValueError("Missing audio bytes")

        features = extract_features(
            audio_bytes
        )

        all_features.append(
            features
        )

        all_labels.append(
            converted_label
        )

        all_sources.append(
            "DFADD"
        )


        if converted_label == "real":
            dfadd_real += 1

        else:
            dfadd_spoof += 1


        total = (
            dfadd_real
            + dfadd_spoof
        )

        if total % 100 == 0:
            print(
                f"DFADD: {total}/8000 "
                f"(Real: {dfadd_real}, "
                f"Spoof: {dfadd_spoof})"
            )


        if (
            dfadd_real >= TARGET_PER_CLASS_PER_DATASET
            and
            dfadd_spoof >= TARGET_PER_CLASS_PER_DATASET
        ):
            break


    except Exception as error:

        dfadd_skipped += 1

        if dfadd_skipped <= 5:
            print(
                "DFADD skipped:",
                error
            )


print("\nDFADD complete.")
print("Real:", dfadd_real)
print("Spoofed:", dfadd_spoof)
print("Skipped:", dfadd_skipped)


# --------------------------------------------------
# Collect ASVspoof TRAIN + VALIDATION
# --------------------------------------------------

print("\n================================")
print("COLLECTING ASVSPOOF")
print("================================")


asv_real = 0
asv_spoof = 0
asv_skipped = 0


for split_name in [
    "train",
    "validation"
]:

    print(
        f"\nLoading ASVspoof {split_name}..."
    )


    dataset = load_dataset(
        "Bisher/ASVspoof_2019_LA",
        split=split_name,
        streaming=True
    )

    dataset = dataset.cast_column(
        "audio",
        Audio(decode=False)
    )

    dataset = dataset.shuffle(
        seed=42,
        buffer_size=10000
    )


    for sample in dataset:

        label = sample["key"]


        # 0 = bonafide
        # 1 = spoof

        if label == 0:

            if asv_real >= TARGET_PER_CLASS_PER_DATASET:
                continue

            converted_label = "real"


        elif label == 1:

            if asv_spoof >= TARGET_PER_CLASS_PER_DATASET:
                continue

            converted_label = "spoofed"

        else:
            continue


        try:

            audio_bytes = sample["audio"]["bytes"]

            if audio_bytes is None:
                raise ValueError(
                    "Missing audio bytes"
                )

            features = extract_features(
                audio_bytes
            )


            all_features.append(
                features
            )

            all_labels.append(
                converted_label
            )

            all_sources.append(
                "ASVspoof2019"
            )


            if converted_label == "real":
                asv_real += 1

            else:
                asv_spoof += 1


            total = (
                asv_real
                + asv_spoof
            )

            if total % 100 == 0:
                print(
                    f"ASVspoof: {total}/8000 "
                    f"(Real: {asv_real}, "
                    f"Spoof: {asv_spoof})"
                )


            if (
                asv_real >= TARGET_PER_CLASS_PER_DATASET
                and
                asv_spoof >= TARGET_PER_CLASS_PER_DATASET
            ):
                break


        except Exception as error:

            asv_skipped += 1

            if asv_skipped <= 5:
                print(
                    "ASVspoof skipped:",
                    error
                )


    if (
        asv_real >= TARGET_PER_CLASS_PER_DATASET
        and
        asv_spoof >= TARGET_PER_CLASS_PER_DATASET
    ):
        break


print("\nASVspoof complete.")
print("Real:", asv_real)
print("Spoofed:", asv_spoof)
print("Skipped:", asv_skipped)


# --------------------------------------------------
# Validate counts
# --------------------------------------------------

if (
    dfadd_real < TARGET_PER_CLASS_PER_DATASET
    or
    dfadd_spoof < TARGET_PER_CLASS_PER_DATASET
):

    raise RuntimeError(
        "Could not collect enough DFADD samples."
    )


if (
    asv_real < TARGET_PER_CLASS_PER_DATASET
    or
    asv_spoof < TARGET_PER_CLASS_PER_DATASET
):

    raise RuntimeError(
        "Could not collect enough ASVspoof samples."
    )


# --------------------------------------------------
# Convert to arrays
# --------------------------------------------------

X = np.array(
    all_features,
    dtype=np.float32
)

y = np.array(
    all_labels
)

sources = np.array(
    all_sources
)


# --------------------------------------------------
# Shuffle final dataset
# --------------------------------------------------

indices = np.arange(
    len(X)
)

rng.shuffle(
    indices
)

X = X[indices]
y = y[indices]
sources = sources[indices]


# --------------------------------------------------
# Final information
# --------------------------------------------------

print("\n================================")
print("FINAL RICH FEATURE DATASET")
print("================================")

print("Total:", len(X))

print(
    "Real:",
    np.sum(y == "real")
)

print(
    "Spoofed:",
    np.sum(y == "spoofed")
)

print(
    "DFADD:",
    np.sum(
        sources == "DFADD"
    )
)

print(
    "ASVspoof:",
    np.sum(
        sources == "ASVspoof2019"
    )
)

print(
    "Feature shape:",
    X.shape
)

print(
    "Features per audio clip:",
    X.shape[1]
)


# --------------------------------------------------
# Save
# --------------------------------------------------

np.savez_compressed(
    "data/richer_multidataset_features_16000.npz",
    X=X,
    y=y,
    source=sources
)

print("\nSaved:")
print(
    "data/richer_multidataset_features_16000.npz"
)