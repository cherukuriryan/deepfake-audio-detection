import io
import librosa
import numpy as np

from datasets import load_dataset, Audio


TARGET_PER_CLASS_PER_DATASET = 1000

rng = np.random.default_rng(42)


# --------------------------------------------------
# Load audio bytes
# --------------------------------------------------

def load_audio(audio_bytes):

    y, sr = librosa.load(
        io.BytesIO(audio_bytes),
        sr=16000,
        mono=True
    )

    if len(y) < 2048:
        raise ValueError("Audio clip too short")

    return y, sr


# --------------------------------------------------
# Realistic audio augmentation
# --------------------------------------------------

def augment_audio(y, sr):

    augmented = y.copy()

    # Random volume change: -4 dB to +4 dB
    gain_db = rng.uniform(-4.0, 4.0)

    augmented = augmented * (
        10 ** (gain_db / 20)
    )


    # Add mild background noise
    signal_power = np.mean(
        augmented ** 2
    )

    if signal_power > 0:

        snr_db = rng.uniform(
            25.0,
            40.0
        )

        noise_power = (
            signal_power
            / (10 ** (snr_db / 10))
        )

        noise = rng.normal(
            0,
            np.sqrt(noise_power),
            len(augmented)
        )

        augmented = (
            augmented
            + noise
        )


    # Sometimes simulate lower-quality
    # microphone / resampling conditions
    if rng.random() < 0.50:

        temporary_sr = int(
            rng.choice(
                [8000, 12000, 22050]
            )
        )

        augmented = librosa.resample(
            augmented,
            orig_sr=sr,
            target_sr=temporary_sr
        )

        augmented = librosa.resample(
            augmented,
            orig_sr=temporary_sr,
            target_sr=sr
        )


    # Prevent clipping
    max_value = np.max(
        np.abs(augmented)
    )

    if max_value > 1.0:

        augmented = (
            augmented
            / max_value
        )


    return augmented.astype(
        np.float32
    )


# --------------------------------------------------
# Same 130 features as current best model
# --------------------------------------------------

def extract_features(y, sr):

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


    centroid = (
        librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )
    )

    bandwidth = (
        librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr
        )
    )

    rolloff = (
        librosa.feature.spectral_rolloff(
            y=y,
            sr=sr
        )
    )

    zero_crossing = (
        librosa.feature.zero_crossing_rate(
            y
        )
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


    if not np.all(
        np.isfinite(features)
    ):
        raise ValueError(
            "Invalid feature values"
        )


    return features.astype(
        np.float32
    )


# --------------------------------------------------
# Storage
# --------------------------------------------------

all_features = []
all_labels = []
all_sources = []
all_augmented = []
all_group_ids = []


def save_original_and_augmented(
    y,
    sr,
    label,
    source,
    group_id
):

    # Original
    original_features = (
        extract_features(
            y,
            sr
        )
    )

    all_features.append(
        original_features
    )

    all_labels.append(
        label
    )

    all_sources.append(
        source
    )

    all_augmented.append(
        False
    )

    all_group_ids.append(
        group_id
    )


    # Augmented copy
    augmented_y = augment_audio(
        y,
        sr
    )

    augmented_features = (
        extract_features(
            augmented_y,
            sr
        )
    )

    all_features.append(
        augmented_features
    )

    all_labels.append(
        label
    )

    all_sources.append(
        source
    )

    all_augmented.append(
        True
    )

    # SAME group ID
    all_group_ids.append(
        group_id
    )


# --------------------------------------------------
# DFADD
# --------------------------------------------------

print("\n==============================")
print("COLLECTING DFADD")
print("==============================")


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
dfadd_group_number = 0


for sample in dfadd:

    label = sample["label"]


    if label == "real":

        if (
            dfadd_real
            >= TARGET_PER_CLASS_PER_DATASET
        ):
            continue

        converted_label = "real"


    elif label == "spoofed":

        if (
            dfadd_spoof
            >= TARGET_PER_CLASS_PER_DATASET
        ):
            continue

        converted_label = "spoofed"


    else:
        continue


    try:

        audio_bytes = (
            sample["audio"]["bytes"]
        )

        if audio_bytes is None:
            raise ValueError(
                "Missing audio bytes"
            )


        y, sr = load_audio(
            audio_bytes
        )


        group_id = (
            f"DFADD_"
            f"{dfadd_group_number}"
        )


        save_original_and_augmented(
            y,
            sr,
            converted_label,
            "DFADD",
            group_id
        )


        dfadd_group_number += 1


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
            dfadd_real
            >= TARGET_PER_CLASS_PER_DATASET
            and
            dfadd_spoof
            >= TARGET_PER_CLASS_PER_DATASET
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
# ASVspoof train + validation
# --------------------------------------------------

print("\n==============================")
print("COLLECTING ASVSPOOF")
print("==============================")


asv_real = 0
asv_spoof = 0
asv_skipped = 0
asv_group_number = 0


for split_name in [
    "train",
    "validation"
]:

    print(
        f"\nLoading ASVspoof "
        f"{split_name}..."
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


        if label == 0:

            if (
                asv_real
                >= TARGET_PER_CLASS_PER_DATASET
            ):
                continue

            converted_label = "real"


        elif label == 1:

            if (
                asv_spoof
                >= TARGET_PER_CLASS_PER_DATASET
            ):
                continue

            converted_label = "spoofed"


        else:
            continue


        try:

            audio_bytes = (
                sample["audio"]["bytes"]
            )

            if audio_bytes is None:
                raise ValueError(
                    "Missing audio bytes"
                )


            y, sr = load_audio(
                audio_bytes
            )


            group_id = (
                f"ASV_"
                f"{asv_group_number}"
            )


            save_original_and_augmented(
                y,
                sr,
                converted_label,
                "ASVspoof2019",
                group_id
            )


            asv_group_number += 1


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
                    f"ASVspoof: "
                    f"{total}/8000 "
                    f"(Real: {asv_real}, "
                    f"Spoof: {asv_spoof})"
                )


            if (
                asv_real
                >= TARGET_PER_CLASS_PER_DATASET
                and
                asv_spoof
                >= TARGET_PER_CLASS_PER_DATASET
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
        asv_real
        >= TARGET_PER_CLASS_PER_DATASET
        and
        asv_spoof
        >= TARGET_PER_CLASS_PER_DATASET
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
    dfadd_real
    < TARGET_PER_CLASS_PER_DATASET
    or
    dfadd_spoof
    < TARGET_PER_CLASS_PER_DATASET
):
    raise RuntimeError(
        "Not enough DFADD samples."
    )


if (
    asv_real
    < TARGET_PER_CLASS_PER_DATASET
    or
    asv_spoof
    < TARGET_PER_CLASS_PER_DATASET
):
    raise RuntimeError(
        "Not enough ASVspoof samples."
    )


# --------------------------------------------------
# Convert arrays
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

augmented = np.array(
    all_augmented
)

group_ids = np.array(
    all_group_ids
)


# --------------------------------------------------
# Final information
# --------------------------------------------------

print("\n================================")
print("FINAL AUGMENTED FEATURE DATASET")
print("================================")

print("Total rows:", len(X))

print(
    "Original clips:",
    np.sum(
        augmented == False
    )
)

print(
    "Augmented clips:",
    np.sum(
        augmented == True
    )
)

print(
    "Real rows:",
    np.sum(
        y == "real"
    )
)

print(
    "Spoofed rows:",
    np.sum(
        y == "spoofed"
    )
)

print(
    "DFADD rows:",
    np.sum(
        sources == "DFADD"
    )
)

print(
    "ASVspoof rows:",
    np.sum(
        sources == "ASVspoof2019"
    )
)

print(
    "Unique original groups:",
    len(
        np.unique(group_ids)
    )
)

print(
    "Feature shape:",
    X.shape
)


# --------------------------------------------------
# Save
# --------------------------------------------------

np.savez_compressed(
    "data/augmented_richer_features_32000.npz",
    X=X,
    y=y,
    source=sources,
    augmented=augmented,
    group_id=group_ids
)


print("\nSaved:")
print(
    "data/augmented_richer_features_32000.npz"
)