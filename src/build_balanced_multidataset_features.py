import io
import librosa
import numpy as np

from datasets import load_dataset, Audio


DFADD_PER_CLASS = 4000
ASV_PER_CLASS = 4000

rng = np.random.default_rng(42)


# --------------------------------------------------
# Load existing DFADD features
# --------------------------------------------------

print("Loading existing DFADD features...")

dfadd = np.load("data/final_mlp_features_20000.npz")

X_dfadd_all = dfadd["X"]
y_dfadd_all = dfadd["y"]


real_indices = np.where(y_dfadd_all == "real")[0]
spoof_indices = np.where(y_dfadd_all == "spoofed")[0]


selected_real = rng.choice(
    real_indices,
    DFADD_PER_CLASS,
    replace=False
)

selected_spoof = rng.choice(
    spoof_indices,
    DFADD_PER_CLASS,
    replace=False
)


selected_dfadd = np.concatenate(
    [selected_real, selected_spoof]
)

rng.shuffle(selected_dfadd)


X_dfadd = X_dfadd_all[selected_dfadd]
y_dfadd = y_dfadd_all[selected_dfadd]


print("DFADD selected:", len(X_dfadd))
print("DFADD real:", np.sum(y_dfadd == "real"))
print("DFADD spoofed:", np.sum(y_dfadd == "spoofed"))


# --------------------------------------------------
# ASVspoof feature extraction
# --------------------------------------------------

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


asv_features = []
asv_labels = []

real_count = 0
spoof_count = 0
skipped = 0


# --------------------------------------------------
# Use TRAIN + VALIDATION only
# Evaluation/test stays untouched
# --------------------------------------------------

for split_name in ["train", "validation"]:

    print(f"\nLoading ASVspoof {split_name} split...")

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

        if label == 0 and real_count >= ASV_PER_CLASS:
            continue

        if label == 1 and spoof_count >= ASV_PER_CLASS:
            continue

        try:

            audio_bytes = sample["audio"]["bytes"]

            if audio_bytes is None:
                raise ValueError("Audio bytes missing")

            features = extract_features(audio_bytes)

            if label == 0:
                asv_labels.append("real")
                real_count += 1

            elif label == 1:
                asv_labels.append("spoofed")
                spoof_count += 1

            else:
                continue

            asv_features.append(features)

            total = real_count + spoof_count

            if total % 100 == 0:
                print(
                    f"ASVspoof collected: {total}/8000 "
                    f"(Real: {real_count}, Spoofed: {spoof_count})"
                )

            if (
                real_count >= ASV_PER_CLASS
                and spoof_count >= ASV_PER_CLASS
            ):
                break

        except Exception as error:

            skipped += 1

            if skipped <= 5:
                print("Skipped sample:", error)

    if (
        real_count >= ASV_PER_CLASS
        and spoof_count >= ASV_PER_CLASS
    ):
        break


print("\nASVspoof collection complete.")
print("Real:", real_count)
print("Spoofed:", spoof_count)
print("Skipped:", skipped)


if (
    real_count < ASV_PER_CLASS
    or spoof_count < ASV_PER_CLASS
):
    raise RuntimeError(
        "Could not collect 4000 real and 4000 spoof ASVspoof samples."
    )


X_asv = np.array(asv_features)
y_asv = np.array(asv_labels)


# --------------------------------------------------
# Combine equal amounts from both datasets
# --------------------------------------------------

X_combined = np.concatenate(
    [X_dfadd, X_asv],
    axis=0
)

y_combined = np.concatenate(
    [y_dfadd, y_asv],
    axis=0
)

sources = np.concatenate(
    [
        np.full(len(X_dfadd), "DFADD"),
        np.full(len(X_asv), "ASVspoof2019")
    ]
)


# Shuffle final dataset
indices = np.arange(len(X_combined))
rng.shuffle(indices)

X_combined = X_combined[indices]
y_combined = y_combined[indices]
sources = sources[indices]


print("\nFINAL BALANCED MULTI-DATASET SET")
print("--------------------------------")

print("Total:", len(X_combined))
print("Real:", np.sum(y_combined == "real"))
print("Spoofed:", np.sum(y_combined == "spoofed"))

print("DFADD:", np.sum(sources == "DFADD"))
print(
    "ASVspoof:",
    np.sum(sources == "ASVspoof2019")
)

print("Feature shape:", X_combined.shape)


# --------------------------------------------------
# Save
# --------------------------------------------------

np.savez(
    "data/balanced_multidataset_features_16000.npz",
    X=X_combined,
    y=y_combined,
    source=sources
)

print("\nSaved:")
print(
    "data/balanced_multidataset_features_16000.npz"
)