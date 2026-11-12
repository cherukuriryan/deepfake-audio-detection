import io
import numpy as np
import librosa
import torch

from datasets import load_dataset, Audio
from transformers import WavLMModel, Wav2Vec2FeatureExtractor


MODEL_NAME = "microsoft/wavlm-base-plus"
MODEL_REVISION = "refs/pr/1"

TARGET_PER_CLASS_PER_DATASET = 250
SAMPLE_RATE = 16000

# Keep the pilot manageable on the Intel Mac.
MAX_SECONDS = 4
MAX_SAMPLES = SAMPLE_RATE * MAX_SECONDS


print("Loading WavLM...")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION
)

wavlm = WavLMModel.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
    use_safetensors=True
)

wavlm.eval()

print("WavLM loaded.")
print("Embedding size:", wavlm.config.hidden_size)


def load_audio_from_example(audio_info):
    """
    Load audio from a Hugging Face Audio(decode=False) example.
    """

    if audio_info.get("bytes") is not None:
        audio_source = io.BytesIO(audio_info["bytes"])

    elif audio_info.get("path") is not None:
        audio_source = audio_info["path"]

    else:
        raise ValueError("Audio example has no bytes or path.")

    audio, sr = librosa.load(
        audio_source,
        sr=SAMPLE_RATE,
        mono=True
    )

    return audio


def prepare_audio(audio):
    """
    Limit every recording to the same maximum duration.
    """

    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]

    return audio.astype(np.float32)


def extract_wavlm_embedding(audio):
    """
    Convert one waveform into one 768-dimensional WavLM embedding.
    """

    audio = prepare_audio(audio)

    inputs = feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = wavlm(
            input_values=inputs.input_values,
            attention_mask=inputs.get("attention_mask")
        )

    # Mean pool across time.
    embedding = (
        outputs.last_hidden_state
        .mean(dim=1)
        .squeeze(0)
        .cpu()
        .numpy()
    )

    return embedding.astype(np.float32)


X = []
y = []
sources = []


def add_example(audio_info, label, source):
    audio = load_audio_from_example(audio_info)

    embedding = extract_wavlm_embedding(audio)

    X.append(embedding)
    y.append(label)
    sources.append(source)


# ============================================================
# DFADD
# ============================================================

print("\nLoading DFADD stream...")

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


for example in dfadd:

    label = str(example["label"]).lower()

    if label == "real":

        if dfadd_real >= TARGET_PER_CLASS_PER_DATASET:
            continue

        add_example(
            example["audio"],
            "real",
            "DFADD"
        )

        dfadd_real += 1

    elif label == "spoofed":

        if dfadd_spoof >= TARGET_PER_CLASS_PER_DATASET:
            continue

        add_example(
            example["audio"],
            "spoofed",
            "DFADD"
        )

        dfadd_spoof += 1

    else:
        continue

    total = dfadd_real + dfadd_spoof

    if total % 25 == 0:
        print(
            f"DFADD: {total}/"
            f"{TARGET_PER_CLASS_PER_DATASET * 2}"
        )

    if (
        dfadd_real >= TARGET_PER_CLASS_PER_DATASET
        and
        dfadd_spoof >= TARGET_PER_CLASS_PER_DATASET
    ):
        break


print(
    f"DFADD complete: "
    f"{dfadd_real} real, "
    f"{dfadd_spoof} spoofed"
)


# ============================================================
# ASVspoof 2019 LA
# ============================================================

print("\nLoading ASVspoof stream...")

asv = load_dataset(
    "Bisher/ASVspoof_2019_LA",
    split="train",
    streaming=True
)

asv = asv.cast_column(
    "audio",
    Audio(decode=False)
)

asv = asv.shuffle(
    seed=42,
    buffer_size=5000
)

asv_real = 0
asv_spoof = 0


for example in asv:

    raw_label = example["key"]

    # Dataset:
    # 0 = bonafide
    # 1 = spoof
    if int(raw_label) == 0:

        if asv_real >= TARGET_PER_CLASS_PER_DATASET:
            continue

        label = "real"

        add_example(
            example["audio"],
            label,
            "ASVspoof2019"
        )

        asv_real += 1

    elif int(raw_label) == 1:

        if asv_spoof >= TARGET_PER_CLASS_PER_DATASET:
            continue

        label = "spoofed"

        add_example(
            example["audio"],
            label,
            "ASVspoof2019"
        )

        asv_spoof += 1

    total = asv_real + asv_spoof

    if total % 25 == 0:
        print(
            f"ASVspoof: {total}/"
            f"{TARGET_PER_CLASS_PER_DATASET * 2}"
        )

    if (
        asv_real >= TARGET_PER_CLASS_PER_DATASET
        and
        asv_spoof >= TARGET_PER_CLASS_PER_DATASET
    ):
        break


print(
    f"ASVspoof complete: "
    f"{asv_real} real, "
    f"{asv_spoof} spoofed"
)


# ============================================================
# Save
# ============================================================

X = np.asarray(
    X,
    dtype=np.float32
)

y = np.asarray(y)

sources = np.asarray(sources)


print("\nFINAL DATASET")
print("-------------------------")
print("Total:", len(X))
print("Real:", np.sum(y == "real"))
print("Spoofed:", np.sum(y == "spoofed"))
print("DFADD:", np.sum(sources == "DFADD"))
print(
    "ASVspoof:",
    np.sum(sources == "ASVspoof2019")
)
print("Feature shape:", X.shape)


output_path = "data/wavlm_features_1000.npz"

np.savez_compressed(
    output_path,
    X=X,
    y=y,
    source=sources
)

print("\nSaved:")
print(output_path)