from datasets import load_dataset, Audio
import io
import soundfile as sf
dataset = load_dataset(
    "isjwdu/DFADD",
    split="train",
    streaming=True
)
dataset = dataset.cast_column("audio", Audio(decode=False))

sample = next(iter(dataset))
audio_bytes = sample["audio"]["bytes"]
audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
print("Sample rate:", sample_rate)
print("Audio samples:", len(audio_data))
print("Audio name:", sample["audio_name"])
print("Label:", sample["label"])
print("Split:", sample["split"])
print("Audio path:", sample["audio"]["path"])
