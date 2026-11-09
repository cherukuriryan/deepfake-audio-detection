import json
from datasets import load_dataset, Audio

dataset = load_dataset(
    "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA",
    split="test",
    streaming=True
)

dataset = dataset.cast_column("audio", Audio(decode=False))

print("Dataset loaded.")
print("Features:", dataset.features)

sample = next(iter(dataset))

print("\nColumns:", sample.keys())
print("Path:", sample["path"])
print("Label field:", sample["label"])

notes = sample["notes"]

if isinstance(notes, str):
    notes = json.loads(notes)

print("Utterance ID:", notes.get("utterance_id"))
print("Speaker ID:", notes.get("speaker_id"))
print("Subset:", notes.get("subset"))