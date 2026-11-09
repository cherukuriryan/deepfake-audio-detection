import json
from datasets import load_dataset, Audio

dataset = load_dataset(
    "SpeechAntiSpoofingBenchmarks/InTheWild",
    split="test",
    streaming=True
)

dataset = dataset.cast_column("audio", Audio(decode=False))

print("Label feature:", dataset.features["label"])

sample = next(iter(dataset))

print("Columns:", sample.keys())
print("Path:", sample.get("path"))
print("Label field:", sample.get("label"))

notes = sample.get("notes")

if isinstance(notes, str):
    notes = json.loads(notes)

print("Speaker:", notes.get("speaker"))
print("Label from notes:", notes.get("label"))