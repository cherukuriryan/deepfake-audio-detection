from datasets import load_dataset

dataset = load_dataset(
    "DeepFense/WaveFake",
    split="train",
    streaming=True
)

print("Dataset loaded.")
print("Features:", dataset.features)

sample = next(iter(dataset))

print("\nSample:")
for key, value in sample.items():
    print(f"{key}: {value}")