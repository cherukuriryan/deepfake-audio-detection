from datasets import load_dataset

dataset = load_dataset(
    "isjwdu/DFADD",
    split="train",
    streaming=True
)

sample = next(iter(dataset))

print(sample)
