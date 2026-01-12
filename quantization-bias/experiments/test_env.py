import torch
import transformers
import bitsandbytes as bnb
from datasets import load_dataset

print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
print("bitsandbytes:", bnb.__version__)

load_dataset(
    "nyu-mll/crows_pairs",
    split="test",
    trust_remote_code=True
)

print("CrowS-Pairs loaded:", len(dataset["test"]))
