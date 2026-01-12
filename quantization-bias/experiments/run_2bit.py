from src.load_model import load_model
from src.evaluate_crows import evaluate_crows_pairs
from src.utils import save_results
from src.quantize_int2 import quantize_model_int2

MODEL = "FacebookAI/roberta-base"
DATA = "data/crows_pairs.csv"
OUT = "results/bias_scores_2bit.csv"

# Load FP32 model
model, tokenizer = load_model(MODEL, precision="fp32")

# Move model to GPU if available
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Apply simulated 2-bit quantization
model = quantize_model_int2(model)

# Evaluate
summary, overall = evaluate_crows_pairs(model, tokenizer, DATA)

# Save
save_results(summary, overall, "int2_simulated", OUT)
