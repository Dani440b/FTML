from src.load_model import load_model
from src.evaluate_crows import evaluate_crows_pairs
from src.utils import save_results

MODEL = "FacebookAI/roberta-base"
DATA = "data/crows_pairs.csv"
OUT = "results/bias_scores_fp16.csv"

model, tokenizer = load_model(MODEL, precision="FP16")
summary, overall = evaluate_crows_pairs(model, tokenizer, DATA)
save_results(summary, overall, "FP16", OUT)
