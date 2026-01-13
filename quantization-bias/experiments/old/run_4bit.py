from src.load_model import load_model
from src.evaluate_crows import evaluate_crows_pairs
from src.utils import save_results

MODEL = "FacebookAI/roberta-base"
DATA = "data/crows_pairs.csv"
OUT = "results/bias_scores_4bit.csv"

model, tokenizer = load_model(MODEL, precision="4bit")

summary, overall = evaluate_crows_pairs(model, tokenizer, DATA)

save_results(summary, overall, "4bit", OUT)
