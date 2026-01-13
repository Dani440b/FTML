from src.load_model import load_model
from src.evaluate_crows import evaluate_crows_pairs
from src.utils import save_results

MODEL = "FacebookAI/roberta-base"
DATA  = "data/crows_pairs.csv"
OUT   = "results/crows_pairs_detailed.csv"

PRECISIONS = ["fp32", "fp16", "8bit", "4bit", "2bit"]


def run_crows():
    for precision in PRECISIONS:
        print(f"[CrowS-Pairs] Running {precision}")

        model, tokenizer = load_model(MODEL, precision)
        summary_df, overall_bias = evaluate_crows_pairs(model, tokenizer, DATA)

        save_results(
            summary_df=summary_df,
            overall_score=overall_bias,
            precision=precision,
            out_path=OUT,
        )


if __name__ == "__main__":
    run_crows()
