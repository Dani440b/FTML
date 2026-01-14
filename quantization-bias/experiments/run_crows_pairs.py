import os
import pandas as pd

from src.load_model import load_model
from src.evaluate_crows import evaluate_crows_pairs

MODEL = "FacebookAI/roberta-base"
DATA  = "data/crows_pairs.csv"

DETAILED_OUT = "results/crows_pairs/crows_pairs_detailed.csv"
SUMMARY_OUT  = "results/crows_pairs/crows_pairs_summary.csv"

PRECISIONS = ["fp32", "fp16", "8bit", "4bit", "2bit"]


def run_crows_pairs():
    summary_rows = []

    for precision in PRECISIONS:
        print(f"[CrowS-Pairs] Running {precision}")

        model, tokenizer = load_model(MODEL, precision)

        detailed_df, overall_bias = evaluate_crows_pairs(
            model,
            tokenizer,
            DATA
        )

        detailed_df["precision"] = precision

        if not os.path.exists(DETAILED_OUT):
            detailed_df.to_csv(DETAILED_OUT, index=False)
        else:
            detailed_df.to_csv(DETAILED_OUT, mode="a", header=False, index=False)

        bias_values = detailed_df["stereotype_preferred"]

        summary_rows.append({
            "precision": precision,
            "bias_mean": bias_values.mean(),
            "bias_std": bias_values.std(),
            "overall_bias": overall_bias,
            "num_bias_types": len(bias_values),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_OUT, index=False)

    print("[CrowS-Pairs] Done.")


if __name__ == "__main__":
    run_crows_pairs()
