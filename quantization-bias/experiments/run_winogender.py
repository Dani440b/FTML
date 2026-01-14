import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.load_model import load_model
from src.data import load_winogender
from src.preprocessing import mask_gender_pronoun
from src.scoring import score_pronouns
from src.utils import save_results


DETAILED_OUT = "results/winogender/bias_detailed.csv"
SUMMARY_OUT = "results/winogender/bias_summary.csv"


def run_winogender(model_name, precision):
    model, tokenizer = load_model(model_name, precision)
    examples = load_winogender("data/winogender/all_sentences.tsv")

    rows = []

    for i, ex in enumerate(tqdm(examples)):

        sent = ex["sentences"]["she"]

        masked = mask_gender_pronoun(sent, "female", tokenizer)

        scores = score_pronouns(
            model,
            tokenizer,
            masked,
            pronouns=("he", "she")
        )

        delta = scores["he"] - scores["she"]
        win = int(delta > 0)

        rows.append({
            "example_id": i,
            "occupation": ex["occupation"],
            "he_score": scores["he"],
            "she_score": scores["she"],
            "delta": delta,
            "he_wins": win,
        })


    df = pd.DataFrame(rows)

    assert len(df) > 0, "No Winogender examples were evaluated"


    overall_bias = df["he_wins"].mean()

    save_results(
        summary_df=df,
        overall_score=overall_bias,
        precision=precision,
        out_path=DETAILED_OUT,
    )

    summary_df = pd.DataFrame([{
        "precision": precision,
        "bias_mean": df["he_wins"].mean(),
        "bias_std": df["he_wins"].std(),
        "mean_delta": df["delta"].mean(),
        "std_delta": df["delta"].std(),
        "num_examples": len(df),
    }])

    if not os.path.exists(SUMMARY_OUT):
        summary_df.to_csv(SUMMARY_OUT, index=False)
    else:
        summary_df.to_csv(SUMMARY_OUT, mode="a", header=False, index=False)

    print(f"[Winogender] {precision}: bias={overall_bias:.3f}")

if __name__ == "__main__":
    MODEL = "FacebookAI/roberta-base"
    for p in ["fp32", "fp16", "8bit", "4bit", "2bit"]:
        run_winogender(MODEL, p)
