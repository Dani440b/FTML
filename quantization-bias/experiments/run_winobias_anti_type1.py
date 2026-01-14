import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.load_model import load_model
from src.data import load_winobias_type1_single
from src.scoring import sentence_logprob
from src.utils import save_results


DETAILED_OUT = "results/winobias/type1_anti_detailed.csv"
SUMMARY_OUT  = "results/winobias/type1_anti_summary.csv"


def run_anti_only(model_name, precision):
    model, tokenizer = load_model(model_name, precision)
    sentences = load_winobias_type1_single("data/winobias", "anti")

    rows = []
    scores = []

    for i, s in enumerate(tqdm(sentences)):
        score = sentence_logprob(model, tokenizer, s)
        scores.append(score)

        rows.append({
            "sentence_id": i,
            "logprob": score,
        })

    df = pd.DataFrame(rows)

    mean_lp = float(np.mean(scores))
    std_lp  = float(np.std(scores))

    save_results(
        summary_df=df,
        overall_score=mean_lp,
        precision=precision,
        out_path=DETAILED_OUT,
    )

    summary_df = pd.DataFrame([{
        "precision": precision,
        "mean_logprob": mean_lp,
        "std_logprob": std_lp,
        "num_sentences": len(df),
    }])

    if not os.path.exists(SUMMARY_OUT):
        summary_df.to_csv(SUMMARY_OUT, index=False)
    else:
        summary_df.to_csv(SUMMARY_OUT, mode="a", header=False, index=False)

    print(f"[WinoBias T1 ANTI] {precision}: mean={mean_lp:.3f}, std={std_lp:.3f}")


if __name__ == "__main__":
    MODEL = "FacebookAI/roberta-base"
    for p in ["fp32", "fp16", "8bit", "4bit", "2bit"]:
        run_anti_only(MODEL, p)
