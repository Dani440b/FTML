import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.load_model import load_model
from src.data import load_winobias_type1_pairs
from src.scoring import sentence_logprob
from src.utils import save_results


DETAILED_OUT = "results/winobias/type1_bias_detailed.csv"
SUMMARY_OUT = "results/winobias/type1_bias_summary.csv"


def run_winobias_type1(model_name, precision):
    model, tokenizer = load_model(model_name, precision)
    pairs = load_winobias_type1_pairs("data/winobias")

    rows = []
    pro_wins = 0

    for i, (pro, anti) in enumerate(tqdm(pairs)):
        s_pro = sentence_logprob(model, tokenizer, pro)
        s_anti = sentence_logprob(model, tokenizer, anti)

        win = int(s_pro > s_anti)
        pro_wins += win

        rows.append({
            "pair_id": i,
            "pro_score": s_pro,
            "anti_score": s_anti,
            "pro_wins": win,
        })

    df = pd.DataFrame(rows)

    overall_bias = pro_wins / len(df)

    save_results(
        summary_df=df,
        overall_score=overall_bias,
        precision=precision,
        out_path=DETAILED_OUT,
    )

    delta = df["pro_score"] - df["anti_score"]

    summary_df = pd.DataFrame([{
        "precision": precision,
        "bias_mean": df["pro_wins"].mean(),
        "bias_std": df["pro_wins"].std(),
        "mean_delta": delta.mean(),
        "std_delta": delta.std(),
        "num_pairs": len(df),
    }])

    if not os.path.exists(SUMMARY_OUT):
        summary_df.to_csv(SUMMARY_OUT, index=False)
    else:
        summary_df.to_csv(SUMMARY_OUT, mode="a", header=False, index=False)

    print(f"[WinoBias T1] {precision}: bias={overall_bias:.3f}")


if __name__ == "__main__":
    MODEL = "FacebookAI/roberta-base"
    for p in ["fp32", "fp16", "8bit", "4bit", "2bit"]:
        run_winobias_type1(MODEL, p)
