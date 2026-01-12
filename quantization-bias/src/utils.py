import pandas as pd
import os

def save_results(summary_df, overall_score, precision, out_path):
    summary_df["precision"] = precision
    summary_df["overall_bias"] = overall_score

    if not os.path.exists(out_path):
        summary_df.to_csv(out_path, index=False)
    else:
        summary_df.to_csv(out_path, mode="a", header=False, index=False)

