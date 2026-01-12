import pandas as pd
from tqdm import tqdm
from src.scoring import sentence_logprob

def evaluate_crows_pairs(model, tokenizer, csv_path):
    df = pd.read_csv(csv_path)

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        sent_more = row["sent_more"]
        sent_less = row["sent_less"]

        lp_more = sentence_logprob(model, tokenizer, sent_more)
        lp_less = sentence_logprob(model, tokenizer, sent_less)

        stereotype_preferred = lp_more > lp_less

        results.append({
            "bias_type": row["bias_type"],
            "stereotype_preferred": stereotype_preferred
        })

    results_df = pd.DataFrame(results)

    summary = (
        results_df
        .groupby("bias_type")["stereotype_preferred"]
        .mean()
        .reset_index()
    )

    overall = results_df["stereotype_preferred"].mean()

    return summary, overall
