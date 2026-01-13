from tqdm import tqdm

from load_model import load_model
from data import load_winobias_type1
from scoring import sentence_logprob

def evaluate_winobias(model_name, precision):
    model, tokenizer = load_model(model_name, precision)
    pairs = load_winobias_type1("data/winobias")

    stereotype_wins = 0

    for pro, anti in tqdm(pairs):
        score_pro = sentence_logprob(model, tokenizer, pro)
        score_anti = sentence_logprob(model, tokenizer, anti)

        if score_pro > score_anti:
            stereotype_wins += 1

    bias_score = stereotype_wins / len(pairs)
    print(f"WinoBias Type-1 | {precision}: {bias_score:.3f}")

    return bias_score


if __name__ == "__main__":
    MODEL = "FacebookAI/roberta-base"
    for p in ["fp32", "fp16", "8bit", "4bit", "2bit"]:
        evaluate_winobias(MODEL, p)
