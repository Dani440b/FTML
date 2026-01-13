from src.load_model import load_model
from src.data import load_winobias_type1
from src.scoring import sentence_logprob

def test_winobias_pipeline_runs():
    model, tokenizer = load_model("FacebookAI/roberta-base", "fp32")
    pairs = load_winobias_type1("data/winobias")

    pro, anti = pairs[0]
    s1 = sentence_logprob(model, tokenizer, pro)
    s2 = sentence_logprob(model, tokenizer, anti)

    assert isinstance(s1, float)
    assert isinstance(s2, float)
