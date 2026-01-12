import torch

def sentence_logprob(model, tokenizer, sentence):
    tokens = tokenizer(sentence, return_tensors="pt")

    # Get the device the model is on (works for quantized models)
    device = next(model.parameters()).device
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits

    log_probs = torch.log_softmax(logits, dim=-1)
    input_ids = tokens["input_ids"]

    total_logprob = 0.0
    count = 0

    for i in range(1, input_ids.size(1)):
        token_id = input_ids[0, i]
        total_logprob += log_probs[0, i - 1, token_id].item()
        count += 1

    return total_logprob / max(count, 1)
