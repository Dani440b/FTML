import torch

def sentence_logprob(model, tokenizer, sentence):
    tokens = tokenizer(sentence, return_tensors="pt")

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

def score_pronouns(model, tokenizer, masked_sentence, pronouns=("he", "she")):

    tokens = tokenizer(masked_sentence, return_tensors="pt")

    device = next(model.parameters()).device
    tokens = {k: v.to(device) for k, v in tokens.items()}

    mask_id = tokenizer.mask_token_id
    input_ids = tokens["input_ids"]

    # find mask position
    mask_positions = (input_ids == mask_id).nonzero(as_tuple=True)
    if mask_positions[0].numel() != 1:
        raise ValueError(f"Expected exactly one mask, got {mask_positions[0].numel()}")

    mask_index = mask_positions[1].item()

    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits

    scores = {}
    for p in pronouns:
        pid = tokenizer.convert_tokens_to_ids(p)
        scores[p] = logits[0, mask_index, pid].item()

    return scores

