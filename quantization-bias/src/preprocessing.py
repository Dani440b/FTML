import re

FEMALE_FORMS = ["she", "her", "hers"]
MALE_FORMS = ["he", "him", "his"]

def mask_gender_pronoun(sentence, gender, tokenizer):
    forms = FEMALE_FORMS if gender == "female" else MALE_FORMS

    for p in forms:
        if re.search(rf"\b{p}\b", sentence.lower()):
            return re.sub(rf"\b{p}\b", tokenizer.mask_token, sentence, count=1)

    raise ValueError(f"No {gender} pronoun found in: {sentence}")

