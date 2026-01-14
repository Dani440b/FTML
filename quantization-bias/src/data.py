from pathlib import Path
from collections import defaultdict

def _read_winobias_file(path):
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            sent = line.strip().split("\t")[0]
            sentences.append(sent)
    return sentences


def load_winobias_type1_pairs(root):
    root = Path(root)
    pro = _read_winobias_file(root / "pro_stereotyped_type1.txt.test")
    anti = _read_winobias_file(root / "anti_stereotyped_type1.txt.test")

    assert len(pro) == len(anti)
    return list(zip(pro, anti))


def load_winobias_type1_single(root, kind):
    """
    kind: 'pro' or 'anti'
    """
    root = Path(root)

    if kind == "pro":
        path = root / "pro_stereotyped_type1.txt.test"
    elif kind == "anti":
        path = root / "anti_stereotyped_type1.txt.test"
    else:
        raise ValueError("kind must be 'pro' or 'anti'")

    return _read_winobias_file(path)

GENDER_TO_PRONOUN = {
    "male": "he",
    "female": "she",
    "neutral": "they",
}

def load_winogender(path):
    """
    Loads Winogender all_sentences.tsv with columns:
    sentid \t sentence

    sentid format:
    occupation.participant.index.gender.txt
    """
    groups = defaultdict(dict)

    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        if header != ["sentid", "sentence"]:
            raise ValueError(f"Unexpected header format: {header}")

        for line in f:
            sentid, sentence = line.strip().split("\t")

            # remove .txt and split
            if not sentid.endswith(".txt"):
                raise ValueError(f"Unexpected sentid format: {sentid}")

            core = sentid[:-4]  # remove ".txt"
            parts = core.split(".")

            if len(parts) != 4:
                raise ValueError(f"Unexpected sentid format: {sentid}")

            occupation, participant, ex_id, gender = parts

            if gender not in GENDER_TO_PRONOUN:
                raise ValueError(f"Unknown gender tag: {gender}")

            pronoun = GENDER_TO_PRONOUN[gender]

            key = (occupation, participant, ex_id)
            groups[key][pronoun] = sentence

    examples = []
    for (occupation, participant, ex_id), sents in groups.items():
        if "he" in sents and "she" in sents:
            examples.append({
                "occupation": occupation,
                "participant": participant,
                "example_id": ex_id,
                "sentences": sents,  # he / she / maybe they
            })

    return examples

