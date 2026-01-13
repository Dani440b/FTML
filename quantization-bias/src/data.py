from pathlib import Path

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
