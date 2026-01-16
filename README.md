## Overview ▶️

This project investigates how model quantization affects gender bias in pretrained language models.
While quantization is widely used to reduce memory and compute costs, its impact on fairness and bias is underexplored.

We evaluate how different precision levels (FP32, FP16, 8-bit, 4-bit, 2-bit) change bias metrics on established benchmarks:

```
- WinoBias
- Winogender
- CrowS-Pairs
```

The focus is on whether quantization amplifies, reduces, or destabilizes bias rather than accuracy alone.

## Structure 📁

The structure of the project is:

    quantization-bias/
      src/
        data.py            - Dataset loaders (WinoBias, Winogender, CrowS-Pairs)
        preprocessing.py   - Pronoun masking utilities (Winogender)
        scoring.py         - Sentence and masked-token scoring
        load_model.py      - Model loading + quantization
        utils.py           - Result saving helpers

      experiments/
        run_winobias_type1.py
        run_winogender.py
        run_crows_pairs.py
        ...                - Other experimental scripts (can be ignored)

      data/
        winobias/
        winogender/
          all_sentences.tsv
        crows_pairs.csv

      results/
        winobias/
        winogender/
        crows_pairs/

## Python enviroment 😭

I personally used micromamba to manage my python enviroment with python 3.10, here are the required packages:

```
- torch 
- transformers 
- pandas 
- tqdm 
- bitsandbytes
- numpy
```

## Running 🏃‍➡️

All experiments are run as Python modules from the quantization-bias dir, as follow:

```
python -m experiments.run_winogender
python -m experiments.run_winobias_type1
python -m experiments.run_crows_pairs
```

and will save the results in the results dir.

## Credits 🤝

Models:
- RoBERTa-base (Hugging Face): https://huggingface.co/FacebookAI/roberta-base

Datasets:
- WinoBias: https://github.com/uclanlp/corefBias
- Winogender: https://github.com/rudinger/winogender-schemas
- CrowS-Pairs: https://github.com/nyu-mll/crows-pairs
