This project investigates how model quantization affects gender bias in pretrained language models.
While quantization is widely used to reduce memory and compute costs, its impact on fairness and bias is underexplored.

We evaluate how different precision levels (FP32, FP16, 8-bit, 4-bit, 2-bit) change bias metrics on established benchmarks:

- WinoBias
- Winogender
- CrowS-Pairs

The focus is on whether quantization amplifies, reduces, or destabilizes bias rather than accuracy alone.

The structure of the project is:

quantization-bias/
├── src/
│   ├── data.py            # Dataset loaders (WinoBias, Winogender)
│   ├── preprocessing.py   # Pronoun masking utilities for winogender
│   ├── scoring.py         # Sentence and masked-token scoring
│   ├── load_model.py      # Model + quantization loading
│   └── utils.py           # Result saving helpers
│
├── experiments/
│   ├── run_winobias_type1.py
│   ├── run_winogender.py
│   ├── run_crows_pairs.py
│   └── ... 		   # Other possible tests, can safely be ignored
│
├── data/
│   ├── winobias/
│   ├── winogender/
│   └── crows_pairs.csv    
│
└── results/
    ├── winobias/
    ├── winogender/
    └── crows_pairs/

I personally used micromamba to manage my python enviroment with python 3.10, here are the required packages:

- torch 
- transformers 
- pandas 
- tqdm 
- bitsandbytes
- numpy


All experiments are run as Python modules from the quantization-bias dir, as follow:

python -m experiments.run_winogender
python -m experiments.run_winobias
python -m experiments.run_crows_pairs

and will save the results in the results dir.
