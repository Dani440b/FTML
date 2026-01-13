from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

from src.quantize_int2 import quantize_model_int2

def load_model(model_name, precision="fp32"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if precision == "fp32":
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            device_map="auto"
        )

    elif precision == "fp16":
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    elif precision == "8bit":
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto"
        )

    elif precision == "4bit":
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map="auto"
        )

    elif precision == "2bit":
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            device_map="auto"
        )
        model = quantize_model_int2(model)

    else:
        raise ValueError(f"Unknown precision: {precision}")

    model.eval()
    return model, tokenizer
