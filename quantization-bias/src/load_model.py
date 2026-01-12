from transformers import AutoModelForMaskedLM, AutoTokenizer

def load_model(model_name, precision="fp32"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if precision == "fp32":
        model = AutoModelForMaskedLM.from_pretrained(model_name)

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

    else:
        raise ValueError(f"Unknown precision: {precision}")

    model.eval()
    return model, tokenizer
