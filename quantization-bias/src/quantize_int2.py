import torch

def quantize_model_int2(model):

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dtype not in (torch.float16, torch.float32):
                continue

            min_val = param.data.min()
            max_val = param.data.max()

            if min_val == max_val:
                continue  # skip degenerate tensors

            # 2-bit = 4 levels
            scale = (max_val - min_val) / 3.0

            # Quantize + dequantize
            param.data = torch.round((param.data - min_val) / scale) * scale + min_val

    return model
