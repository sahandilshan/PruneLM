import numpy as np
import torch

# https://pytorch.org/docs/stable/tensor_attributes.html
dtype2bits = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1,
}


def get_original_model_size(model):
    total_model_size = 0
    for param_tensor in model.parameters():
        tensor_size = np.prod(param_tensor.shape)
        bits = dtype2bits[param_tensor.dtype]
        tensor_size *= bits
        total_model_size += tensor_size
    return round(total_model_size / 8388608, 2)  # 8388608 = 8 * 1024 * 1024


def get_pruned_model_size(pruned_model):
    nonzero_params = 0
    for tensor in pruned_model.parameters():
        nz = np.sum(tensor.detach().cpu().numpy() != 0.0)
        bits = dtype2bits[tensor.dtype]
        nz *= bits
        nonzero_params += nz
    return round(nonzero_params / 8388608, 2)  # 8388608 = 8 * 1024 * 1024
