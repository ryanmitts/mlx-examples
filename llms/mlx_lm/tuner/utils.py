import os
import re
from typing import Dict

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from .lora import LoRALinear


def linear_to_lora_layers(
    model: nn.Module,
    num_lora_layers: int,
    config: Dict,
):
    """
    Convert some of the models linear layers to lora layers.

    Args:
        model (nn.Module): The neural network model.
        num_lora_layers (int): The number of blocks to convert to lora layers
        starting from the last layer.
        config (dict): More configuration parameters for LoRA, including the
          rank, alpha, scale, and optional layer keys.
    """

    num_layers = len(model.layers)
    if num_lora_layers > num_layers:
        raise ValueError(
            f"Requested {num_lora_layers} LoRA layers "
            f"but the model only has {num_layers} layers."
        )

    to_lora = lambda lin: LoRALinear.from_linear(
        lin,
        r=config["rank"],
        alpha=config["alpha"],
        scale=config["scale"],
        dropout=config["dropout"],
    )

    keys = config.get("keys", None)
    if keys is not None:
        keys = set(keys)
    elif model.model_type in [
        "mistral",
        "llama",
        "phi",
        "mixtral",
        "stablelm",
        "qwen2",
        "gemma",
        "starcoder2",
        "cohere",
    ]:
        keys = set(["self_attn.q_proj", "self_attn.v_proj"])
        if model.model_type == "mixtral":
            keys.add("block_sparse_moe.gate")
    elif model.model_type == "olmo":
        keys = set(["att_proj"])
    elif model.model_type == "phi-msft":
        keys = set(["mixer.Wqkv", "moe.gate"])
    else:
        raise ValueError(f"Lora does not support {model.model_type}")

    for l in model.layers[num_layers - num_lora_layers :]:
        modules = l.named_modules()
        lora_layers = [(k, to_lora(m)) for k, m in l.named_modules() if k in keys]
        l.update_modules(tree_unflatten(lora_layers))


def apply_lora_layers(model: nn.Module, adapter_file: str) -> nn.Module:
    """
    Apply LoRA layers to the model.

    Args:
        model (nn.Module): The neural network model.
        adapter_file (str): Path to the adapter configuration file.

    Returns:
        nn.Module: The updated model with LoRA layers applied.
    """
    if not os.path.exists(adapter_file):
        raise FileNotFoundError(f"The adapter file does not exist: {adapter_file}")

    _, adapters_extension = os.path.splitext(adapter_file)

    linear_replacements = []

    if adapters_extension == ".npz":
        adapters_import = mx.load(adapter_file)  # Metadata not supported on npz
    else:
        adapters_import, metadata = mx.load(adapter_file, return_metadata=True)
        if not metadata or metadata["format"] != "pt":
            raise ValueError(f"Unsupported adapters format {metadata['format']}")
        # PyTorchs nn.Linear module has the dimensions reversed than what is needed.
        # https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L98
        # This normalized the layer names to remove any extra `base_model` or `model` names.
        # Also removes the PyTorch `.weight` layer, which is the weights for the MLX linear layer.
        adapters_import = {
            re.search(r"model.(?!model\.)(.*?)(?=$|.weight)", k.lower()).group(
                0
            ): mx.transpose(v)
            for k, v in adapters_import.items()
        }

    adapters = list(adapters_import.items())

    lora_layers = set(
        [name.replace(".lora_a", "").replace(".lora_b", "") for name, _ in adapters]
    )

    for name, module in model.named_modules():
        if name in lora_layers:
            replacement_module = LoRALinear.from_linear(module)
            linear_replacements.append((name, replacement_module))

    model.update_modules(tree_unflatten(linear_replacements))
    model.update(tree_unflatten(adapters))

    return model


def dequantize(model: nn.Module) -> nn.Module:
    """
    Dequantize the quantized linear layers in the model.

    Args:
        model (nn.Module): The model with quantized linear layers.

    Returns:
        nn.Module: The model with dequantized layers.
    """
    de_quantize_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            bias = "bias" in module
            weight = module.weight
            weight = mx.dequantize(
                weight,
                module.scales,
                module.biases,
                module.group_size,
                module.bits,
            ).astype(mx.float16)
            output_dims, input_dims = weight.shape
            linear = nn.Linear(input_dims, output_dims, bias=bias)
            linear.weight = weight
            if bias:
                linear.bias = module.bias
            de_quantize_layers.append((name, linear))
    if len(de_quantize_layers) > 0:
        model.update_modules(tree_unflatten(de_quantize_layers))
    return model


def remove_lora_layers(model: nn.Module) -> nn.Module:
    """
    Remove the LoRA layers from the model.

    Args:
        model (nn.Module): The model with LoRA layers.

    Returns:
        nn.Module: The model without LoRA layers.
    """
    reset_layers = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            reset_layers.append((name, module.linear))
    if len(reset_layers) > 0:
        model.update_modules(tree_unflatten(reset_layers))
    return model
