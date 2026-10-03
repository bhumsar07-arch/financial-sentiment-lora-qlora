"""
Model and Hardware Utilities Module.

Handles:
- Accurate hardware detection (GPU name, VRAM, CUDA, PyTorch, Transformers)
- Hardware feasibility checking with clear warnings (NO silent model downgrades)
- Tokenizer loading with proper padding and chat template support
- 4-bit NF4 quantization configuration via BitsAndBytes
- Loading base causal language models (Qwen2.5 family)
- LoRA and QLoRA model preparation via Hugging Face PEFT
- Trainable parameter analysis (total, trainable, percentage)
- Adapter disk size calculation
"""

import os
import sys
import gc
from typing import Dict, List, Tuple, Any, Optional
import torch


def detect_hardware() -> Dict[str, Any]:
    """
    Detect GPU, VRAM, CUDA version, PyTorch and Transformers versions.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": "None",
        "gpu_vram_gb": 0.0,
        "gpu_vram_mb": 0.0,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "pytorch_version": torch.__version__,
        "transformers_version": "N/A",
        "bf16_supported": False,
        "recommended_dtype": "float32",
    }

    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        pass

    if info["cuda_available"]:
        device_props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = device_props.name
        info["gpu_vram_gb"] = round(device_props.total_memory / (1024 ** 3), 2)
        info["gpu_vram_mb"] = round(device_props.total_memory / (1024 ** 2), 2)
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
        info["recommended_dtype"] = "bfloat16" if info["bf16_supported"] else "float16"

    return info


def print_hardware_summary(info: Optional[Dict[str, Any]] = None) -> None:
    """
    Print a formatted summary of hardware and environment.
    """
    if info is None:
        info = detect_hardware()

    print("=" * 65)
    print("           HARDWARE & RUNTIME ENVIRONMENT SUMMARY           ")
    print("=" * 65)
    print(f"CUDA Available:        {info['cuda_available']}")
    if info["cuda_available"]:
        print(f"GPU Name:              {info['gpu_name']}")
        print(f"GPU VRAM:              {info['gpu_vram_gb']} GB ({info['gpu_vram_mb']} MB)")
        print(f"CUDA Version:          {info['cuda_version']}")
        print(f"BF16 Supported:        {info['bf16_supported']}")
    else:
        print("GPU:                   No CUDA GPU detected! CPU mode active.")
    print(f"PyTorch Version:       {info['pytorch_version']}")
    print(f"Transformers Version:  {info['transformers_version']}")
    print(f"Compute Dtype:         {info['recommended_dtype']}")
    print("=" * 65)


def check_vram_feasibility(
    model_name: str,
    method: str = "lora",
    hardware_info: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """
    Check if the selected model and method will fit in the detected GPU VRAM.
    NEVER silently switches models. Returns (is_feasible, warning_message).
    """
    if hardware_info is None:
        hardware_info = detect_hardware()

    vram_gb = hardware_info.get("gpu_vram_gb", 0.0)

    # Minimum VRAM guidelines for Qwen2.5-3B-Instruct
    vram_requirements = {
        "Qwen/Qwen2.5-3B-Instruct": {
            "zero_shot": 8.0,
            "lora": 13.0,
            "qlora": 6.5,
        },
    }

    reqs = vram_requirements.get(model_name, {"zero_shot": 8.0, "lora": 13.0, "qlora": 6.5})
    min_required = reqs.get(method.lower(), 13.0)

    if not hardware_info["cuda_available"]:
        msg = (
            f"[WARNING] No GPU detected! Running {model_name} on CPU will be extremely slow. "
            "Please switch runtime to GPU in Colab (Runtime > Change runtime type > T4) "
            "or Kaggle (Settings > Accelerator > GPU T4/P100)."
        )
        return False, msg

    if vram_gb < min_required:
        msg = (
            f"[HARDWARE WARNING] Model '{model_name}' with method '{method.upper()}' "
            f"typically requires at least {min_required} GB VRAM, but your current GPU "
            f"({hardware_info['gpu_name']}) has only {vram_gb} GB VRAM.\n"
            f"Suggestions to reduce memory:\n"
            f"  1. If running LoRA (16-bit), run QLoRA (4-bit) which requires ~{reqs['qlora']} GB VRAM.\n"
            f"  2. Increase gradient_accumulation_steps and keep micro_batch_size = 1.\n"
            f"  3. Set max_sequence_length = 128."
        )
        return False, msg

    return True, f"Hardware check PASSED: {vram_gb} GB VRAM is sufficient for {model_name} ({method.upper()})."


def get_compute_dtype(hardware_info: Optional[Dict[str, Any]] = None) -> torch.dtype:
    """
    Select optimal compute dtype (bfloat16 if hardware supports, else float16).
    """
    if hardware_info is None:
        hardware_info = detect_hardware()

    if hardware_info.get("bf16_supported", False):
        return torch.bfloat16
    elif hardware_info.get("cuda_available", False):
        return torch.float16
    return torch.float32


def get_quantization_config(
    compute_dtype: Optional[torch.dtype] = None,
    use_double_quant: bool = True,
    quant_type: str = "nf4",
) -> Any:
    """
    Build BitsAndBytesConfig for 4-bit QLoRA.
    Uses NormalFloat4 (NF4) and nested double quantization.
    """
    from transformers import BitsAndBytesConfig

    if compute_dtype is None:
        compute_dtype = get_compute_dtype()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    return bnb_config


def load_tokenizer(model_name: str) -> Any:
    """
    Load tokenizer from HuggingFace and configure padding.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",  # left padding for decoder-only batched generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def load_base_model(
    model_name: str,
    quantization: bool = False,
    compute_dtype: Optional[torch.dtype] = None,
    device_map: str = "auto",
) -> Any:
    """
    Load the base pretrained model with optional 4-bit quantization.
    """
    from transformers import AutoModelForCausalLM

    if compute_dtype is None:
        compute_dtype = get_compute_dtype()

    load_kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if quantization:
        quant_config = get_quantization_config(compute_dtype=compute_dtype)
        load_kwargs["quantization_config"] = quant_config
    else:
        load_kwargs["torch_dtype"] = compute_dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    return model


def apply_lora_to_model(
    model: Any,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    is_qlora: bool = False,
    gradient_checkpointing: bool = True,
) -> Any:
    """
    Wrap the base model with PEFT LoRA adapter.
    If is_qlora=True, prepares model for k-bit training first.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    if is_qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )
    elif gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    peft_model = get_peft_model(model, peft_config)
    return peft_model


def count_parameters(model: Any) -> Dict[str, Any]:
    """
    Count total and trainable parameters in the model.
    Formula: Trainable % = (Trainable Parameters / Total Parameters) * 100
    """
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    trainable_pct = (trainable_params / all_params * 100) if all_params > 0 else 0.0

    return {
        "total_parameters": all_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": all_params - trainable_params,
        "trainable_percentage": round(trainable_pct, 4),
        "total_parameters_millions": round(all_params / 1e6, 2),
        "trainable_parameters_millions": round(trainable_params / 1e6, 4),
    }


def get_adapter_disk_size_mb(adapter_dir: str) -> float:
    """
    Compute total disk size of saved adapter files in megabytes.
    """
    if not os.path.exists(adapter_dir):
        return 0.0

    total_bytes = 0
    for root, _, files in os.walk(adapter_dir):
        for f in files:
            fp = os.path.join(root, f)
            total_bytes += os.path.getsize(fp)

    return round(total_bytes / (1024 ** 2), 2)


def free_memory() -> None:
    """
    Garbage collect and empty PyTorch CUDA cache.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
