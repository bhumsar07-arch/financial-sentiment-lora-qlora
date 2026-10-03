"""
Benchmarking and Efficiency Metrics Module.

Handles:
- Inference efficiency measurement:
  - Average, median, and 95th percentile latency (ms)
  - Throughput (samples per second)
  - Peak GPU inference memory
- Training efficiency measurement:
  - Total training duration
  - Throughput (steps/sec, samples/sec)
  - Peak GPU training memory
  - Trainable parameter percentage
- Generation of the Final Efficiency Scorecard
- Calculation of resume-ready metrics:
  - LoRA Macro-F1 improvement over Zero-shot
  - QLoRA Macro-F1 improvement over Zero-shot
  - VRAM savings percentage (LoRA vs QLoRA)
  - Parameter reduction percentage
"""

import os
import time
import json
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from .memory_utils import MemoryTracker, reset_memory_stats


def benchmark_inference(
    model: Any,
    tokenizer: Any,
    test_prompts: List[str],
    max_new_tokens: int = 8,
    batch_size: int = 1,
    warmup_steps: int = 3,
) -> Dict[str, Any]:
    """
    Run controlled inference benchmarking on a set of prompts.
    Measures latency per sample (avg, median, p95), throughput, and peak memory.
    """
    import torch

    latencies_ms = []
    device = next(model.parameters()).device

    # Warmup runs
    warmup_prompts = test_prompts[:min(warmup_steps, len(test_prompts))]
    for prompt in warmup_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    reset_memory_stats()
    tracker = MemoryTracker("Inference Benchmark")
    tracker.__enter__()

    total_start = time.perf_counter()

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    total_duration = time.perf_counter() - total_start
    tracker.__exit__(None, None, None)

    latencies_arr = np.array(latencies_ms)
    num_samples = len(test_prompts)

    results = {
        "num_samples": num_samples,
        "total_time_seconds": round(total_duration, 3),
        "throughput_samples_per_sec": round(num_samples / total_duration, 2) if total_duration > 0 else 0.0,
        "latency_avg_ms": round(float(np.mean(latencies_arr)), 2),
        "latency_median_ms": round(float(np.median(latencies_arr)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies_arr, 95)), 2),
        "latency_min_ms": round(float(np.min(latencies_arr)), 2),
        "latency_max_ms": round(float(np.max(latencies_arr)), 2),
        "peak_inference_memory_mb": tracker.peak_allocated_mb,
        "peak_inference_reserved_mb": tracker.peak_reserved_mb,
        "peak_inference_memory_gb": tracker.peak_allocated_gb,
    }
    return results


def create_efficiency_scorecard(
    zero_shot_metrics: Optional[Dict[str, Any]] = None,
    lora_metrics: Optional[Dict[str, Any]] = None,
    qlora_metrics: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Construct the final multi-dimensional efficiency scorecard comparing:
    - Zero-Shot
    - LoRA
    - QLoRA

    Follows the strict No-Fabrication policy: any unrun method is marked NOT_RUN.
    """
    methods = [
        ("Zero-shot", zero_shot_metrics),
        ("LoRA", lora_metrics),
        ("QLoRA", qlora_metrics),
    ]

    rows = []
    for name, data in methods:
        if not data or data.get("status") == "NOT_RUN":
            rows.append({
                "Method": name,
                "Accuracy": "NOT_RUN",
                "Macro-F1": "NOT_RUN",
                "Trainable %": "0.0%" if name == "Zero-shot" else "NOT_RUN",
                "Peak VRAM (GB)": "NOT_RUN",
                "Training Time": "N/A" if name == "Zero-shot" else "NOT_RUN",
                "Adapter Size (MB)": "N/A" if name == "Zero-shot" else "NOT_RUN",
                "Latency (ms)": "NOT_RUN",
            })
            continue

        acc = f"{data.get('accuracy', 0.0)*100:.2f}%" if "accuracy" in data else "N/A"
        f1 = f"{data.get('macro_f1', 0.0):.4f}" if "macro_f1" in data else "N/A"
        trainable_pct = f"{data.get('trainable_percentage', 0.0):.4f}%" if "trainable_percentage" in data else ("0.0%" if name == "Zero-shot" else "N/A")
        vram = f"{data.get('peak_vram_gb', data.get('peak_memory_gb', 0.0)):.2f}" if ("peak_vram_gb" in data or "peak_memory_gb" in data) else "N/A"

        train_time = data.get("training_time_str", "N/A")
        if train_time == "N/A" and "training_time_seconds" in data:
            sec = data["training_time_seconds"]
            train_time = f"{sec/60:.1f} min" if sec > 60 else f"{sec:.1f} sec"

        adapter_size = f"{data.get('adapter_size_mb', 0.0):.2f}" if "adapter_size_mb" in data else ("N/A" if name == "Zero-shot" else "N/A")
        latency = f"{data.get('latency_median_ms', data.get('latency_avg_ms', 0.0)):.1f}" if ("latency_median_ms" in data or "latency_avg_ms" in data) else "N/A"

        rows.append({
            "Method": name,
            "Accuracy": acc,
            "Macro-F1": f1,
            "Trainable %": trainable_pct,
            "Peak VRAM (GB)": vram,
            "Training Time": train_time,
            "Adapter Size (MB)": adapter_size,
            "Latency (ms)": latency,
        })

    return pd.DataFrame(rows)


def compute_resume_metrics(
    zero_shot: Dict[str, Any],
    lora: Dict[str, Any],
    qlora: Dict[str, Any],
    output_dir: str = "results",
) -> Dict[str, Any]:
    """
    Compute specific quantitative improvements for resume and portfolio discussion:
    - LoRA F1 Improvement = LoRA MacroF1 - ZeroShot MacroF1
    - QLoRA F1 Improvement = QLoRA MacroF1 - ZeroShot MacroF1
    - Memory Reduction = ((LoRA VRAM - QLoRA VRAM) / LoRA VRAM) * 100
    - Trainable parameter reduction vs full fine-tuning
    - Latency differences

    Saves results to:
      results/resume_metrics.json
      results/resume_metrics.txt
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = {}

    zs_f1 = zero_shot.get("macro_f1")
    lora_f1 = lora.get("macro_f1")
    qlora_f1 = qlora.get("macro_f1")

    # F1 Improvements
    if lora_f1 is not None and zs_f1 is not None:
        metrics["lora_f1_improvement"] = round(lora_f1 - zs_f1, 4)
        metrics["lora_f1_relative_gain_pct"] = round(((lora_f1 - zs_f1) / zs_f1) * 100, 2) if zs_f1 > 0 else 0.0

    if qlora_f1 is not None and zs_f1 is not None:
        metrics["qlora_f1_improvement"] = round(qlora_f1 - zs_f1, 4)
        metrics["qlora_f1_relative_gain_pct"] = round(((qlora_f1 - zs_f1) / zs_f1) * 100, 2) if zs_f1 > 0 else 0.0

    if lora_f1 is not None and qlora_f1 is not None:
        metrics["qlora_vs_lora_f1_diff"] = round(qlora_f1 - lora_f1, 4)

    # Memory Savings (LoRA VRAM vs QLoRA VRAM)
    lora_vram = lora.get("peak_vram_gb", lora.get("peak_training_memory_gb"))
    qlora_vram = qlora.get("peak_vram_gb", qlora.get("peak_training_memory_gb"))

    if lora_vram is not None and qlora_vram is not None and lora_vram > 0:
        mem_savings_pct = ((lora_vram - qlora_vram) / lora_vram) * 100
        metrics["qlora_memory_reduction_pct"] = round(mem_savings_pct, 2)
        metrics["vram_saved_gb"] = round(lora_vram - qlora_vram, 2)

    # Trainable Parameters
    trainable_pct = lora.get("trainable_percentage", qlora.get("trainable_percentage"))
    if trainable_pct is not None:
        metrics["trainable_percentage"] = trainable_pct
        metrics["parameter_reduction_vs_full_pct"] = round(100.0 - trainable_pct, 4)

    # Adapter Sizes
    if "adapter_size_mb" in lora:
        metrics["lora_adapter_size_mb"] = lora["adapter_size_mb"]
    if "adapter_size_mb" in qlora:
        metrics["qlora_adapter_size_mb"] = qlora["adapter_size_mb"]

    # Save to JSON
    json_path = os.path.join(output_dir, "resume_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save human-readable text bullets for direct resume copy-paste
    txt_path = os.path.join(output_dir, "resume_metrics.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("      QUANTITATIVE HIGHLIGHTS FOR RESUME & INTERVIEWS       \n")
        f.write("=" * 65 + "\n\n")

        if "lora_f1_improvement" in metrics:
            f.write(
                f"- Fine-tuned Qwen2.5-3B using LoRA for 3-class financial sentiment, "
                f"improving Macro-F1 by +{metrics['lora_f1_improvement']:.4f} "
                f"(+{metrics.get('lora_f1_relative_gain_pct', 0.0)}% relative) over zero-shot baseline.\n"
            )

        if "qlora_f1_improvement" in metrics:
            f.write(
                f"- Evaluated 4-bit QLoRA (NF4 + double quantization), maintaining within "
                f"{abs(metrics.get('qlora_vs_lora_f1_diff', 0.0)):.4f} Macro-F1 of full 16-bit LoRA.\n"
            )

        if "qlora_memory_reduction_pct" in metrics:
            f.write(
                f"- Reduced peak GPU training VRAM by {metrics['qlora_memory_reduction_pct']}% "
                f"({metrics.get('vram_saved_gb', 0.0)} GB saved) using QLoRA vs standard 16-bit LoRA.\n"
            )

        if "parameter_reduction_vs_full_pct" in metrics:
            f.write(
                f"- Trained only {metrics.get('trainable_percentage', '0.1')}% of model parameters "
                f"({metrics['parameter_reduction_vs_full_pct']}% parameter reduction vs full fine-tuning), "
                f"enabling compact adapter checkpointing (~{metrics.get('lora_adapter_size_mb', 'N/A')} MB).\n"
            )

        f.write("\n" + "=" * 65 + "\n")

    return metrics
