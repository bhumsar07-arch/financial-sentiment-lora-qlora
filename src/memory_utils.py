"""
GPU Memory Utilities and Profiling Module.

Handles:
- Accurate tracking of allocated and reserved CUDA memory
- Peak memory recording during training and inference phases
- Safe execution on both CUDA and CPU environments
- Structured logging of memory snapshots across project stages
"""

import gc
from typing import Dict, Any, Optional
import torch


def get_gpu_memory_stats() -> Dict[str, float]:
    """
    Get current and peak GPU memory statistics in Megabytes (MB) and Gigabytes (GB).
    Uses actual PyTorch CUDA memory management statistics.
    """
    if not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "peak_allocated_mb": 0.0,
            "peak_reserved_mb": 0.0,
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "peak_allocated_gb": 0.0,
            "peak_reserved_gb": 0.0,
        }

    alloc_bytes = torch.cuda.memory_allocated()
    res_bytes = torch.cuda.memory_reserved()
    max_alloc_bytes = torch.cuda.max_memory_allocated()
    max_res_bytes = torch.cuda.max_memory_reserved()

    mb = 1024 ** 2
    gb = 1024 ** 3

    return {
        "allocated_mb": round(alloc_bytes / mb, 2),
        "reserved_mb": round(res_bytes / mb, 2),
        "peak_allocated_mb": round(max_alloc_bytes / mb, 2),
        "peak_reserved_mb": round(max_res_bytes / mb, 2),
        "allocated_gb": round(alloc_bytes / gb, 3),
        "reserved_gb": round(res_bytes / gb, 3),
        "peak_allocated_gb": round(max_alloc_bytes / gb, 3),
        "peak_reserved_gb": round(max_res_bytes / gb, 3),
    }


def reset_memory_stats() -> None:
    """
    Reset peak memory statistics in PyTorch CUDA runtime.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()


class MemoryTracker:
    """
    Context manager to track peak GPU memory consumption during a code block.

    Example:
        with MemoryTracker("Model Fine-Tuning") as tracker:
            trainer.train()
        print(tracker.peak_allocated_mb, tracker.peak_reserved_mb)
    """

    def __init__(self, stage_name: str = "Operation"):
        self.stage_name = stage_name
        self.start_stats: Dict[str, float] = {}
        self.end_stats: Dict[str, float] = {}
        self.peak_allocated_mb: float = 0.0
        self.peak_reserved_mb: float = 0.0
        self.peak_allocated_gb: float = 0.0
        self.peak_reserved_gb: float = 0.0
        self.delta_allocated_mb: float = 0.0

    def __enter__(self):
        reset_memory_stats()
        self.start_stats = get_gpu_memory_stats()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_stats = get_gpu_memory_stats()
        self.peak_allocated_mb = self.end_stats["peak_allocated_mb"]
        self.peak_reserved_mb = self.end_stats["peak_reserved_mb"]
        self.peak_allocated_gb = self.end_stats["peak_allocated_gb"]
        self.peak_reserved_gb = self.end_stats["peak_reserved_gb"]
        self.delta_allocated_mb = round(
            self.end_stats["allocated_mb"] - self.start_stats["allocated_mb"], 2
        )

    def summary(self) -> str:
        return (
            f"[{self.stage_name}] Peak Allocated: {self.peak_allocated_mb} MB ({self.peak_allocated_gb} GB) | "
            f"Peak Reserved: {self.peak_reserved_mb} MB ({self.peak_reserved_gb} GB)"
        )


class MemorySnapshotLogger:
    """
    Maintains a record of memory snapshots across all experimental phases:
    1. before model loading
    2. after model loading
    3. before training
    4. during training
    5. peak training memory
    6. after training
    7. during inference
    8. peak inference memory
    """

    def __init__(self):
        self.snapshots: Dict[str, Dict[str, float]] = {}

    def record(self, stage_name: str) -> Dict[str, float]:
        stats = get_gpu_memory_stats()
        self.snapshots[stage_name] = stats
        return stats

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return self.snapshots
