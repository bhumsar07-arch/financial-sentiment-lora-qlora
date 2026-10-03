"""
Efficient Financial Sentiment Fine-Tuning with LoRA and QLoRA.
Package Initialization.
"""

# Core data and evaluation metrics are always available (lightweight, zero-torch required)
from .data_utils import (
    load_raw_dataset,
    check_data_quality,
    create_stratified_split,
    format_instruction_prompt,
    format_chat_prompt,
    save_split_metadata,
    ID2LABEL,
    LABEL2ID,
    PROMPT_TEMPLATE,
)

from .metrics import (
    parse_model_output,
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_invalid_rate,
    format_metrics_summary,
    VALID_LABELS,
)

# Deep learning and GPU utilities are imported when PyTorch is installed
try:
    from .model_utils import (
        detect_hardware,
        print_hardware_summary,
        check_vram_feasibility,
        get_compute_dtype,
        get_quantization_config,
        load_tokenizer,
        load_base_model,
        apply_lora_to_model,
        count_parameters,
        get_adapter_disk_size_mb,
        free_memory,
    )
except ImportError:
    pass

try:
    from .memory_utils import (
        get_gpu_memory_stats,
        reset_memory_stats,
        MemoryTracker,
        MemorySnapshotLogger,
    )
except ImportError:
    pass

try:
    from .benchmarking import (
        benchmark_inference,
        create_efficiency_scorecard,
        compute_resume_metrics,
    )
except ImportError:
    pass

try:
    from .visualization import (
        plot_class_distribution,
        plot_split_distribution,
        plot_sentence_lengths,
        plot_confusion_matrix_heatmap,
        plot_performance_comparison,
        plot_efficiency_comparison,
        plot_rank_ablation,
        plot_performance_vs_efficiency,
    )
except ImportError:
    pass
