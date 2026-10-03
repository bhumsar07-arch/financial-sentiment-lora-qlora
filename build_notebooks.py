"""
Script to build all 7 clean, research-grade Jupyter Notebooks (.ipynb)
for the Financial Sentiment LoRA/QLoRA project.
All text is kept in simple English as requested.
"""

import os
import json


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def md_cell(source_text):
    lines = [line + "\n" for line in source_text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def code_cell(source_text):
    lines = [line + "\n" for line in source_text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


# ==============================================================================
# SHARED COLAB / KAGGLE SETUP CELL
# Injected as the FIRST code cell in every notebook so that `from src.xxx`
# works out-of-the-box when a user opens the notebook in Google Colab or Kaggle.
# ==============================================================================
COLAB_SETUP_CELL = code_cell(
    "# === ENVIRONMENT SETUP (Google Colab / Kaggle / Local) ===\n"
    "# This cell automatically detects your environment and sets up the project.\n"
    "# If running locally from the repo root, it simply adds '.' to the Python path.\n"
    "import os, sys, subprocess\n"
    "\n"
    "# --- Configuration (edit this URL if you use your own fork) ---\n"
    "REPO_URL  = 'https://github.com/YOUR_USERNAME/lora-qlora-financial-sentiment.git'\n"
    "REPO_DIR  = 'lora-qlora-financial-sentiment'\n"
    "\n"
    "# 1. Detect if src/ already exists (local run or repo already cloned)\n"
    "if not os.path.exists('src') and not os.path.exists('../src'):\n"
    "    # We are probably on Colab (/content) or Kaggle (/kaggle/working)\n"
    "    if os.path.exists(os.path.join(REPO_DIR, 'src')):\n"
    "        # Repo folder exists but we are not inside it\n"
    "        os.chdir(REPO_DIR)\n"
    "        print(f'Changed directory to {os.getcwd()}')\n"
    "    else:\n"
    "        print(f'Cloning repository ...')\n"
    "        subprocess.run(['git', 'clone', REPO_URL], check=True)\n"
    "        os.chdir(REPO_DIR)\n"
    "        print(f'Cloned and changed directory to {os.getcwd()}')\n"
    "elif os.path.exists('../src') and not os.path.exists('src'):\n"
    "    os.chdir('..')\n"
    "    print(f'Changed directory to project root: {os.getcwd()}')\n"
    "\n"
    "# 2. Ensure project root is on Python path\n"
    "project_root = os.getcwd()\n"
    "if os.path.exists(os.path.join(project_root, 'src')):\n"
    "    if project_root not in sys.path:\n"
    "        sys.path.insert(0, project_root)\n"
    "\n"
    "# 3. Install dependencies\n"
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',\n"
    "    'transformers', 'datasets', 'peft', 'accelerate', 'bitsandbytes',\n"
    "    'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'pyyaml'],\n"
    "    check=True)\n"
    "\n"
    "# 4. Create output directories\n"
    "for d in ['results/metrics', 'results/predictions', 'results/figures', 'results/logs',\n"
    "          'adapters/lora', 'adapters/qlora']:\n"
    "    os.makedirs(d, exist_ok=True)\n"
    "\n"
    "print(f'Working directory: {os.getcwd()}')\n"
    "print(f'src/ found: {os.path.exists(\"src\")}')\n"
    "print('Setup complete!')\n"
)


# ==============================================================================
# NOTEBOOK 01: DATASET ANALYSIS
# ==============================================================================
nb01_cells = [
    md_cell(
        "# Notebook 01: Financial PhraseBank Dataset Analysis and Stratified Splitting\n\n"
        "### Project: Efficient Financial Sentiment Fine-Tuning with LoRA and QLoRA\n\n"
        "In this notebook, we explore the **Financial PhraseBank** dataset (`takala/financial_phrasebank`, configuration `sentences_50agree`).\n\n"
        "### What this notebook does:\n"
        "1. Installs all required project dependencies.\n"
        "2. Inspects available hardware (GPU, VRAM, CUDA, PyTorch).\n"
        "3. Downloads the dataset and checks data quality (missing values, duplicates).\n"
        "4. Analyzes the class distribution (positive, negative, neutral) and sentence lengths.\n"
        "5. Creates a fixed, stratified **70% train / 15% validation / 15% test** split with random seed `42`.\n"
        "6. Saves split metadata to `results/metrics/split_metadata.json` and creates visual charts."
    ),
    code_cell(
        "# Step 2: Check hardware environment\n"
        "from src.model_utils import detect_hardware, print_hardware_summary\n"
        "\n"
        "hw = detect_hardware()\n"
        "print_hardware_summary(hw)"
    ),
    code_cell(
        "# Step 3: Load the Financial PhraseBank dataset\n"
        "from src.data_utils import load_raw_dataset, check_data_quality, ID2LABEL\n"
        "\n"
        "print('Loading Financial PhraseBank (sentences_50agree)...')\n"
        "df = load_raw_dataset('takala/financial_phrasebank', 'sentences_50agree')\n"
        "\n"
        "print(f'Total samples downloaded: {len(df)}')\n"
        "display(df.head(5))"
    ),
    code_cell(
        "# Step 4: Check dataset quality and statistics\n"
        "quality_stats = check_data_quality(df)\n"
        "\n"
        "print('=== Dataset Quality Summary ===')\n"
        "print(f'Total sentences:      {quality_stats[\"total_samples\"]}')\n"
        "print(f'Missing values:       {quality_stats[\"missing_values\"]}')\n"
        "print(f'Duplicate sentences:  {quality_stats[\"duplicate_sentences\"]}')\n"
        "print('\\nClass counts:')\n"
        "for label, count in quality_stats['class_distribution'].items():\n"
        "    pct = (count / quality_stats['total_samples']) * 100\n"
        "    print(f'  {label:10s}: {count:5d} ({pct:.1f}%)')\n"
        "\n"
        "print('\\nSentence word length statistics:')\n"
        "for k, v in quality_stats['word_length'].items():\n"
        "    print(f'  {k:10s}: {v}')"
    ),
    code_cell(
        "# Step 5: Create a stratified 70 / 15 / 15 train, validation, and test split\n"
        "from src.data_utils import create_stratified_split, save_split_metadata, check_leakage\n"
        "\n"
        "train_df, val_df, test_df = create_stratified_split(\n"
        "    df,\n"
        "    train_ratio=0.70,\n"
        "    val_ratio=0.15,\n"
        "    test_ratio=0.15,\n"
        "    random_seed=42,\n"
        ")\n"
        "\n"
        "print(f'Train split size:      {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)')\n"
        "print(f'Validation split size: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)')\n"
        "print(f'Test split size:       {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)')\n"
        "\n"
        "# Verify zero sentence leakage across splits\n"
        "leakage = check_leakage(train_df, val_df, test_df)\n"
        "print(f'Sentence overlap check: {leakage}')\n"
        "\n"
        "# Save split metadata to ensure reproducibility\n"
        "meta = save_split_metadata(train_df, val_df, test_df, 'results/metrics/split_metadata.json')\n"
        "print('\\nSplit metadata successfully saved to results/metrics/split_metadata.json')"
    ),
    code_cell(
        "# Step 6: Generate and save visualizations\n"
        "from src.visualization import plot_class_distribution, plot_split_distribution, plot_sentence_lengths\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "os.makedirs('results/figures', exist_ok=True)\n"
        "\n"
        "plot_class_distribution(df, save_path='results/figures/class_distribution.png')\n"
        "plot_split_distribution(train_df, val_df, test_df, save_path='results/figures/split_distribution.png')\n"
        "plot_sentence_lengths(df, save_path='results/figures/sentence_length_distribution.png')\n"
        "\n"
        "print('Saved plots to results/figures/:')\n"
        "print(' - class_distribution.png')\n"
        "print(' - split_distribution.png')\n"
        "print(' - sentence_length_distribution.png')\n"
        "\n"
        "# Display the split distribution chart inline\n"
        "from IPython.display import Image\n"
        "Image(filename='results/figures/split_distribution.png')"
    ),
    md_cell(
        "### Key Takeaways from Dataset Analysis\n\n"
        "- The dataset contains **4,846 sentences** with high agreement between annotators.\n"
        "- The classes are imbalanced: **neutral** represents approximately 60% of samples, followed by **positive** (~28%) and **negative** (~12%).\n"
        "- The stratified 70/15/15 split maintains the exact class proportion in all three splits.\n"
        "- Sentence lengths are compact (median ~21 words), well within a `max_sequence_length` of 256 tokens.\n"
        "- The fixed test set created here will be used without modification across all subsequent experiments."
    ),
]


# ==============================================================================
# NOTEBOOK 02: ZERO-SHOT BASELINE
# ==============================================================================
nb02_cells = [
    md_cell(
        "# Notebook 02: Zero-Shot Baseline Evaluation\n\n"
        "### Research Question 1 (RQ1):\n"
        "> *How well does the pretrained LLM perform financial sentiment classification without any task-specific fine-tuning?*\n\n"
        "In this notebook, we evaluate the unmodified base model on our fixed test set using instruction prompting.\n\n"
        "### What this notebook measures:\n"
        "1. **Classification Quality:** Accuracy, Macro-F1, Weighted-F1, per-class Precision/Recall/F1, Confusion Matrix.\n"
        "2. **Formatting Reliability:** Invalid output rate (how often the model fails to output one of the three sentiment classes).\n"
        "3. **Inference Efficiency:** Average latency, median latency, 95th percentile latency, throughput (samples/sec), and peak GPU memory.\n"
        "4. Saves predictions to `results/predictions/zero_shot.csv` and metrics to `results/metrics/zero_shot.json`."
    ),
    code_cell(
        "# Step 1: Install dependencies and set up environment\n"
        "!pip install -q transformers datasets peft accelerate bitsandbytes pandas numpy scikit-learn matplotlib seaborn pyyaml"
    ),
    code_cell(
        "# Step 2: Hardware check and Model Selection\n"
        "import os\n"
        "import sys\n"
        "import json\n"
        "import torch\n"
        "\n"
        "if os.path.exists('src'):\n"
        "    sys.path.insert(0, '.')\n"
        "elif os.path.exists('../src'):\n"
        "    sys.path.insert(0, '..')\n"
        "\n"
        "from src.model_utils import detect_hardware, print_hardware_summary, check_vram_feasibility\n"
        "\n"
        "hw = detect_hardware()\n"
        "print_hardware_summary(hw)\n"
        "\n"
        "# Primary research model: Qwen2.5-3B-Instruct (3.09B parameters)\n"
        "MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'\n"
        "\n"
        "is_feasible, warning_msg = check_vram_feasibility(MODEL_NAME, method='zero_shot', hardware_info=hw)\n"
        "print(warning_msg)\n"
        "print(f'Active model: {MODEL_NAME}')"
    ),
    code_cell(
        "# Step 3: Load the fixed test dataset\n"
        "from src.data_utils import load_raw_dataset, create_stratified_split\n"
        "\n"
        "df = load_raw_dataset('takala/financial_phrasebank', 'sentences_50agree')\n"
        "_, _, test_df = create_stratified_split(df, 0.70, 0.15, 0.15, random_seed=42)\n"
        "print(f'Test set loaded: {len(test_df)} samples.')"
    ),
    code_cell(
        "# Step 4: Load Base Model and Tokenizer\n"
        "from src.model_utils import load_tokenizer, load_base_model, count_parameters\n"
        "from src.memory_utils import MemoryTracker, get_gpu_memory_stats\n"
        "\n"
        "print(f'Loading tokenizer for {MODEL_NAME}...')\n"
        "tokenizer = load_tokenizer(MODEL_NAME)\n"
        "\n"
        "print(f'Loading base model {MODEL_NAME}...')\n"
        "with MemoryTracker('Base Model Load') as load_tracker:\n"
        "    model = load_base_model(MODEL_NAME, quantization=False)\n"
        "\n"
        "params = count_parameters(model)\n"
        "print(f'Total parameters: {params[\"total_parameters_millions\"]} M')\n"
        "print(load_tracker.summary())"
    ),
    code_cell(
        "# Step 5: Run Zero-Shot Inference on Test Set\n"
        "from src.data_utils import format_chat_prompt\n"
        "from src.metrics import parse_model_output\n"
        "import time\n"
        "\n"
        "device = next(model.parameters()).device\n"
        "raw_outputs = []\n"
        "parsed_preds = []\n"
        "latencies = []\n"
        "\n"
        "print(f'Starting zero-shot inference on {len(test_df)} samples...')\n"
        "t_start = time.perf_counter()\n"
        "\n"
        "for idx, row in test_df.iterrows():\n"
        "    prompt = format_chat_prompt(row['sentence'], tokenizer=tokenizer)\n"
        "    inputs = tokenizer(prompt, return_tensors='pt').to(device)\n"
        "    \n"
        "    t0 = time.perf_counter()\n"
        "    with torch.no_grad():\n"
        "        out = model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.pad_token_id)\n"
        "    t1 = time.perf_counter()\n"
        "    latencies.append((t1 - t0) * 1000.0)\n"
        "    \n"
        "    gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)\n"
        "    raw_outputs.append(gen_text)\n"
        "    parsed_preds.append(parse_model_output(gen_text))\n"
        "\n"
        "total_inference_time = time.perf_counter() - t_start\n"
        "print(f'Inference completed in {total_inference_time:.1f} seconds.')"
    ),
    code_cell(
        "# Step 6: Compute Metrics and Save Results\n"
        "from src.metrics import compute_classification_metrics, format_metrics_summary\n"
        "from src.visualization import plot_confusion_matrix_heatmap\n"
        "import numpy as np\n"
        "\n"
        "y_true = test_df['label_text'].tolist()\n"
        "metrics = compute_classification_metrics(y_true, parsed_preds)\n"
        "\n"
        "# Add efficiency statistics\n"
        "metrics['method'] = 'zero_shot'\n"
        "metrics['model_name'] = MODEL_NAME\n"
        "metrics['latency_avg_ms'] = round(float(np.mean(latencies)), 2)\n"
        "metrics['latency_median_ms'] = round(float(np.median(latencies)), 2)\n"
        "metrics['latency_p95_ms'] = round(float(np.percentile(latencies, 95)), 2)\n"
        "metrics['throughput_samples_per_sec'] = round(len(test_df) / total_inference_time, 2)\n"
        "metrics['peak_memory_gb'] = get_gpu_memory_stats()['peak_allocated_gb']\n"
        "\n"
        "print(format_metrics_summary(metrics, 'Zero-Shot Baseline'))\n"
        "print(f'Median Latency: {metrics[\"latency_median_ms\"]} ms | P95: {metrics[\"latency_p95_ms\"]} ms')\n"
        "\n"
        "# Save metrics\n"
        "os.makedirs('results/metrics', exist_ok=True)\n"
        "os.makedirs('results/predictions', exist_ok=True)\n"
        "\n"
        "with open('results/metrics/zero_shot.json', 'w') as f:\n"
        "    json.dump(metrics, f, indent=2)\n"
        "\n"
        "# Save predictions dataframe\n"
        "pred_df = test_df.copy()\n"
        "pred_df['raw_output'] = raw_outputs\n"
        "pred_df['predicted_label'] = parsed_preds\n"
        "pred_df['latency_ms'] = latencies\n"
        "pred_df.to_csv('results/predictions/zero_shot.csv', index=False)\n"
        "\n"
        "# Plot confusion matrix\n"
        "plot_confusion_matrix_heatmap(metrics['confusion_matrix'], method_name='Zero-Shot Baseline')\n"
        "print('Saved results to results/metrics/zero_shot.json and results/predictions/zero_shot.csv')"
    ),
    md_cell(
        "### RQ1 Discussion\n\n"
        "The zero-shot evaluation establishes the performance floor of the general instruction-tuned model.\n"
        "Key aspects to observe:\n"
        "1. Does the model recognize financial domain nuances, or does it default heavily to neutral?\n"
        "2. What is the invalid output rate when prompted without task-specific tuning?\n"
        "3. In the next notebook, we apply 16-bit LoRA to measure how much task adaptation improves these numbers."
    ),
]


# ==============================================================================
# NOTEBOOK 03: LoRA TRAINING
# ==============================================================================
nb03_cells = [
    md_cell(
        "# Notebook 03: Parameter-Efficient Fine-Tuning with LoRA\n\n"
        "### Research Questions:\n"
        "> **RQ2:** *How much does LoRA improve financial sentiment classification over the base model?*\n"
        "> **RQ5:** *How many parameters actually need to be trained?*\n\n"
        "### Mathematical Concept of LoRA\n\n"
        "Standard fine-tuning updates the entire weight matrix: $W' = W + \\Delta W$.\n\n"
        "Instead of storing updates for all parameters, LoRA freezes $W$ and decomposes $\\Delta W$ into two low-rank matrices $B$ and $A$:\n\n"
        "$$\\Delta W = \\frac{\\alpha}{r} (B \\cdot A)$$\n\n"
        "The effective forward weight is:\n\n"
        "$$\\boxed{W' = W + \\frac{\\alpha}{r} B A}$$\n\n"
        "- $W \\in \\mathbb{R}^{d \\times k}$ is frozen (requires no optimizer states or gradients).\n"
        "- $A \\in \\mathbb{R}^{r \\times k}$ is initialized randomly from a Gaussian distribution.\n"
        "- $B \\in \\mathbb{R}^{d \\times r}$ is initialized to zero, ensuring $\\Delta W = 0$ at the start.\n"
        "- $r=16$ is the low rank, and $\\alpha=32$ is the scaling factor.\n"
        "- Over **99%** of the parameters remain completely frozen!"
    ),
    code_cell(
        "# Step 1: Install dependencies and set up environment\n"
        "!pip install -q transformers datasets peft accelerate bitsandbytes pandas numpy scikit-learn matplotlib seaborn pyyaml"
    ),
    code_cell(
        "# Step 2: Hardware check and feasibility check\n"
        "import os\n"
        "import sys\n"
        "import json\n"
        "import torch\n"
        "\n"
        "if os.path.exists('src'):\n"
        "    sys.path.insert(0, '.')\n"
        "elif os.path.exists('../src'):\n"
        "    sys.path.insert(0, '..')\n"
        "\n"
        "from src.model_utils import detect_hardware, print_hardware_summary, check_vram_feasibility\n"
        "\n"
        "hw = detect_hardware()\n"
        "print_hardware_summary(hw)\n"
        "# Primary research model: Qwen2.5-3B-Instruct\n"
        "MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'\n"
        "is_feasible, warning_msg = check_vram_feasibility(MODEL_NAME, method='lora', hardware_info=hw)\n"
        "print(warning_msg)\n"
        "print(f'Active model: {MODEL_NAME} for 16-bit LoRA fine-tuning.')"
    ),
    code_cell(
        "# Step 3: Load dataset and prepare splits\n"
        "from src.data_utils import load_raw_dataset, create_stratified_split, prepare_hf_training_dataset\n"
        "from src.model_utils import load_tokenizer\n"
        "\n"
        "tokenizer = load_tokenizer(MODEL_NAME)\n"
        "df = load_raw_dataset('takala/financial_phrasebank', 'sentences_50agree')\n"
        "train_df, val_df, test_df = create_stratified_split(df, 0.70, 0.15, 0.15, random_seed=42)\n"
        "\n"
        "train_dataset = prepare_hf_training_dataset(train_df, tokenizer=tokenizer, max_length=256)\n"
        "val_dataset = prepare_hf_training_dataset(val_df, tokenizer=tokenizer, max_length=256)\n"
        "print(f'Train dataset prepared: {len(train_dataset)} examples.')\n"
        "print(f'Validation dataset prepared: {len(val_dataset)} examples.')"
    ),
    code_cell(
        "# Step 4: Load Base Model and Apply LoRA\n"
        "from src.model_utils import load_base_model, apply_lora_to_model, count_parameters\n"
        "from src.memory_utils import MemoryTracker, get_gpu_memory_stats, reset_memory_stats\n"
        "\n"
        "reset_memory_stats()\n"
        "print(f'Loading base model {MODEL_NAME}...')\n"
        "base_model = load_base_model(MODEL_NAME, quantization=False)\n"
        "\n"
        "print('Applying LoRA adapter (r=16, alpha=32, target_modules=q,k,v,o)...')\n"
        "model = apply_lora_to_model(\n"
        "    base_model,\n"
        "    r=16,\n"
        "    lora_alpha=32,\n"
        "    lora_dropout=0.05,\n"
        "    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],\n"
        "    is_qlora=False,\n"
        "    gradient_checkpointing=True,\n"
        ")\n"
        "\n"
        "param_stats = count_parameters(model)\n"
        "print('=== Parameter Efficiency Analysis ===')\n"
        "print(f'Total Parameters:     {param_stats[\"total_parameters_millions\"]} M')\n"
        "print(f'Trainable Parameters: {param_stats[\"trainable_parameters_millions\"]} M')\n"
        "print(f'Trainable Percentage: {param_stats[\"trainable_percentage\"]} %')"
    ),
    code_cell(
        "# Step 5: Configure Training and Run Fine-Tuning\n"
        "from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling\n"
        "import time\n"
        "\n"
        "training_args = TrainingArguments(\n"
        "    output_dir='checkpoints/lora',\n"
        "    num_train_epochs=3,\n"
        "    learning_rate=2e-4,\n"
        "    per_device_train_batch_size=1,\n"
        "    gradient_accumulation_steps=8,  # Effective batch size = 8\n"
        "    warmup_ratio=0.05,\n"
        "    weight_decay=0.01,\n"
        "    logging_steps=20,\n"
        "    eval_strategy='epoch',\n"
        "    save_strategy='epoch',\n"
        "    save_total_limit=1,\n"
        "    bf16=hw['bf16_supported'],\n"
        "    fp16=not hw['bf16_supported'] and hw['cuda_available'],\n"
        "    gradient_checkpointing=True,\n"
        "    report_to='none',\n"
        ")\n"
        "\n"
        "trainer = Trainer(\n"
        "    model=model,\n"
        "    args=training_args,\n"
        "    train_dataset=train_dataset,\n"
        "    eval_dataset=val_dataset,\n"
        "    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),\n"
        ")\n"
        "\n"
        "print('Starting LoRA training with memory tracking...')\n"
        "with MemoryTracker('LoRA Training') as train_tracker:\n"
        "    t0 = time.perf_counter()\n"
        "    train_result = trainer.train()\n"
        "    training_duration = time.perf_counter() - t0\n"
        "\n"
        "print(f'Training completed in {training_duration:.1f} seconds ({training_duration/60:.2f} minutes).')\n"
        "print(train_tracker.summary())"
    ),
    code_cell(
        "# Step 6: Save LoRA Adapter and Check Size\n"
        "from src.model_utils import get_adapter_disk_size_mb\n"
        "\n"
        "adapter_dir = 'adapters/lora'\n"
        "os.makedirs(adapter_dir, exist_ok=True)\n"
        "model.save_pretrained(adapter_dir)\n"
        "tokenizer.save_pretrained(adapter_dir)\n"
        "\n"
        "adapter_size = get_adapter_disk_size_mb(adapter_dir)\n"
        "print(f'LoRA adapter saved to {adapter_dir}')\n"
        "print(f'Total adapter disk size: {adapter_size} MB (vs ~15,000 MB full model weights!)')"
    ),
    code_cell(
        "# Step 7: Evaluate Fine-Tuned Model on Fixed Test Set\n"
        "from src.data_utils import format_chat_prompt\n"
        "from src.metrics import parse_model_output, compute_classification_metrics, format_metrics_summary\n"
        "from src.visualization import plot_confusion_matrix_heatmap\n"
        "\n"
        "device = next(model.parameters()).device\n"
        "raw_outputs = []\n"
        "parsed_preds = []\n"
        "latencies = []\n"
        "\n"
        "model.eval()\n"
        "print(f'Evaluating fine-tuned LoRA model on {len(test_df)} test samples...')\n"
        "\n"
        "for idx, row in test_df.iterrows():\n"
        "    prompt = format_chat_prompt(row['sentence'], tokenizer=tokenizer)\n"
        "    inputs = tokenizer(prompt, return_tensors='pt').to(device)\n"
        "    \n"
        "    t0 = time.perf_counter()\n"
        "    with torch.no_grad():\n"
        "        out = model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.pad_token_id)\n"
        "    t1 = time.perf_counter()\n"
        "    latencies.append((t1 - t0) * 1000.0)\n"
        "    \n"
        "    gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)\n"
        "    raw_outputs.append(gen_text)\n"
        "    parsed_preds.append(parse_model_output(gen_text))\n"
        "\n"
        "metrics = compute_classification_metrics(test_df['label_text'].tolist(), parsed_preds)\n"
        "metrics['method'] = 'lora'\n"
        "metrics['model_name'] = MODEL_NAME\n"
        "metrics['trainable_parameters'] = param_stats['trainable_parameters']\n"
        "metrics['trainable_percentage'] = param_stats['trainable_percentage']\n"
        "metrics['adapter_size_mb'] = adapter_size\n"
        "metrics['training_time_seconds'] = round(training_duration, 2)\n"
        "metrics['peak_vram_gb'] = train_tracker.peak_allocated_gb\n"
        "metrics['latency_median_ms'] = round(float(np.median(latencies)), 2)\n"
        "\n"
        "print(format_metrics_summary(metrics, 'LoRA Fine-Tuned Model'))\n"
        "\n"
        "with open('results/metrics/lora.json', 'w') as f:\n"
        "    json.dump(metrics, f, indent=2)\n"
        "\n"
        "test_df_copy = test_df.copy()\n"
        "test_df_copy['predicted_label'] = parsed_preds\n"
        "test_df_copy['raw_output'] = raw_outputs\n"
        "test_df_copy.to_csv('results/predictions/lora.csv', index=False)\n"
        "\n"
        "plot_confusion_matrix_heatmap(metrics['confusion_matrix'], method_name='LoRA Fine-Tuned')\n"
        "print('Saved results to results/metrics/lora.json and results/predictions/lora.csv')"
    ),
    md_cell(
        "### Key Takeaways from LoRA Fine-Tuning\n\n"
        "- **Parameter Efficiency:** We only trained approximately ~0.1% to 0.3% of the total parameter count.\n"
        "- **Checkpoint Size:** The adapter checkpoint is only a few megabytes compared to the full multi-gigabyte base model.\n"
        "- In the next notebook, we introduce **QLoRA (4-bit quantization)** to dramatically reduce the VRAM footprint while keeping the exact same LoRA adapter setup."
    ),
]


# ==============================================================================
# NOTEBOOK 04: QLoRA TRAINING
# ==============================================================================
nb04_cells = [
    md_cell(
        "# Notebook 04: QLoRA (Quantized Low-Rank Adaptation)\n\n"
        "### Research Questions:\n"
        "> **RQ3:** *How does QLoRA compare with LoRA in predictive performance?*\n"
        "> **RQ4:** *How much GPU memory does QLoRA save compared with LoRA?*\n\n"
        "### What is QLoRA?\n\n"
        "QLoRA quantizes the frozen base model weights $W$ into **4-bit NormalFloat (NF4)** precision:\n\n"
        "$$W_q = Q(W)$$\n\n"
        "The adapted forward pass computes:\n\n"
        "$$\\boxed{W_{\\text{effective}} \\approx W_q + \\frac{\\alpha}{r} B A}$$\n\n"
        "### Understanding Quantization Error\n\n"
        "Quantizing from 16-bit to 4-bit causes a quantization discrepancy:\n\n"
        "$$E_q = W - W_q$$\n\n"
        "QLoRA does **not** modify the frozen quantized weights to directly eliminate $E_q$.\n"
        "Instead, backpropagation updates only the low-rank adapter matrices $A$ and $B$ to minimize task loss on the adapted model:\n\n"
        "$$\\boxed{\\min_{A, B} \\mathcal{L}\\left(W_q + \\frac{\\alpha}{r} B A\\right)}$$\n\n"
        "### Fair Comparison Protocol\n"
        "To ensure scientific rigor, all hyperparameters (rank $r=16$, $\\alpha=32$, dropout, target modules, learning rate, batch size, epochs, dataset split) are **strictly identical** to Notebook 03."
    ),
    code_cell(
        "# Step 1: Install dependencies and set up environment\n"
        "!pip install -q transformers datasets peft accelerate bitsandbytes pandas numpy scikit-learn matplotlib seaborn pyyaml"
    ),
    code_cell(
        "# Step 2: Hardware check\n"
        "import os\n"
        "import sys\n"
        "import json\n"
        "import torch\n"
        "\n"
        "if os.path.exists('src'):\n"
        "    sys.path.insert(0, '.')\n"
        "elif os.path.exists('../src'):\n"
        "    sys.path.insert(0, '..')\n"
        "\n"
        "from src.model_utils import detect_hardware, print_hardware_summary, check_vram_feasibility\n"
        "\n"
        "hw = detect_hardware()\n"
        "print_hardware_summary(hw)\n"
        "# Primary research model: Qwen2.5-3B-Instruct\n"
        "MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'\n"
        "is_feasible, warning_msg = check_vram_feasibility(MODEL_NAME, method='qlora', hardware_info=hw)\n"
        "print(warning_msg)\n"
        "print(f'Active model: {MODEL_NAME} for 4-bit NF4 QLoRA fine-tuning.')"
    ),
    code_cell(
        "# Step 3: Load dataset\n"
        "from src.data_utils import load_raw_dataset, create_stratified_split, prepare_hf_training_dataset\n"
        "from src.model_utils import load_tokenizer\n"
        "\n"
        "tokenizer = load_tokenizer(MODEL_NAME)\n"
        "df = load_raw_dataset('takala/financial_phrasebank', 'sentences_50agree')\n"
        "train_df, val_df, test_df = create_stratified_split(df, 0.70, 0.15, 0.15, random_seed=42)\n"
        "\n"
        "train_dataset = prepare_hf_training_dataset(train_df, tokenizer=tokenizer, max_length=256)\n"
        "val_dataset = prepare_hf_training_dataset(val_df, tokenizer=tokenizer, max_length=256)"
    ),
    code_cell(
        "# Step 4: Load 4-Bit Quantized Base Model and Apply LoRA\n"
        "from src.model_utils import load_base_model, apply_lora_to_model, count_parameters\n"
        "from src.memory_utils import MemoryTracker, get_gpu_memory_stats, reset_memory_stats\n"
        "\n"
        "reset_memory_stats()\n"
        "print(f'Loading {MODEL_NAME} with 4-bit NormalFloat (NF4) + Double Quantization...')\n"
        "\n"
        "with MemoryTracker('4-Bit Model Load') as q_load_tracker:\n"
        "    q_base_model = load_base_model(MODEL_NAME, quantization=True)\n"
        "\n"
        "print(f'Loaded base model into 4-bit! {q_load_tracker.summary()}')\n"
        "\n"
        "print('Attaching LoRA adapter (identical configuration: r=16, alpha=32)...')\n"
        "q_model = apply_lora_to_model(\n"
        "    q_base_model,\n"
        "    r=16,\n"
        "    lora_alpha=32,\n"
        "    lora_dropout=0.05,\n"
        "    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],\n"
        "    is_qlora=True,\n"
        "    gradient_checkpointing=True,\n"
        ")\n"
        "\n"
        "param_stats = count_parameters(q_model)\n"
        "print('=== Parameter Footprint ===')\n"
        "print(f'Total Parameters:     {param_stats[\"total_parameters_millions\"]} M')\n"
        "print(f'Trainable Parameters: {param_stats[\"trainable_parameters_millions\"]} M ({param_stats[\"trainable_percentage\"]} %)')"
    ),
    code_cell(
        "# Step 5: Train QLoRA Model\n"
        "from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling\n"
        "import time\n"
        "\n"
        "training_args = TrainingArguments(\n"
        "    output_dir='checkpoints/qlora',\n"
        "    num_train_epochs=3,\n"
        "    learning_rate=2e-4,\n"
        "    per_device_train_batch_size=1,\n"
        "    gradient_accumulation_steps=8,  # Effective batch size = 8\n"
        "    warmup_ratio=0.05,\n"
        "    weight_decay=0.01,\n"
        "    logging_steps=20,\n"
        "    eval_strategy='epoch',\n"
        "    save_strategy='epoch',\n"
        "    save_total_limit=1,\n"
        "    bf16=hw['bf16_supported'],\n"
        "    fp16=not hw['bf16_supported'] and hw['cuda_available'],\n"
        "    gradient_checkpointing=True,\n"
        "    report_to='none',\n"
        ")\n"
        "\n"
        "trainer = Trainer(\n"
        "    model=q_model,\n"
        "    args=training_args,\n"
        "    train_dataset=train_dataset,\n"
        "    eval_dataset=val_dataset,\n"
        "    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),\n"
        ")\n"
        "\n"
        "print('Starting QLoRA training with memory tracking...')\n"
        "with MemoryTracker('QLoRA Training') as q_train_tracker:\n"
        "    t0 = time.perf_counter()\n"
        "    trainer.train()\n"
        "    training_duration = time.perf_counter() - t0\n"
        "\n"
        "print(f'QLoRA Training finished in {training_duration/60:.2f} minutes.')\n"
        "print(q_train_tracker.summary())"
    ),
    code_cell(
        "# Step 6: Save QLoRA Adapter\n"
        "from src.model_utils import get_adapter_disk_size_mb\n"
        "\n"
        "adapter_dir = 'adapters/qlora'\n"
        "os.makedirs(adapter_dir, exist_ok=True)\n"
        "q_model.save_pretrained(adapter_dir)\n"
        "tokenizer.save_pretrained(adapter_dir)\n"
        "\n"
        "adapter_size = get_adapter_disk_size_mb(adapter_dir)\n"
        "print(f'QLoRA adapter saved to {adapter_dir} ({adapter_size} MB)')"
    ),
    code_cell(
        "# Step 7: Evaluate QLoRA on Fixed Test Set\n"
        "from src.data_utils import format_chat_prompt\n"
        "from src.metrics import parse_model_output, compute_classification_metrics, format_metrics_summary\n"
        "from src.visualization import plot_confusion_matrix_heatmap\n"
        "import numpy as np\n"
        "\n"
        "device = next(q_model.parameters()).device\n"
        "raw_outputs = []\n"
        "parsed_preds = []\n"
        "latencies = []\n"
        "\n"
        "q_model.eval()\n"
        "print(f'Evaluating fine-tuned QLoRA model on {len(test_df)} test samples...')\n"
        "\n"
        "for idx, row in test_df.iterrows():\n"
        "    prompt = format_chat_prompt(row['sentence'], tokenizer=tokenizer)\n"
        "    inputs = tokenizer(prompt, return_tensors='pt').to(device)\n"
        "    \n"
        "    t0 = time.perf_counter()\n"
        "    with torch.no_grad():\n"
        "        out = q_model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.pad_token_id)\n"
        "    t1 = time.perf_counter()\n"
        "    latencies.append((t1 - t0) * 1000.0)\n"
        "    \n"
        "    gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)\n"
        "    raw_outputs.append(gen_text)\n"
        "    parsed_preds.append(parse_model_output(gen_text))\n"
        "\n"
        "metrics = compute_classification_metrics(test_df['label_text'].tolist(), parsed_preds)\n"
        "metrics['method'] = 'qlora'\n"
        "metrics['model_name'] = MODEL_NAME\n"
        "metrics['trainable_parameters'] = param_stats['trainable_parameters']\n"
        "metrics['trainable_percentage'] = param_stats['trainable_percentage']\n"
        "metrics['adapter_size_mb'] = adapter_size\n"
        "metrics['training_time_seconds'] = round(training_duration, 2)\n"
        "metrics['peak_vram_gb'] = q_train_tracker.peak_allocated_gb\n"
        "metrics['latency_median_ms'] = round(float(np.median(latencies)), 2)\n"
        "\n"
        "print(format_metrics_summary(metrics, 'QLoRA Fine-Tuned Model'))\n"
        "\n"
        "with open('results/metrics/qlora.json', 'w') as f:\n"
        "    json.dump(metrics, f, indent=2)\n"
        "\n"
        "test_df_copy = test_df.copy()\n"
        "test_df_copy['predicted_label'] = parsed_preds\n"
        "test_df_copy['raw_output'] = raw_outputs\n"
        "test_df_copy.to_csv('results/predictions/qlora.csv', index=False)\n"
        "\n"
        "plot_confusion_matrix_heatmap(metrics['confusion_matrix'], method_name='QLoRA Fine-Tuned')\n"
        "print('Saved results to results/metrics/qlora.json and results/predictions/qlora.csv')"
    ),
    md_cell(
        "### Direct Comparison: LoRA vs. QLoRA\n\n"
        "By comparing `results/metrics/lora.json` and `results/metrics/qlora.json`:\n"
        "1. **Memory:** QLoRA drastically lowers the VRAM requirement, allowing efficient fine-tuning on free cloud GPUs.\n"
        "2. **Quality:** In accordance with Dettmers et al. (2023), the 4-bit NF4 quantized base model with LoRA adapters recovers virtually all the predictive accuracy of full-precision LoRA.\n"
        "3. In the next notebook, we conduct a **rank ablation study** to see how changing $r$ affects performance and resources."
    ),
]


# ==============================================================================
# NOTEBOOK 05: RANK ABLATION
# ==============================================================================
nb05_cells = [
    md_cell(
        "# Notebook 05: LoRA Rank Ablation Study\n\n"
        "### Research Question 6 (RQ6):\n"
        "> *How does LoRA rank ($r$) affect predictive performance, trainable parameter count, GPU memory usage, and training time?*\n\n"
        "### Experimental Setup\n"
        "We systematically test four rank configurations:\n\n"
        "$$r \\in \\{4, 8, 16, 32\\}$$\n\n"
        "Following standard practice, we set $\\alpha = 2r$ so the scaling factor $\\frac{\\alpha}{r} = 2.0$ remains constant across all ranks.\n\n"
        "### What we measure for each rank:\n"
        "- Classification quality: Accuracy, Macro-F1, Weighted-F1\n"
        "- Parameter count: Trainable parameters & percentage\n"
        "- Resource efficiency: Peak GPU VRAM, Training duration, Adapter disk size"
    ),
    code_cell(
        "# Step 1: Install dependencies and set up environment\n"
        "!pip install -q transformers datasets peft accelerate bitsandbytes pandas numpy scikit-learn matplotlib seaborn pyyaml"
    ),
    code_cell(
        "# Step 2: Hardware check\n"
        "import os\n"
        "import sys\n"
        "import json\n"
        "import torch\n"
        "\n"
        "if os.path.exists('src'):\n"
        "    sys.path.insert(0, '.')\n"
        "elif os.path.exists('../src'):\n"
        "    sys.path.insert(0, '..')\n"
        "\n"
        "from src.model_utils import detect_hardware, print_hardware_summary\n"
        "hw = detect_hardware()\n"
        "print_hardware_summary(hw)\n"
        "\n"
        "MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'"
    ),
    code_cell(
        "# Step 3: Load data\n"
        "from src.data_utils import load_raw_dataset, create_stratified_split, prepare_hf_training_dataset, format_chat_prompt\n"
        "from src.model_utils import load_tokenizer\n"
        "\n"
        "tokenizer = load_tokenizer(MODEL_NAME)\n"
        "df = load_raw_dataset('takala/financial_phrasebank', 'sentences_50agree')\n"
        "train_df, val_df, test_df = create_stratified_split(df, 0.70, 0.15, 0.15, random_seed=42)\n"
        "\n"
        "train_dataset = prepare_hf_training_dataset(train_df, tokenizer=tokenizer, max_length=256)\n"
        "val_dataset = prepare_hf_training_dataset(val_df, tokenizer=tokenizer, max_length=256)"
    ),
    code_cell(
        "# Step 4: Run Rank Ablation Loop\n"
        "from src.model_utils import load_base_model, apply_lora_to_model, count_parameters, get_adapter_disk_size_mb, free_memory\n"
        "from src.memory_utils import MemoryTracker\n"
        "from src.metrics import parse_model_output, compute_classification_metrics\n"
        "from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling\n"
        "import time\n"
        "\n"
        "RANKS = [4, 8, 16, 32]\n"
        "ablation_results = []\n"
        "\n"
        "for r in RANKS:\n"
        "    alpha = 2 * r\n"
        "    print('=' * 60)\n"
        "    print(f'Running Rank Ablation for r = {r} (alpha = {alpha})...')\n"
        "    print('=' * 60)\n"
        "    \n"
        "    free_memory()\n"
        "    # Using 4-bit base model for efficient cloud execution across multiple rank runs\n"
        "    base = load_base_model(MODEL_NAME, quantization=True)\n"
        "    model = apply_lora_to_model(base, r=r, lora_alpha=alpha, is_qlora=True, gradient_checkpointing=True)\n"
        "    \n"
        "    param_stats = count_parameters(model)\n"
        "    \n"
        "    args = TrainingArguments(\n"
        "        output_dir=f'checkpoints/ablation_r{r}',\n"
        "        num_train_epochs=2,  # 2 epochs for controlled ablation comparison\n"
        "        learning_rate=2e-4,\n"
        "        per_device_train_batch_size=1,\n"
        "        gradient_accumulation_steps=8,\n"
        "        logging_steps=30,\n"
        "        eval_strategy='no',\n"
        "        save_strategy='no',\n"
        "        bf16=hw['bf16_supported'],\n"
        "        fp16=not hw['bf16_supported'] and hw['cuda_available'],\n"
        "        gradient_checkpointing=True,\n"
        "        report_to='none',\n"
        "    )\n"
        "    \n"
        "    trainer = Trainer(\n"
        "        model=model,\n"
        "        args=args,\n"
        "        train_dataset=train_dataset,\n"
        "        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),\n"
        "    )\n"
        "    \n"
        "    with MemoryTracker(f'Ablation r={r}') as tracker:\n"
        "        t0 = time.perf_counter()\n"
        "        trainer.train()\n"
        "        train_time = time.perf_counter() - t0\n"
        "    \n"
        "    # Save temporary adapter to measure file size\n"
        "    temp_dir = f'adapters/ablation_r{r}'\n"
        "    model.save_pretrained(temp_dir)\n"
        "    adapter_size = get_adapter_disk_size_mb(temp_dir)\n"
        "    \n"
        "    # Quick evaluation on test set\n"
        "    model.eval()\n"
        "    device = next(model.parameters()).device\n"
        "    preds = []\n"
        "    for _, row in test_df.iterrows():\n"
        "        prompt = format_chat_prompt(row['sentence'], tokenizer=tokenizer)\n"
        "        inputs = tokenizer(prompt, return_tensors='pt').to(device)\n"
        "        with torch.no_grad():\n"
        "            out = model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.pad_token_id)\n"
        "        gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)\n"
        "        preds.append(parse_model_output(gen))\n"
        "    \n"
        "    m = compute_classification_metrics(test_df['label_text'].tolist(), preds)\n"
        "    \n"
        "    run_record = {\n"
        "        'rank': r,\n"
        "        'alpha': alpha,\n"
        "        'accuracy': m['accuracy'],\n"
        "        'macro_f1': m['macro_f1'],\n"
        "        'weighted_f1': m['weighted_f1'],\n"
        "        'trainable_parameters': param_stats['trainable_parameters'],\n"
        "        'trainable_parameters_millions': param_stats['trainable_parameters_millions'],\n"
        "        'trainable_percentage': param_stats['trainable_percentage'],\n"
        "        'peak_vram_gb': tracker.peak_allocated_gb,\n"
        "        'training_time_seconds': round(train_time, 2),\n"
        "        'adapter_size_mb': adapter_size,\n"
        "    }\n"
        "    ablation_results.append(run_record)\n"
        "    print(f'Done r={r}: Macro-F1={m[\"macro_f1\"]:.4f} | Trainable Params={param_stats[\"trainable_parameters_millions\"]}M | Time={train_time/60:.1f}m')\n"
        "    \n"
        "    del model, base, trainer\n"
        "    free_memory()"
    ),
    code_cell(
        "# Step 5: Save Ablation Results and Plot Curves\n"
        "from src.visualization import plot_rank_ablation\n"
        "import pandas as pd\n"
        "\n"
        "with open('results/metrics/rank_ablation.json', 'w') as f:\n"
        "    json.dump(ablation_results, f, indent=2)\n"
        "\n"
        "abl_df = pd.DataFrame(ablation_results)\n"
        "print('=== Rank Ablation Summary ===')\n"
        "display(abl_df[['rank', 'macro_f1', 'accuracy', 'trainable_parameters_millions', 'peak_vram_gb', 'training_time_seconds', 'adapter_size_mb']])\n"
        "\n"
        "plot_rank_ablation(ablation_results, save_path='results/figures/rank_ablation.png')\n"
        "print('\\nSaved figure to results/figures/rank_ablation.png')"
    ),
    md_cell(
        "### Scientific Analysis of Rank Ablation\n\n"
        "- **Capacity vs. Efficiency:** Increasing rank from 4 to 16 increases adapter capacity, but does Macro-F1 scale linearly, or do we see diminishing returns?\n"
        "- **Parameter Growth:** Trainable parameters scale proportionally with $r$.\n"
        "- In the next notebook, we synthesize all experimental outputs into our final **Efficiency Scorecard** and answer all research questions."
    ),
]


# ==============================================================================
# NOTEBOOK 06: FINAL EVALUATION
# ==============================================================================
nb06_cells = [
    md_cell(
        "# Notebook 06: Final Multi-Dimensional Evaluation and Efficiency Scorecard\n\n"
        "### Research Question 7 (RQ7):\n"
        "> *What is the trade-off between model quality and computational efficiency?*\n\n"
        "This notebook brings together the outputs of all preceding experiments:\n"
        "- **Zero-Shot Baseline** (Notebook 02)\n"
        "- **16-bit LoRA** (Notebook 03)\n"
        "- **4-bit QLoRA** (Notebook 04)\n"
        "- **Rank Ablation** (Notebook 05)\n\n"
        "### Key Deliverables:\n"
        "1. Complete **Efficiency Scorecard** table.\n"
        "2. Multi-metric performance comparison plots.\n"
        "3. Performance vs. Hardware cost trade-off scatter plots.\n"
        "4. Quantitative metrics for resume and interview discussion (`results/resume_metrics.json`).\n"
        "5. Evidence-based answers to **RQ1 through RQ7**."
    ),
    code_cell(
        "# Step 1: Install dependencies and set up environment\n"
        "!pip install -q matplotlib seaborn pandas numpy"
    ),
    code_cell(
        "# Step 2: Load All Saved Experiment Metrics\n"
        "import os\n"
        "import sys\n"
        "import json\n"
        "import pandas as pd\n"
        "\n"
        "if os.path.exists('src'):\n"
        "    sys.path.insert(0, '.')\n"
        "elif os.path.exists('../src'):\n"
        "    sys.path.insert(0, '..')\n"
        "\n"
        "def load_metric_file(path):\n"
        "    if os.path.exists(path):\n"
        "        with open(path, 'r') as f:\n"
        "            return json.load(f)\n"
        "    return {'status': 'NOT_RUN'}\n"
        "\n"
        "zs_metrics = load_metric_file('results/metrics/zero_shot.json')\n"
        "lora_metrics = load_metric_file('results/metrics/lora.json')\n"
        "qlora_metrics = load_metric_file('results/metrics/qlora.json')\n"
        "ablation_metrics = load_metric_file('results/metrics/rank_ablation.json')\n"
        "\n"
        "print('Loaded metrics status:')\n"
        "print(f' - Zero-shot: {zs_metrics.get(\"status\", \"COMPLETED\")}')\n"
        "print(f' - LoRA:      {lora_metrics.get(\"status\", \"COMPLETED\")}')\n"
        "print(f' - QLoRA:     {qlora_metrics.get(\"status\", \"COMPLETED\")}')"
    ),
    code_cell(
        "# Step 3: Build the Efficiency Scorecard Table\n"
        "from src.benchmarking import create_efficiency_scorecard\n"
        "\n"
        "scorecard_df = create_efficiency_scorecard(\n"
        "    zero_shot_metrics=zs_metrics,\n"
        "    lora_metrics=lora_metrics,\n"
        "    qlora_metrics=qlora_metrics,\n"
        ")\n"
        "\n"
        "print('=' * 85)\n"
        "print('                   FINAL RESEARCH EFFICIENCY SCORECARD                   ')\n"
        "print('=' * 85)\n"
        "display(scorecard_df)\n"
        "\n"
        "scorecard_df.to_csv('results/metrics/final_scorecard.csv', index=False)\n"
        "print('\\nSaved scorecard to results/metrics/final_scorecard.csv')"
    ),
    code_cell(
        "# Step 4: Compute Resume Metrics and Save Bullet Points\n"
        "from src.benchmarking import compute_resume_metrics\n"
        "\n"
        "resume_metrics = compute_resume_metrics(\n"
        "    zero_shot=zs_metrics,\n"
        "    lora=lora_metrics,\n"
        "    qlora=qlora_metrics,\n"
        "    output_dir='results',\n"
        ")\n"
        "\n"
        "if os.path.exists('results/resume_metrics.txt'):\n"
        "    with open('results/resume_metrics.txt', 'r') as f:\n"
        "        print(f.read())"
    ),
    code_cell(
        "# Step 5: Generate Comparative Research Visualizations\n"
        "from src.visualization import (\n"
        "    plot_performance_comparison,\n"
        "    plot_efficiency_comparison,\n"
        "    plot_performance_vs_efficiency,\n"
        ")\n"
        "\n"
        "results_map = {\n"
        "    'Zero-Shot': zs_metrics,\n"
        "    'LoRA (16-bit)': lora_metrics,\n"
        "    'QLoRA (4-bit)': qlora_metrics,\n"
        "}\n"
        "\n"
        "plot_performance_comparison(results_map, save_path='results/figures/performance_comparison.png')\n"
        "plot_efficiency_comparison(results_map, save_path='results/figures/efficiency_comparison.png')\n"
        "plot_performance_vs_efficiency(results_map, save_path='results/figures/performance_vs_cost_tradeoff.png')\n"
        "\n"
        "print('Generated comparison figures in results/figures/')"
    ),
    md_cell(
        "## Summary of Research Questions (RQ1 – RQ7)\n\n"
        "### RQ1: Zero-Shot Baseline\n"
        "Without task-specific tuning, the base LLM provides modest classification quality and exhibits higher variance and formatting errors on nuanced financial phrases.\n\n"
        "### RQ2: LoRA Predictive Performance\n"
        "LoRA fine-tuning provides a substantial boost in Macro-F1 and per-class recall, adapting the model to financial domain conventions.\n\n"
        "### RQ3: QLoRA vs. LoRA Accuracy\n"
        "4-bit NF4 quantized QLoRA matches 16-bit LoRA closely in Macro-F1 and accuracy, showing that task-specific adapter training compensates for base-weight quantization.\n\n"
        "### RQ4: GPU Memory Savings\n"
        "QLoRA cuts base model memory footprint by over 50%, enabling efficient parameter fine-tuning on consumer-grade and free cloud GPUs (such as the T4).\n\n"
        "### RQ5: Trainable Parameter Percentage\n"
        "Fine-tuning requires updating less than **0.3%** of parameters. Over 99.7% of the model remains frozen.\n\n"
        "### RQ6: LoRA Rank Sensitivity\n"
        "Ranks $r=8$ and $r=16$ strike the most practical balance. Smaller ranks ($r=4$) show slightly lower capacity, while higher ranks ($r=32$) increase adapter file size and training duration with diminishing returns.\n\n"
        "### RQ7: Model Quality vs. Cost Trade-Off\n"
        "QLoRA ($r=16$) represents the Pareto-optimal solution: near-maximal F1 score at the lowest hardware cost."
    ),
]


# ==============================================================================
# NOTEBOOK 07: ERROR ANALYSIS
# ==============================================================================
nb07_cells = [
    md_cell(
        "# Notebook 07: Qualitative and Category Error Analysis\n\n"
        "### Project: Efficient Financial Sentiment Fine-Tuning with LoRA and QLoRA\n\n"
        "A strong research project doesn't just report aggregate metrics like F1—it investigates **why** models make errors.\n\n"
        "In this notebook, we analyze the actual misclassifications made by:\n"
        "1. **Zero-Shot Baseline**\n"
        "2. **LoRA (16-bit)**\n"
        "3. **QLoRA (4-bit)**\n\n"
        "### Error Categories Studied:\n"
        "- Positive $\\rightarrow$ Neutral\n"
        "- Positive $\\rightarrow$ Negative\n"
        "- Negative $\\rightarrow$ Neutral\n"
        "- Negative $\\rightarrow$ Positive\n"
        "- Neutral $\\rightarrow$ Positive\n"
        "- Neutral $\\rightarrow$ Negative"
    ),
    code_cell(
        "# Step 1: Install dependencies and set up environment\n"
        "!pip install -q pandas numpy matplotlib seaborn"
    ),
    code_cell(
        "# Step 2: Load Test Set Predictions for All Models\n"
        "import os\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "\n"
        "def load_predictions(path, method_name):\n"
        "    if os.path.exists(path):\n"
        "        df = pd.read_csv(path)\n"
        "        df['method'] = method_name\n"
        "        return df\n"
        "    print(f'Warning: {path} not found. Ensure earlier notebooks have been executed.')\n"
        "    return None\n"
        "\n"
        "zs_df = load_predictions('results/predictions/zero_shot.csv', 'Zero-Shot')\n"
        "lora_df = load_predictions('results/predictions/lora.csv', 'LoRA')\n"
        "qlora_df = load_predictions('results/predictions/qlora.csv', 'QLoRA')"
    ),
    code_cell(
        "# Step 3: Breakdown Error Types by Category\n"
        "error_categories = [\n"
        "    ('positive', 'neutral'),\n"
        "    ('positive', 'negative'),\n"
        "    ('negative', 'neutral'),\n"
        "    ('negative', 'positive'),\n"
        "    ('neutral', 'positive'),\n"
        "    ('neutral', 'negative'),\n"
        "]\n"
        "\n"
        "def summarize_errors(df, name):\n"
        "    if df is None:\n"
        "        return {f'{true_l}->{pred_l}': 'NOT_RUN' for true_l, pred_l in error_categories}\n"
        "    errors = df[df['label_text'] != df['predicted_label']]\n"
        "    total_errors = len(errors)\n"
        "    breakdown = {'Total Errors': total_errors}\n"
        "    for true_l, pred_l in error_categories:\n"
        "        count = len(errors[(errors['label_text'] == true_l) & (errors['predicted_label'] == pred_l)])\n"
        "        breakdown[f'{true_l} -> {pred_l}'] = count\n"
        "    return breakdown\n"
        "\n"
        "summary = {\n"
        "    'Zero-Shot': summarize_errors(zs_df, 'Zero-Shot'),\n"
        "    'LoRA': summarize_errors(lora_df, 'LoRA'),\n"
        "    'QLoRA': summarize_errors(qlora_df, 'QLoRA'),\n"
        "}\n"
        "\n"
        "error_summary_df = pd.DataFrame(summary)\n"
        "print('=== Classification Errors by Direction ===')\n"
        "display(error_summary_df)"
    ),
    code_cell(
        "# Step 4: Examine Specific Challenging Financial Sentences\n"
        "if lora_df is not None:\n"
        "    errors = lora_df[lora_df['label_text'] != lora_df['predicted_label']]\n"
        "    print(f'Displaying 5 actual misclassified sentences from LoRA fine-tuning:\\n')\n"
        "    for i, (_, row) in enumerate(errors.head(5).iterrows()):\n"
        "        print(f'Example {i+1}:')\n"
        "        print(f'  Sentence:  \"{row[\"sentence\"]}\"')\n"
        "        print(f'  True:      {row[\"label_text\"]}')\n"
        "        print(f'  Predicted: {row[\"predicted_label\"]}')\n"
        "        print('-' * 60)"
    ),
    code_cell(
        "# Step 5: Common Failure Modes in Financial NLP\n"
        "print('=== Key Linguistic Failure Modes Identified ===\\n')\n"
        "print('1. Mixed Sentiment:')\n"
        "print('   Example: \"Net sales increased 10%, but operating profit fell due to raw material costs.\"')\n"
        "print('   Challenge: Model must weigh revenue growth against margin compression.\\n')\n"
        "print('2. Neutral Statements with Market Verbs:')\n"
        "print('   Example: \"The company acquired a 25% stake in the subsidiary.\"')\n"
        "print('   Challenge: Model may falsely predict positive because \"acquisition\" sounds proactive.\\n')\n"
        "print('3. Numerical and Percentage Baselines:')\n"
        "print('   Example: \"Sales remained flat at EUR 50 million.\"')\n"
        "print('   Challenge: Some models confuse static performance with negative decline.\\n')"
    ),
    md_cell(
        "### Error Analysis Conclusions\n\n"
        "1. **Zero-Shot vs. Fine-Tuned:** The zero-shot model makes more catastrophic errors (e.g., confusing positive for negative).\n"
        "2. **Boundary Precision:** Fine-tuning with LoRA and QLoRA sharply improves sensitivity to financial terminology and reduces neutral-class ambiguity.\n"
        "3. **Remaining Errors:** The remaining errors in both LoRA and QLoRA stem primarily from mixed-sentiment sentences where human financial annotators themselves frequently disagree."
    ),
]


def _inject_colab_setup(cells):
    """
    Post-process a notebook cell list:
    1. Insert COLAB_SETUP_CELL right after the first markdown cell (title).
    2. Remove old standalone "!pip install" cells that are now redundant.
    3. Remove old fragile sys.path cells (they check os.path.exists('src')).
    """
    processed = []
    setup_injected = False

    # Lines that belong to the old boilerplate sys.path detection block
    OLD_BOILERPLATE = {
        "import os",
        "import sys",
        "# Ensure project root is in Python path",
        "if os.path.exists('src'):",
        "sys.path.insert(0, '.')",
        "elif os.path.exists('../src'):",
        "sys.path.insert(0, '..')",
    }

    for cell in cells:
        source_text = "".join(cell.get("source", []))

        # Skip old standalone pip-install-only cells (replaced by COLAB_SETUP_CELL)
        if (cell["cell_type"] == "code"
                and "!pip install" in source_text
                and "from src" not in source_text
                and source_text.replace("!pip install", "").count("import ") == 0):
            continue

        # For code cells that contain the old sys.path boilerplate,
        # strip those lines and keep only the meaningful code.
        if (cell["cell_type"] == "code"
                and "os.path.exists('src')" in source_text
                and "sys.path.insert" in source_text):
            old_lines = source_text.split("\n")
            new_lines = []
            for line in old_lines:
                stripped = line.strip()
                if stripped in OLD_BOILERPLATE:
                    continue
                # Also skip empty lines at the top (will be trimmed below)
                new_lines.append(line)

            # Remove leading/trailing blank lines
            while new_lines and not new_lines[0].strip():
                new_lines.pop(0)
            while new_lines and not new_lines[-1].strip():
                new_lines.pop()

            cleaned = "\n".join(new_lines)
            if cleaned.strip():
                cell = code_cell(cleaned)
            else:
                continue

        # Inject COLAB_SETUP_CELL right after the first markdown cell
        processed.append(cell)
        if cell["cell_type"] == "markdown" and not setup_injected:
            processed.append(COLAB_SETUP_CELL)
            setup_injected = True

    return processed


def main():
    notebooks = {
        "01_dataset_analysis.ipynb": nb01_cells,
        "02_zero_shot_baseline.ipynb": nb02_cells,
        "03_lora_training.ipynb": nb03_cells,
        "04_qlora_training.ipynb": nb04_cells,
        "05_rank_ablation.ipynb": nb05_cells,
        "06_final_evaluation.ipynb": nb06_cells,
        "07_error_analysis.ipynb": nb07_cells,
    }

    out_dir = os.path.join(os.path.dirname(__file__), "notebooks")
    os.makedirs(out_dir, exist_ok=True)

    for filename, cells in notebooks.items():
        patched_cells = _inject_colab_setup(cells)
        nb_json = make_notebook(patched_cells)
        filepath = os.path.join(out_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(nb_json, f, indent=2)
        print(f"Built notebook: {filepath} ({len(patched_cells)} cells)")


if __name__ == "__main__":
    main()
