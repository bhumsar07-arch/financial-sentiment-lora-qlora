# Cloud Run Guide: Google Colab & Kaggle

This guide explains in simple English how to run this research project on **Google Colab** and **Kaggle Notebooks** using free or paid cloud GPUs.

Everything is designed to be executed directly through the Jupyter notebooks in the `notebooks/` folder. You do not need to run commands from your local computer terminal.

---

## 1. Quick Hardware Overview

The project uses **`Qwen/Qwen2.5-3B-Instruct`** (3.09 Billion parameters) as the dedicated baseline model. This model comfortably fits standard cloud GPUs (such as the free Google Colab T4 15GB GPU or Kaggle GPUs) across all training paradigms:

| Experiment / Method | Precision | VRAM Required | Recommended Cloud GPU |
| :--- | :---: | :---: | :--- |
| **Zero-Shot Baseline** (`02`) | 16-bit | ~8.0 GB | Colab T4 (15GB) / Kaggle T4 or P100 |
| **LoRA Fine-Tuning** (`03`) | 16-bit | ~13.0 GB | Colab T4 (15GB) / Kaggle T4 or P100 |
| **QLoRA Fine-Tuning** (`04`) | 4-bit (NF4) | ~6.5 GB | Colab T4 (15GB) / Kaggle T4 or P100 |

### Dedicated Model Policy
- The dedicated model across all notebooks and scripts is strictly **`Qwen/Qwen2.5-3B-Instruct`**.
- No fallback models or silent model substitutions are used. Every experiment runs cleanly on this single dedicated foundation model.

---

## 2. Running on Google Colab

### Step 1: Open Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Click **File > Upload notebook** and select `01_dataset_analysis.ipynb` (or upload the whole project folder to Google Drive).

### Step 2: Enable GPU Accelerator
1. In the top menu, click **Runtime > Change runtime type**.
2. Under **Hardware accelerator**, select **GPU**.
3. Under **GPU type**, select **T4 GPU** (standard free GPU on Google Colab, with 15GB VRAM — more than enough for all experiments with Qwen2.5-3B-Instruct).
4. Click **Save**.

### Step 3: Clone or Upload the Repository
In the very first cell of the notebook, run:

```bash
!git clone https://github.com/YOUR_USERNAME/lora-qlora-financial-sentiment.git
%cd lora-qlora-financial-sentiment
```

*(Alternatively, if you uploaded the project folder directly to Google Drive, mount your Google Drive and navigate to the folder.)*

### Step 4: Install Dependencies
Every notebook includes an initial cell that installs the necessary libraries:

```bash
!pip install -r requirements.txt
```

### Step 5: Run the Notebooks in Order
Run each notebook cell by cell from top to bottom.

---

## 3. Running on Kaggle Notebooks

### Step 1: Create a Kaggle Notebook
1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **New Notebook**.
2. Click **File > Upload Notebook** to upload the notebook you want to run.

### Step 2: Enable GPU Accelerator
1. In the right-hand sidebar under **Notebook options**, find **Accelerator**.
2. Select **GPU T4 x 2** or **GPU P100**.
3. Under **Internet**, toggle the switch to **On** (required to download Hugging Face models and datasets).

### Step 3: Add Repository Code
You can add your GitHub repository as an input dataset, or clone it in the first cell:

```bash
!git clone https://github.com/YOUR_USERNAME/lora-qlora-financial-sentiment.git
%cd lora-qlora-financial-sentiment
!pip install -r requirements.txt
```

---

## 4. Recommended Execution Sequence

Run the notebooks in this exact numerical sequence:

### `01_dataset_analysis.ipynb`
- Downloads and analyzes Financial PhraseBank (`sentences_50agree`).
- Creates the fixed 70% train / 15% validation / 15% test stratified split with seed `42`.
- Generates data distribution charts saved to `results/figures/`.
- Saves split metadata to `results/metrics/split_metadata.json`.

### `02_zero_shot_baseline.ipynb`
- Loads the base model (`Qwen2.5-3B-Instruct`).
- Measures zero-shot classification performance on the test set.
- Records baseline latency, throughput, and GPU memory usage.
- Saves results to `results/metrics/zero_shot.json` and `results/predictions/zero_shot.csv`.

### `03_lora_training.ipynb`
- Applies 16-bit LoRA ($r=16, \alpha=32$) to attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
- Trains for 3 epochs with micro-batch 1 and gradient accumulation 8.
- Tracks peak GPU memory and logs parameter savings.
- Saves adapter to `adapters/lora/` and metrics to `results/metrics/lora.json`.

### `04_qlora_training.ipynb`
- Loads base model with 4-bit NormalFloat (NF4) and nested double quantization.
- Trains with the identical hyperparameter setup as Notebook 03 for a scientifically fair comparison.
- Records memory savings vs. LoRA.
- Saves adapter to `adapters/qlora/` and metrics to `results/metrics/qlora.json`.

### `05_rank_ablation.ipynb`
- Systematically investigates the effect of LoRA rank: $r \in \{4, 8, 16, 32\}$.
- Evaluates the impact of rank on Macro-F1, trainable parameter count, training time, and peak memory.
- Saves ablation charts and metrics to `results/metrics/rank_ablation.json`.

### `06_final_evaluation.ipynb`
- Loads all metric files and compiles the **Efficiency Scorecard**.
- Produces comparison bar charts and performance vs. cost trade-off scatter plots.
- Computes exact resume metrics saved to `results/resume_metrics.json` and `results/resume_metrics.txt`.
- Answers the seven research questions (RQ1 through RQ7).

### `07_error_analysis.ipynb`
- Analyzes actual classification mistakes made by each model on the test set.
- Groups errors by sentiment category (e.g., positive predicted as neutral).
- Identifies common financial phrasing patterns that cause errors (mixed sentiment, numerical statements, neutral ambiguity).

---

## 5. Saving and Downloading Your Results

After running the experiments, you can download all generated metrics, figures, and adapter weights.

In Google Colab, zip the output folders:

```python
!zip -r experiment_results.zip results/ adapters/
from google.colab import files
files.download("experiment_results.zip")
```

In Kaggle:
- Any file saved under `/kaggle/working/` (or your local directory) will appear in the **Output** tab in the right sidebar. Click **Download all**.

---

## 6. Troubleshooting Common Issues

### Issue 1: `CUDA Out of Memory` (OOM)
- **Solution A:** Ensure `gradient_checkpointing: true` is enabled in `configs/base.yaml`.
- **Solution B:** Keep `micro_batch_size: 1` and accumulate gradients with `gradient_accumulation_steps: 8`.
- **Solution C:** Decrease `max_sequence_length` from 256 to 128 if handling unusually long context prompts.
- **Solution D:** Switch to **QLoRA** (`04_qlora_training.ipynb`), which uses only ~6.5 GB VRAM.

### Issue 2: `bitsandbytes` installation or CUDA error
- On Colab or Kaggle, ensure GPU accelerator is enabled before running `pip install bitsandbytes`.
- If on Windows, note that native bitsandbytes 4-bit quantization requires specific Windows wheels; Colab and Kaggle Linux environments work automatically out of the box.

### Issue 3: Disconnected Runtime
- Free Google Colab instances may time out after periods of inactivity. Keep the browser tab open while training is running, or consider Colab Pro for background execution.
