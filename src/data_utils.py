"""
Data Utilities Module for Financial Sentiment Analysis.

Handles:
- Loading the Financial PhraseBank dataset (takala/financial_phrasebank, sentences_50agree)
- Stratified 70/15/15 dataset splitting with fixed seed 42
- Data quality validation (missing values, duplicates, leakage checks)
- Instruction prompt generation for zero-shot and fine-tuning
- Metadata tracking and persistence
"""

import os
import json
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np


# Label definitions for Financial PhraseBank
ID2LABEL: Dict[int, str] = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

LABEL2ID: Dict[str, int] = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

PROMPT_TEMPLATE: str = (
    "You are a financial sentiment classifier.\n\n"
    "Classify the following financial statement as exactly one of:\n\n"
    "positive\n"
    "negative\n"
    "neutral\n\n"
    "Statement:\n"
    "{sentence}\n\n"
    "Answer:"
)


def _download_and_parse_phrasebank(config_name: str = "sentences_50agree") -> pd.DataFrame:
    """
    Directly download and parse the Financial PhraseBank dataset from Hugging Face Hub archive.
    This works independently of the `datasets` library version, bypassing the
    'Dataset scripts are no longer supported' error in datasets >= 3.0.
    """
    import urllib.request
    import zipfile
    import io

    config_file_map = {
        "sentences_50agree": "Sentences_50Agree.txt",
        "sentences_66agree": "Sentences_66Agree.txt",
        "sentences_75agree": "Sentences_75Agree.txt",
        "sentences_allagree": "Sentences_AllAgree.txt",
    }
    target_file = config_file_map.get(config_name, "Sentences_50Agree.txt")

    # Local cache check
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(cache_dir, exist_ok=True)
    local_txt = os.path.join(cache_dir, target_file)

    if os.path.exists(local_txt):
        with open(local_txt, "r", encoding="latin-1") as f:
            content = f.read()
    else:
        url = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            zip_bytes = resp.read()

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            full_member_path = f"FinancialPhraseBank-v1.0/{target_file}"
            content = z.read(full_member_path).decode("latin-1")

        # Save to local cache
        try:
            with open(local_txt, "w", encoding="latin-1") as f:
                f.write(content)
        except Exception:
            pass

    sentences = []
    labels = []
    for line in content.splitlines():
        line = line.strip()
        if "@" in line:
            s, l = line.rsplit("@", 1)
            s, l = s.strip(), l.strip().lower()
            if l in LABEL2ID:
                sentences.append(s)
                labels.append(LABEL2ID[l])

    df = pd.DataFrame({"sentence": sentences, "label": labels})
    df["label_text"] = df["label"].map(ID2LABEL)
    return df


def load_raw_dataset(
    dataset_name: str = "takala/financial_phrasebank",
    config_name: str = "sentences_50agree",
) -> pd.DataFrame:
    """
    Load Financial PhraseBank dataset using Hugging Face datasets.
    Falls back seamlessly to direct Hugging Face raw archive download if `datasets` >= 3.0
    rejects dataset loading scripts.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, config_name, split="train")
        df = pd.DataFrame({
            "sentence": ds["sentence"],
            "label": ds["label"],
        })
        df["label_text"] = df["label"].map(ID2LABEL)
        return df
    except Exception as e:
        print(f"Loading via `datasets` failed ({e}). Using direct Hugging Face Hub download fallback...")
        try:
            return _download_and_parse_phrasebank(config_name=config_name)
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load dataset '{dataset_name}' with config '{config_name}'. "
                f"Both datasets library ({e}) and direct download ({e2}) failed. "
                "Please verify internet connection."
            )


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspect dataset for missing values, duplicates, and general stats.
    """
    missing_count = int(df["sentence"].isna().sum()) + int(df["label"].isna().sum())
    duplicate_count = int(df["sentence"].duplicated().sum())
    total_count = len(df)
    class_counts = df["label"].value_counts().to_dict()
    class_distribution = {ID2LABEL[k]: int(v) for k, v in class_counts.items()}

    # Calculate sentence length statistics on non-null sentences
    valid_sentences = df["sentence"].dropna().astype(str)
    if len(valid_sentences) > 0:
        word_lengths = valid_sentences.apply(lambda s: len(str(s).split()))
        char_lengths = valid_sentences.apply(lambda s: len(str(s)))
        word_stats = {
            "mean": float(word_lengths.mean()),
            "std": float(word_lengths.std()) if len(word_lengths) > 1 else 0.0,
            "min": int(word_lengths.min()),
            "median": float(word_lengths.median()),
            "max": int(word_lengths.max()),
        }
        char_stats = {
            "mean": float(char_lengths.mean()),
            "std": float(char_lengths.std()) if len(char_lengths) > 1 else 0.0,
            "min": int(char_lengths.min()),
            "median": float(char_lengths.median()),
            "max": int(char_lengths.max()),
        }
    else:
        word_stats = {"mean": 0.0, "std": 0.0, "min": 0, "median": 0.0, "max": 0}
        char_stats = {"mean": 0.0, "std": 0.0, "min": 0, "median": 0.0, "max": 0}

    stats = {
        "total_samples": total_count,
        "missing_values": missing_count,
        "duplicate_sentences": duplicate_count,
        "class_distribution": class_distribution,
        "word_length": word_stats,
        "char_length": char_stats,
    }
    return stats


def create_stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create a fixed, stratified 70/15/15 train/validation/test split.
    Uses random_seed=42 for strict reproducibility.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        # Fallback pure python/numpy stratification if scikit-learn is missing
        return _stratified_split_numpy(df, train_ratio, val_ratio, test_ratio, random_seed)

    # First split off the test set (15%)
    # Remaining is 85%
    test_size = test_ratio
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df["label"],
    )

    # From remaining 85%, validation is 15/85 of remaining
    val_relative_ratio = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_ratio,
        random_state=random_seed,
        stratify=train_val_df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def _stratified_split_numpy(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pure numpy stratified split fallback."""
    rng = np.random.RandomState(seed)
    train_indices, val_indices, test_indices = [], [], []

    for label, group in df.groupby("label"):
        n = len(group)
        indices = group.index.values.copy()
        rng.shuffle(indices)

        n_train = int(np.round(n * train_ratio))
        n_val = int(np.round(n * val_ratio))

        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train : n_train + n_val])
        test_indices.extend(indices[n_train + n_val :])

    return (
        df.loc[train_indices].reset_index(drop=True),
        df.loc[val_indices].reset_index(drop=True),
        df.loc[test_indices].reset_index(drop=True),
    )


def check_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, int]:
    """
    Check for sentence overlap between splits.
    """
    train_sentences = set(train_df["sentence"].str.strip().str.lower())
    val_sentences = set(val_df["sentence"].str.strip().str.lower())
    test_sentences = set(test_df["sentence"].str.strip().str.lower())

    train_val_overlap = len(train_sentences.intersection(val_sentences))
    train_test_overlap = len(train_sentences.intersection(test_sentences))
    val_test_overlap = len(val_sentences.intersection(test_sentences))

    return {
        "train_val_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
    }


def format_instruction_prompt(sentence: str, template: str = PROMPT_TEMPLATE) -> str:
    """
    Format a single input sentence into the standard instruction prompt.
    """
    return template.format(sentence=sentence.strip())


def format_chat_prompt(
    sentence: str,
    tokenizer: Any = None,
    template: str = PROMPT_TEMPLATE,
) -> str:
    """
    Format prompt using tokenizer chat template if available,
    otherwise fallback to raw instruction format.
    """
    raw_prompt = format_instruction_prompt(sentence, template)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [{"role": "user", "content": raw_prompt}]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return raw_prompt
    return raw_prompt


def prepare_hf_training_dataset(
    df: pd.DataFrame,
    tokenizer: Any,
    max_length: int = 256,
    template: str = PROMPT_TEMPLATE,
) -> Any:
    """
    Convert pandas DataFrame into HuggingFace Dataset ready for SFTTrainer / CausalLM fine-tuning.
    Formats training text as: <Prompt><Answer>\n
    """
    from datasets import Dataset

    texts = []
    for _, row in df.iterrows():
        prompt = format_instruction_prompt(row["sentence"], template)
        target = row["label_text"]
        # In chat template format
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ]
            try:
                formatted = tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception:
                formatted = f"{prompt} {target}"
        else:
            formatted = f"{prompt} {target}"
        texts.append(formatted)

    return Dataset.from_dict({"text": texts, "label": df["label"].tolist()})


def save_split_metadata(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: str = "results/metrics/split_metadata.json",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Save metadata about the train/val/test splits to ensure complete reproducibility.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def _split_stats(df: pd.DataFrame) -> Dict[str, Any]:
        counts = df["label"].value_counts().to_dict()
        dist = {ID2LABEL.get(k, str(k)): int(v) for k, v in counts.items()}
        pcts = {k: round(v / len(df) * 100, 2) for k, v in dist.items()}
        return {
            "total_samples": len(df),
            "class_counts": dist,
            "class_percentages": pcts,
        }

    leakage = check_leakage(train_df, val_df, test_df)

    metadata = {
        "dataset_name": "takala/financial_phrasebank",
        "dataset_config": "sentences_50agree",
        "random_seed": seed,
        "split_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "train": _split_stats(train_df),
        "validation": _split_stats(val_df),
        "test": _split_stats(test_df),
        "data_leakage_check": leakage,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata
