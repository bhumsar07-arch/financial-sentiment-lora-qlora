"""
Unit Tests for Data Utilities.
Verifies label mapping, dataset splitting, prompt construction, and data quality checks
without requiring external network downloads or large models.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_utils import (
    ID2LABEL,
    LABEL2ID,
    create_stratified_split,
    format_instruction_prompt,
    check_data_quality,
    check_leakage,
    PROMPT_TEMPLATE,
)


def test_label_mapping_integrity():
    """Verify bidirectional mapping between IDs and string sentiment labels."""
    assert len(ID2LABEL) == 3
    assert len(LABEL2ID) == 3
    for id_val, label_str in ID2LABEL.items():
        assert LABEL2ID[label_str] == id_val
    assert ID2LABEL[0] == "negative"
    assert ID2LABEL[1] == "neutral"
    assert ID2LABEL[2] == "positive"


def test_format_instruction_prompt():
    """Verify that instruction prompt correctly embeds statement and choices."""
    sentence = "Operating profit increased by 14% to EUR 12.3 million."
    prompt = format_instruction_prompt(sentence)

    assert "You are a financial sentiment classifier." in prompt
    assert "positive" in prompt
    assert "negative" in prompt
    assert "neutral" in prompt
    assert sentence in prompt
    assert prompt.endswith("Answer:")


def test_stratified_split_ratios_and_reproducibility():
    """Verify exact 70/15/15 stratified split and seed reproducibility."""
    # Synthetic dataset with imbalanced classes resembling Financial PhraseBank
    np.random.seed(42)
    labels = [1] * 60 + [2] * 25 + [0] * 15  # 100 samples
    sentences = [f"Sample financial sentence number {i}" for i in range(100)]
    df = pd.DataFrame({"sentence": sentences, "label": labels})
    df["label_text"] = df["label"].map(ID2LABEL)

    train_df1, val_df1, test_df1 = create_stratified_split(df, 0.70, 0.15, 0.15, random_seed=42)
    train_df2, val_df2, test_df2 = create_stratified_split(df, 0.70, 0.15, 0.15, random_seed=42)

    # Check split sizes
    assert len(train_df1) == 70
    assert len(val_df1) == 15
    assert len(test_df1) == 15

    # Check exact reproducibility with same seed
    pd.testing.assert_frame_equal(train_df1, train_df2)
    pd.testing.assert_frame_equal(val_df1, val_df2)
    pd.testing.assert_frame_equal(test_df1, test_df2)

    # Verify no index overlap (disjoint splits)
    train_s = set(train_df1["sentence"])
    val_s = set(val_df1["sentence"])
    test_s = set(test_df1["sentence"])
    assert len(train_s.intersection(val_s)) == 0
    assert len(train_s.intersection(test_s)) == 0
    assert len(val_s.intersection(test_s)) == 0


def test_check_data_quality():
    """Verify detection of missing values and duplicate rows."""
    df = pd.DataFrame({
        "sentence": ["Revenue rose.", "Revenue rose.", "Profit dropped.", None],
        "label": [2, 2, 0, 1],
    })
    stats = check_data_quality(df)

    assert stats["total_samples"] == 4
    assert stats["missing_values"] == 1
    assert stats["duplicate_sentences"] == 1


def test_check_leakage():
    """Verify leakage detection identifies overlapping sentences across splits."""
    train_df = pd.DataFrame({"sentence": ["Apple sales grew.", "Oil price fell."]})
    val_df = pd.DataFrame({"sentence": ["Tesla opened new plant.", "Oil price fell."]})
    test_df = pd.DataFrame({"sentence": ["Market remains steady."]})

    leakage = check_leakage(train_df, val_df, test_df)
    assert leakage["train_val_overlap"] == 1
    assert leakage["train_test_overlap"] == 0
    assert leakage["val_test_overlap"] == 0
