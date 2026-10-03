"""
Unit Tests for Classification Metrics.
Verifies calculation of Accuracy, Precision, Recall, Macro-F1, Weighted-F1,
and Confusion Matrix against hand-calculated ground truth.
"""

import sys
import os
import pytest

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_invalid_rate,
)


def test_perfect_classification():
    """Verify metrics when all predictions are exactly correct."""
    y_true = ["positive", "negative", "neutral", "positive", "neutral", "negative"]
    y_pred = ["positive", "negative", "neutral", "positive", "neutral", "negative"]

    metrics = compute_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["invalid_output_rate"] == 0.0
    assert metrics["invalid_count"] == 0


def test_known_imperfect_classification():
    """
    Verify metrics against hand-calculated values.
    Classes:
      positive: true=2, tp=1, fp=1, fn=1 -> P=1/2=0.5, R=1/2=0.5, F1=0.5
      negative: true=2, tp=2, fp=0, fn=0 -> P=1.0, R=1.0, F1=1.0
      neutral:  true=2, tp=1, fp=1, fn=1 -> P=1/2=0.5, R=1/2=0.5, F1=0.5

    Macro-F1 = (0.5 + 1.0 + 0.5) / 3 = 2.0 / 3 = 0.6667
    Accuracy = 4 / 6 = 0.6667
    """
    y_true = ["positive", "positive", "negative", "negative", "neutral", "neutral"]
    y_pred = ["positive", "neutral",  "negative", "negative", "positive", "neutral"]

    metrics = compute_classification_metrics(y_true, y_pred)

    assert pytest.approx(metrics["accuracy"], 0.001) == 0.6667
    assert pytest.approx(metrics["macro_f1"], 0.001) == 0.6667
    assert metrics["per_class"]["negative"]["f1"] == 1.0
    assert pytest.approx(metrics["per_class"]["positive"]["f1"], 0.001) == 0.5
    assert pytest.approx(metrics["per_class"]["neutral"]["f1"], 0.001) == 0.5


def test_invalid_prediction_handling():
    """
    Verify that unparseable/invalid predictions are counted as incorrect
    and accurately recorded in invalid_output_rate.
    """
    y_true = ["positive", "negative", "neutral", "positive"]
    y_pred = ["positive", "invalid",  "neutral", "invalid"]

    metrics = compute_classification_metrics(y_true, y_pred)

    assert metrics["invalid_count"] == 2
    assert metrics["invalid_output_rate"] == 50.0  # 2 out of 4
    assert metrics["accuracy"] == 0.5  # 2 correct out of 4


def test_confusion_matrix_dimensions():
    """Verify confusion matrix counts match prediction pairs."""
    y_true = ["positive", "negative", "neutral"]
    y_pred = ["neutral",  "negative", "positive"]

    cm = compute_confusion_matrix(y_true, y_pred)
    counts = cm["counts"]

    assert counts["negative"]["negative"] == 1
    assert counts["positive"]["neutral"] == 1
    assert counts["neutral"]["positive"] == 1
    assert counts["negative"]["positive"] == 0
