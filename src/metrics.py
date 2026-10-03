"""
Metrics and Output Parsing Module for Financial Sentiment Classification.

Handles:
- Robust parsing of raw LLM outputs to extract 'positive', 'negative', 'neutral'
- Tracking unrecognized/invalid predictions
- Computing standard classification metrics:
  - Accuracy
  - Macro Precision, Recall, F1
  - Weighted F1
  - Per-class Precision, Recall, F1
  - Confusion Matrix
  - Invalid Output Rate
"""

import re
import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


VALID_LABELS = ["negative", "neutral", "positive"]
LABEL_SYNONYMS = {
    "positive": ["positive", "bullish", "optimistic", "favorable", "pos"],
    "negative": ["negative", "bearish", "pessimistic", "unfavorable", "neg"],
    "neutral": ["neutral", "flat", "unchanged", "mixed", "neu"],
}


def parse_model_output(raw_text: str) -> str:
    """
    Robustly extract sentiment label ('positive', 'negative', 'neutral')
    from raw LLM generation. Returns 'invalid' if unrecognized.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return "invalid"

    # Normalize text: lowercase, remove markdown formatting (*, _, `, #)
    text = raw_text.strip().lower()
    text = re.sub(r"[\*\_`\#]", "", text)

    # Check for direct exact match first
    if text in VALID_LABELS:
        return text

    # Handle prefixed outputs like "Answer: positive" or "Sentiment: negative"
    prefix_patterns = [
        r"^(?:answer|sentiment|classification|label)\s*[:\-]?\s*(positive|negative|neutral)",
        r"\b(?:the sentiment is|the answer is|this is)\s*(positive|negative|neutral)\b",
    ]
    for pattern in prefix_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    # Check words at the start of output (LLM answers first word)
    first_token = re.split(r"[^a-zA-Z]", text)[0] if text else ""
    if first_token in VALID_LABELS:
        return first_token

    # Check for presence of distinct labels
    found_labels = [label for label in VALID_LABELS if re.search(rf"\b{label}\b", text)]
    if len(found_labels) == 1:
        return found_labels[0]

    # Check synonyms if still not identified
    for canonical_label, syns in LABEL_SYNONYMS.items():
        for syn in syns:
            if re.search(rf"\b{syn}\b", text):
                return canonical_label

    return "invalid"


def compute_invalid_rate(predictions: List[str]) -> float:
    """
    Compute percentage of model outputs that could not be parsed into valid classes.
    """
    if not predictions:
        return 0.0
    invalid_count = sum(1 for p in predictions if p == "invalid")
    return round((invalid_count / len(predictions)) * 100, 2)


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    If 'invalid' is present in y_pred, it is counted as an error.
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must be of equal length"
    n_samples = len(y_true)

    # Calculate invalid rate
    invalid_count = sum(1 for p in y_pred if p == "invalid")
    invalid_rate = round((invalid_count / n_samples) * 100, 2) if n_samples > 0 else 0.0

    # Overall Accuracy
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = round(correct / n_samples, 4) if n_samples > 0 else 0.0

    # Per-class metrics
    per_class = {}
    macro_p_sum, macro_r_sum, macro_f1_sum = 0.0, 0.0, 0.0
    weighted_f1_sum = 0.0

    for label in VALID_LABELS:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        support = sum(1 for yt in y_true if yt == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

        macro_p_sum += precision
        macro_r_sum += recall
        macro_f1_sum += f1
        weighted_f1_sum += f1 * support

    num_classes = len(VALID_LABELS)
    macro_precision = round(macro_p_sum / num_classes, 4)
    macro_recall = round(macro_r_sum / num_classes, 4)
    macro_f1 = round(macro_f1_sum / num_classes, 4)
    weighted_f1 = round(weighted_f1_sum / n_samples, 4) if n_samples > 0 else 0.0

    cm = compute_confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "invalid_output_rate": invalid_rate,
        "invalid_count": invalid_count,
        "total_samples": n_samples,
        "per_class": per_class,
        "confusion_matrix": cm,
    }
    return metrics


def compute_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute confusion matrix dictionary including any invalid outputs.
    """
    if labels is None:
        labels = VALID_LABELS

    matrix = {true_lbl: {pred_lbl: 0 for pred_lbl in labels + ["invalid"]} for true_lbl in labels}

    for yt, yp in zip(y_true, y_pred):
        if yt in matrix:
            target_pred = yp if yp in (labels + ["invalid"]) else "invalid"
            matrix[yt][target_pred] += 1

    # Also build 2D list for easy plotting (excluding invalid column or keeping as requested)
    matrix_2d = [[matrix[t][p] for p in labels] for t in labels]

    return {
        "labels": labels,
        "counts": matrix,
        "matrix_2d": matrix_2d,
    }


def format_metrics_summary(metrics: Dict[str, Any], method_name: str = "Model") -> str:
    """
    Create a clean, human-readable terminal/notebook table of metrics.
    """
    lines = [
        "=" * 55,
        f"        CLASSIFICATION PERFORMANCE: {method_name.upper()}",
        "=" * 55,
        f"Accuracy:            {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)",
        f"Macro F1:            {metrics['macro_f1']:.4f}",
        f"Macro Precision:     {metrics['macro_precision']:.4f}",
        f"Macro Recall:        {metrics['macro_recall']:.4f}",
        f"Weighted F1:         {metrics['weighted_f1']:.4f}",
        f"Invalid Output Rate: {metrics['invalid_output_rate']:.2f}% ({metrics['invalid_count']}/{metrics['total_samples']})",
        "-" * 55,
        "Per-Class Performance:",
        f"  Negative  ->  P: {metrics['per_class']['negative']['precision']:.4f} | R: {metrics['per_class']['negative']['recall']:.4f} | F1: {metrics['per_class']['negative']['f1']:.4f} (n={metrics['per_class']['negative']['support']})",
        f"  Neutral   ->  P: {metrics['per_class']['neutral']['precision']:.4f} | R: {metrics['per_class']['neutral']['recall']:.4f} | F1: {metrics['per_class']['neutral']['f1']:.4f} (n={metrics['per_class']['neutral']['support']})",
        f"  Positive  ->  P: {metrics['per_class']['positive']['precision']:.4f} | R: {metrics['per_class']['positive']['recall']:.4f} | F1: {metrics['per_class']['positive']['f1']:.4f} (n={metrics['per_class']['positive']['support']})",
        "=" * 55,
    ]
    return "\n".join(lines)
