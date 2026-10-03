"""
Unit Tests for LLM Output Label Parsing.
Verifies robust extraction of 'positive', 'negative', 'neutral' from varied model generations,
markdown formatting, prefixes, and unrecognized text.
"""

import sys
import os
import pytest

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics import parse_model_output, compute_invalid_rate


def test_exact_matches():
    """Verify standard lowercase exact labels."""
    assert parse_model_output("positive") == "positive"
    assert parse_model_output("negative") == "negative"
    assert parse_model_output("neutral") == "neutral"


def test_case_and_whitespace_insensitivity():
    """Verify handling of uppercase, mixed case, and trailing whitespace."""
    assert parse_model_output("  POSITIVE  ") == "positive"
    assert parse_model_output("Negative\n") == "negative"
    assert parse_model_output("NeuTral\r\n") == "neutral"


def test_markdown_formatting():
    """Verify handling of markdown asterisks, backticks, and bold text."""
    assert parse_model_output("**positive**") == "positive"
    assert parse_model_output("*negative*") == "negative"
    assert parse_model_output("`neutral`") == "neutral"
    assert parse_model_output("### Positive") == "positive"


def test_prefixed_responses():
    """Verify handling of conversational prefixes from chat models."""
    assert parse_model_output("Answer: positive") == "positive"
    assert parse_model_output("Answer: negative.") == "negative"
    assert parse_model_output("Sentiment: neutral") == "neutral"
    assert parse_model_output("The sentiment is positive.") == "positive"
    assert parse_model_output("Classification: negative") == "negative"


def test_financial_synonyms():
    """Verify handling of common financial synonym terms."""
    assert parse_model_output("bullish") == "positive"
    assert parse_model_output("bearish") == "negative"
    assert parse_model_output("unchanged") == "neutral"


def test_invalid_outputs():
    """Verify that unparseable or irrelevant text returns 'invalid'."""
    assert parse_model_output("") == "invalid"
    assert parse_model_output("   ") == "invalid"
    assert parse_model_output("I cannot determine the sentiment from this statement.") == "invalid"
    assert parse_model_output("123456") == "invalid"
    assert parse_model_output("The company was founded in 1998.") == "invalid"


def test_compute_invalid_rate():
    """Verify invalid rate percentage calculation."""
    preds = ["positive", "negative", "invalid", "neutral"]
    assert compute_invalid_rate(preds) == 25.0

    preds_all_valid = ["positive", "negative", "neutral"]
    assert compute_invalid_rate(preds_all_valid) == 0.0

    assert compute_invalid_rate([]) == 0.0
