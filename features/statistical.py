"""Lightweight, dependency-free sentence statistics for AIGTD feature fusion."""
from __future__ import annotations

import re
import string
from typing import Iterable, List

import numpy as np
import pandas as pd

_WORD_RE = re.compile(r"\b[\w']+\b", flags=re.UNICODE)
_SENTENCE_END_RE = re.compile(r"[.!?]+")


def extract_one(text: str) -> dict:
    text = str(text or "")
    words = _WORD_RE.findall(text)
    n_words = len(words)
    n_chars = len(text)
    unique_words = len({w.lower() for w in words})
    alpha_chars = sum(ch.isalpha() for ch in text)
    upper_chars = sum(ch.isupper() for ch in text)
    digit_chars = sum(ch.isdigit() for ch in text)
    punct_chars = sum(ch in string.punctuation for ch in text)
    whitespace_chars = sum(ch.isspace() for ch in text)
    capitalized_tokens = sum(1 for w in words if len(w) > 1 and w[0].isupper())
    long_words = sum(1 for w in words if len(w) >= 7)
    short_words = sum(1 for w in words if len(w) <= 3)
    exclamations = text.count("!")
    questions = text.count("?")
    commas = text.count(",")
    semicolons = text.count(";")
    colons = text.count(":")
    periods = text.count(".")
    sentence_marks = len(_SENTENCE_END_RE.findall(text))
    return {
        "char_count": n_chars,
        "word_count": n_words,
        "unique_word_count": unique_words,
        "type_token_ratio": unique_words / max(n_words, 1),
        "mean_word_length": float(np.mean([len(w) for w in words])) if words else 0.0,
        "max_word_length": max((len(w) for w in words), default=0),
        "long_word_ratio": long_words / max(n_words, 1),
        "short_word_ratio": short_words / max(n_words, 1),
        "alpha_char_ratio": alpha_chars / max(n_chars, 1),
        "uppercase_char_ratio": upper_chars / max(n_chars, 1),
        "digit_char_ratio": digit_chars / max(n_chars, 1),
        "punctuation_ratio": punct_chars / max(n_chars, 1),
        "whitespace_ratio": whitespace_chars / max(n_chars, 1),
        "capitalized_token_ratio": capitalized_tokens / max(n_words, 1),
        "exclamation_count": exclamations,
        "question_count": questions,
        "comma_count": commas,
        "semicolon_count": semicolons,
        "colon_count": colons,
        "period_count": periods,
        "sentence_ending_count": sentence_marks,
        "exclamation_per_word": exclamations / max(n_words, 1),
        "question_per_word": questions / max(n_words, 1),
        "comma_per_word": commas / max(n_words, 1),
        "contains_digit": int(digit_chars > 0),
        "contains_quote": int(('"' in text) or ("'" in text)),
        "contains_parentheses": int(("(" in text) or (")" in text)),
    }


def extract_statistical_features(texts: Iterable[str]) -> pd.DataFrame:
    """Return one numeric feature row per input sentence, preserving input order."""
    return pd.DataFrame([extract_one(text) for text in texts]).replace([np.inf, -np.inf], 0).fillna(0)
