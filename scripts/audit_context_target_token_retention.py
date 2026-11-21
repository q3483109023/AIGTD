#!/usr/bin/env python3
"""Audit how much of each target sentence survives Context RoBERTa truncation.

Only aggregate CSV/JSON reports are written. Raw sentences, context strings,
article identifiers, and row identifiers remain in memory and are never
exported.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ContextSpans:
    target_text: str
    target_start: int
    target_end: int
    previous_chars: int
    next_chars: int


def _component_length(value: str, sentinel: str) -> int:
    value = value.strip()
    return 0 if value == sentinel else len(value)


def parse_context_spans(context_text: str) -> ContextSpans:
    """Return target offsets for the OOF [PREVIOUS]/[TARGET]/[NEXT] format."""
    value = str(context_text)
    previous_marker = "[PREVIOUS]"
    target_marker = "[TARGET]"
    next_marker = "[NEXT]"

    if previous_marker not in value:
        raise ValueError("context_text is missing [PREVIOUS].")
    before_target, separator, after_target = value.partition(target_marker)
    if not separator:
        raise ValueError("context_text is missing [TARGET].")
    if next_marker not in after_target:
        raise ValueError("context_text is missing [NEXT].")

    previous_component = before_target.split(previous_marker, 1)[1].strip()
    target_segment, _, next_component = after_target.rpartition(next_marker)
    target_text = target_segment.strip()
    if not target_text:
        raise ValueError("The target sentence parsed from context_text is empty.")

    leading = len(target_segment) - len(target_segment.lstrip())
    trailing = len(target_segment.rstrip())
    target_start = len(before_target) + len(target_marker) + leading
    target_end = len(before_target) + len(target_marker) + trailing
    if target_start >= target_end:
        raise ValueError("Could not determine a non-empty target character span.")

    return ContextSpans(
        target_text=target_text,
        target_start=target_start,
        target_end=target_end,
        previous_chars=_component_length(previous_component, "[START]"),
        next_chars=_component_length(next_component, "[END]"),
    )


def target_token_positions(
    offsets: Iterable[Iterable[int]], target_start: int, target_end: int
) -> list[int]:
    """Find non-special tokenizer offsets overlapping the target character span."""
    positions: list[int] = []
    for index, pair in enumerate(offsets):
        start, end = int(pair[0]), int(pair[1])
        if end > start and start < target_end and end > target_start:
            positions.append(index)
    return positions


def target_token_counts(
    full_offsets: Iterable[Iterable[int]],
    truncated_offsets: Iterable[Iterable[int]],
    target_start: int,
    target_end: int,
) -> tuple[int, int]:
    full_count = len(target_token_positions(full_offsets, target_start, target_end))
    retained_count = len(target_token_positions(truncated_offsets, target_start, target_end))
    if full_count == 0:
        raise ValueError("No tokenizer tokens overlap the target sentence.")
    if retained_count > full_count:
        raise ValueError("Truncated target-token count exceeds the untruncated count.")
    return full_count, retained_count


def target_length_bucket(char_length: int) -> str:
    if char_length <= 70:
        return "1-70"
    if char_length <= 98:
        return "71-98"
    if char_length <= 130:
        return "99-130"
    return ">130"


def context_length_bucket(char_length: int) -> str:
    if char_length == 0:
        return "0"
    if char_length <= 64:
        return "1-64"
    if char_length <= 128:
        return "65-128"
    if char_length <= 256:
        return "129-256"
    return ">256"


def summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    full_target = group["target_tokens_full"].to_numpy(dtype=float)
    kept_target = group["target_tokens_retained"].to_numpy(dtype=float)
    ratios = group["target_retention_ratio"].to_numpy(dtype=float)
    full_sequence = group["full_sequence_tokens"].to_numpy(dtype=float)
    truncated_sequence = group["truncated_sequence_tokens"].to_numpy(dtype=float)
    dropped = full_target - kept_target
    return {
        "n": int(len(group)),
        "sequence_truncation_rate": float((full_sequence > truncated_sequence).mean()),
        "target_truncation_rate": float((dropped > 0).mean()),
        "target_fully_retained_rate": float(np.isclose(ratios, 1.0).mean()),
        "target_fully_lost_rate": float((kept_target == 0).mean()),
        "mean_target_retention_ratio": float(np.mean(ratios)),
        "median_target_retention_ratio": float(np.median(ratios)),
        "p10_target_retention_ratio": float(np.quantile(ratios, 0.10)),
        "micro_target_retention_ratio": float(kept_target.sum() / full_target.sum()),
        "mean_target_tokens_full": float(full_target.mean()),
        "mean_target_tokens_retained": float(kept_target.mean()),
        "mean_target_tokens_dropped": float(dropped.mean()),
        "mean_full_sequence_tokens": float(full_sequence.mean()),
        "mean_truncated_sequence_tokens": float(truncated_sequence.mean()),
        "mean_previous_context_chars": float(group["previous_context_chars"].mean()),
        "mean_next_context_chars": float(group["next_context_chars"].mean()),
    }


def grouped_aggregates(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_columns, dropna=False, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, values))
        row.update(summarize_group(group))
        rows.append(row)
    return pd.DataFrame(rows)


def build_aggregate_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_tables = []
    all_labels = grouped_aggregates(frame, ["split", "fold"])
    all_labels["label_id"] = "all"
    all_labels["aggregation"] = "fold_all_labels"
    fold_tables.append(all_labels)
    by_label = grouped_aggregates(frame, ["split", "fold", "label_id"])
    by_label["aggregation"] = "fold_by_label"
    fold_tables.append(by_label)
    fold_summary = pd.concat(fold_tables, ignore_index=True, sort=False)

    stratified_rows = []
    for column in (
        "target_length_bucket",
        "previous_context_bucket",
        "next_context_bucket",
    ):
        grouped = grouped_aggregates(frame, ["split", "fold", "label_id", column])
        grouped = grouped.rename(columns={column: "stratum"})
        grouped["stratification"] = column
        stratified_rows.append(grouped)
    stratified_summary = pd.concat(stratified_rows, ignore_index=True, sort=False)
    return fold_summary, stratified_summary


def _measure_batch(
    tokenizer: Any,
    frame: pd.DataFrame,
    split: str,
    fold: str,
    max_length: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    required = {"text", "context_text", "label_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{split} fold={fold} is missing columns: {sorted(missing)}")

    measurements: list[dict[str, Any]] = []
    for start in range(0, len(frame), batch_size):
        chunk = frame.iloc[start : start + batch_size]
        context_texts = chunk["context_text"].astype(str).tolist()
        full = tokenizer(
            context_texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_offsets_mapping=True,
        )
        truncated = tokenizer(
            context_texts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_offsets_mapping=True,
        )

        for local_index, (_, row) in enumerate(chunk.iterrows()):
            context = parse_context_spans(context_texts[local_index])
            expected_target = str(row["text"]).strip()
            if context.target_text != expected_target:
                raise ValueError(
                    f"Target/context mismatch in {split} fold={fold}; "
                    "context_text does not contain the raw target sentence."
                )

            full_offsets = full["offset_mapping"][local_index]
            truncated_offsets = truncated["offset_mapping"][local_index]
            full_positions = target_token_positions(
                full_offsets, context.target_start, context.target_end
            )
            full_target_count, retained_target_count = target_token_counts(
                full_offsets,
                truncated_offsets,
                context.target_start,
                context.target_end,
            )
            full_sequence_tokens = len(full["input_ids"][local_index])
            truncated_sequence_tokens = len(truncated["input_ids"][local_index])
            target_char_length = len(context.target_text)

            label_id = int(row["label_id"])
            if label_id not in (0, 1):
                raise ValueError(f"Unexpected label_id={label_id} in {split} fold={fold}.")

            measurements.append(
                {
                    "split": split,
                    "fold": str(fold),
                    "label_id": label_id,
                    "target_char_length": target_char_length,
                    "target_length_bucket": target_length_bucket(target_char_length),
                    "previous_context_chars": context.previous_chars,
                    "previous_context_bucket": context_length_bucket(context.previous_chars),
                    "next_context_chars": context.next_chars,
                    "next_context_bucket": context_length_bucket(context.next_chars),
                    "full_sequence_tokens": full_sequence_tokens,
                    "truncated_sequence_tokens": truncated_sequence_tokens,
                    "target_tokens_full": full_target_count,
                    "target_tokens_retained": retained_target_count,
                    "target_tokens_dropped": full_target_count - retained_target_count,
                    "target_retention_ratio": retained_target_count / full_target_count,
                    "target_first_token_position": min(full_positions),
                    "target_last_token_position": max(full_positions),
                }
            )
    return measurements


def _load_config(config_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read the experiment config.") from exc

    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[1] / config_path
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--folds-dir", default="data/context_oof_folds")
    parser.add_argument("--model-dir", default="models/oof/context")
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Saved Context-fold tokenizer path; defaults to context_fold_0 when present, else transformer.model_name.",
    )
    parser.add_argument("--output-dir", default="results/context_target_token_retention")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if args.n_splits < 2 or args.batch_size < 1:
        raise ValueError("--n-splits must be >= 2 and --batch-size must be positive.")

    config = _load_config(Path(args.config))
    transformer_config = config.get("transformer", {})
    model_name = str(transformer_config.get("model_name", "roberta-base"))
    max_length = int(args.max_length or transformer_config.get("max_length", 256))
    if max_length < 2:
        raise ValueError("--max-length must be >= 2.")

    model_dir = Path(args.model_dir)
    local_fold_zero = model_dir / "context_fold_0"
    tokenizer_source = args.tokenizer_path or (
        str(local_fold_zero) if local_fold_zero.exists() else model_name
    )

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Transformers is required for tokenizer-based truncation auditing.") from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "A fast tokenizer with offset_mapping is required; refusing to estimate target spans without offsets."
        )

    folds_dir = Path(args.folds_dir)
    all_measurements: list[dict[str, Any]] = []
    oof_row_ids: list[int] = []
    oof_rows = 0

    for fold in range(args.n_splits):
        fold_dir = folds_dir / f"fold_{fold}"
        validation_path = fold_dir / "validation.csv"
        if not validation_path.exists():
            raise FileNotFoundError(validation_path)

        metadata_path = model_dir / f"context_fold_{fold}" / "oof_metadata.json"
        if metadata_path.exists():
            checkpoint_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            checkpoint_max_length = checkpoint_metadata.get("max_length")
            if checkpoint_max_length is not None and int(checkpoint_max_length) != max_length:
                raise ValueError(
                    f"Fold {fold} checkpoint max_length={checkpoint_max_length} "
                    f"does not match audit max_length={max_length}."
                )
            input_column = checkpoint_metadata.get("input_column")
            if input_column not in (None, "context_text"):
                raise ValueError(
                    f"Fold {fold} checkpoint input_column={input_column!r}, not context_text."
                )

        validation = pd.read_csv(validation_path, encoding="utf-8-sig")
        if "oof_row_id" not in validation:
            raise ValueError(f"{validation_path} is missing oof_row_id.")
        oof_row_ids.extend(validation["oof_row_id"].astype(int).tolist())
        oof_rows += len(validation)
        all_measurements.extend(
            _measure_batch(
                tokenizer,
                validation,
                split="oof_validation",
                fold=str(fold),
                max_length=max_length,
                batch_size=args.batch_size,
            )
        )

    if len(oof_row_ids) != len(set(oof_row_ids)):
        raise ValueError("OOF validation folds contain duplicated oof_row_id values.")

    novel_path = folds_dir / "novel_validation.csv"
    if not novel_path.exists():
        raise FileNotFoundError(novel_path)
    novel = pd.read_csv(novel_path, encoding="utf-8-sig")
    if "novel_row_id" not in novel:
        raise ValueError(f"{novel_path} is missing novel_row_id.")
    if novel["novel_row_id"].duplicated().any():
        raise ValueError("novel_validation.csv contains duplicated novel_row_id values.")
    all_measurements.extend(
        _measure_batch(
            tokenizer,
            novel,
            split="novel_validation",
            fold="shared",
            max_length=max_length,
            batch_size=args.batch_size,
        )
    )

    measured = pd.DataFrame(all_measurements)
    if measured.empty:
        raise RuntimeError("No Context windows were measured.")

    fold_summary, stratified_summary = build_aggregate_tables(measured)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_summary.to_csv(output_dir / "fold_summary.csv", index=False, encoding="utf-8-sig")
    stratified_summary.to_csv(
        output_dir / "stratified_summary.csv", index=False, encoding="utf-8-sig"
    )

    split_counts = {
        str(name): int(count)
        for name, count in measured.groupby("split", sort=True).size().items()
    }
    metadata = {
        "audit": "context_target_token_retention",
        "tokenizer_source": str(tokenizer_source),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": bool(tokenizer.is_fast),
        "truncation_side": str(tokenizer.truncation_side),
        "max_length": max_length,
        "add_special_tokens": True,
        "oof_validation_rows": int(oof_rows),
        "unique_oof_row_ids": int(len(set(oof_row_ids))),
        "novel_validation_rows": int(len(novel)),
        "split_row_counts": split_counts,
        "target_span_method": "parse [TARGET] ... [NEXT] boundaries and fast-tokenizer offset_mapping",
        "target_length_buckets_characters": ["1-70", "71-98", "99-130", ">130"],
        "context_length_buckets_characters": ["0", "1-64", "65-128", "129-256", ">256"],
        "privacy": {
            "row_level_exports_written": False,
            "sentence_or_context_text_written": False,
            "article_or_row_identifiers_written": False,
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Context target-token retention audit completed.")
    print(
        f"Tokenizer={type(tokenizer).__name__}; truncation_side={tokenizer.truncation_side}; "
        f"max_length={max_length}"
    )
    print(f"Measured rows: {len(measured):,}; splits={split_counts}")
    print(f"Aggregate outputs: {output_dir.resolve()}")
    print("No row-level predictions, text, or identifiers were exported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
