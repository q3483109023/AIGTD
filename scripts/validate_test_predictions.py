#!/usr/bin/env python3
"""Validate unlabeled AIGTD predictions without disclosing sentences or article IDs.

The JSON is the current generic inference schema, NOT an assumed official
competition submission schema. Source IDs are not required to be unique.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


LABELS = {"human": 0, "machine": 1}


def read_articles(path: Path, name: str) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError(f"{name}: expected a list of article objects")
    return records


def validate(raw_path: Path, json_path: Path, csv_path: Path | None,
             expected_articles: int, expected_sentences: int,
             threshold: float | None = None) -> dict:
    if threshold is not None and not (0 <= threshold <= 1):
        raise ValueError("threshold must be in [0,1]")
    raw = read_articles(raw_path, "raw test")
    preds = read_articles(json_path, "prediction JSON")
    if len(raw) != expected_articles or len(preds) != len(raw):
        raise ValueError(f"article record count mismatch: expected {expected_articles}")
    expected = []
    label_counts = {"human": 0, "machine": 0}
    for article_index, (source, predicted) in enumerate(zip(raw, preds)):
        if "id" not in source or "id" not in predicted:
            raise ValueError(f"missing article id at position {article_index}")
        if str(source["id"]) != str(predicted["id"]):
            raise ValueError(f"article order/id mismatch at position {article_index}")
        sentences = source.get("sent")
        stored_sentences = predicted.get("sent")
        labels = predicted.get("predicted_labels")
        if not isinstance(sentences, list) or not isinstance(stored_sentences, list) or not isinstance(labels, list):
            raise ValueError(f"missing sentence/label lists at article position {article_index}")
        if len(sentences) != len(stored_sentences) or len(labels) != len(sentences):
            raise ValueError(f"sentence/label length mismatch at article position {article_index}")
        for sentence_index, (sentence, stored, label) in enumerate(zip(sentences, stored_sentences, labels)):
            if not isinstance(sentence, str) or not sentence.strip():
                raise ValueError("raw test contains a non-string or blank sentence")
            if stored != sentence.strip():
                raise ValueError(f"sentence order/content mismatch at position {article_index}/{sentence_index}")
            if label not in LABELS:
                raise ValueError(f"invalid label at position {article_index}/{sentence_index}")
            label_counts[label] += 1
            expected.append((f"record_{article_index:08d}", sentence_index, sentence.strip(), label))
    if len(expected) != expected_sentences:
        raise ValueError(f"sentence count mismatch: expected {expected_sentences}, got {len(expected)}")

    if csv_path is not None:
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"article_key", "sentence_index", "text", "predicted_label",
                        "predicted_id", "p_machine"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"prediction CSV missing columns: {sorted(required - set(reader.fieldnames or []))}")
            count = 0
            for row in reader:
                if count >= len(expected):
                    raise ValueError("prediction CSV has extra rows")
                key, index, text, label = expected[count]
                if row["article_key"] != key:
                    raise ValueError(f"CSV article key/order mismatch at row {count}")
                try:
                    row_index = int(row["sentence_index"])
                    pred_id = int(row["predicted_id"])
                    p_machine = float(row["p_machine"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"non-numeric CSV prediction at row {count}") from exc
                if row_index != index or row["text"] != text:
                    raise ValueError(f"CSV sentence order/content mismatch at row {count}")
                if row["predicted_label"] != label or pred_id != LABELS[label]:
                    raise ValueError(f"CSV label disagrees with prediction JSON at row {count}")
                if not math.isfinite(p_machine) or not 0 <= p_machine <= 1:
                    raise ValueError(f"invalid machine score at row {count}")
                if threshold is not None and (p_machine >= threshold) != (label == "machine"):
                    raise ValueError(f"prediction label/threshold mismatch at row {count}")
                count += 1
            if count != len(expected):
                raise ValueError(f"prediction CSV incomplete: {count}/{len(expected)}")
    return {
        "status": "PASS",
        "schema": "generic_aigtd_predictions_not_official_submission",
        "articles": len(raw),
        "sentences": len(expected),
        "labels": label_counts,
        "json_order_and_text_match": True,
        "csv_consistent": csv_path is not None,
        "threshold_checked": threshold is not None and csv_path is not None,
        "source_article_ids_may_repeat": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-test", type=Path, default=Path("data/raw/AIGTD_test_without_labels.json"))
    parser.add_argument("--pred-json", type=Path, default=Path("results/test_predictions.json"))
    parser.add_argument("--pred-csv", type=Path, default=Path("results/test_predictions.csv"))
    parser.add_argument("--expected-articles", type=int, default=500)
    parser.add_argument("--expected-sentences", type=int, default=4264)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Optional frozen classification threshold; do not tune on test.")
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()
    try:
        result = validate(args.raw_test, args.pred_json, args.pred_csv,
                          args.expected_articles, args.expected_sentences, args.threshold)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
