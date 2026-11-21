import unittest

import pandas as pd

from scripts.audit_context_target_token_retention import (
    build_aggregate_tables,
    context_length_bucket,
    parse_context_spans,
    summarize_group,
    target_length_bucket,
    target_token_counts,
    target_token_positions,
)


class ContextSpanTests(unittest.TestCase):
    def test_parse_target_span_and_context_lengths(self):
        context = (
            "[PREVIOUS] previous words [TARGET] Target sentence ends here. "
            "[NEXT] following sentence"
        )
        parsed = parse_context_spans(context)

        self.assertEqual(parsed.target_text, "Target sentence ends here.")
        self.assertEqual(
            context[parsed.target_start : parsed.target_end],
            parsed.target_text,
        )
        self.assertEqual(parsed.previous_chars, len("previous words"))
        self.assertEqual(parsed.next_chars, len("following sentence"))

    def test_start_end_sentinels_have_zero_context_length(self):
        parsed = parse_context_spans(
            "[PREVIOUS] [START] [TARGET] Only sentence. [NEXT] [END]"
        )
        self.assertEqual(parsed.previous_chars, 0)
        self.assertEqual(parsed.next_chars, 0)

    def test_missing_markers_fail_closed(self):
        with self.assertRaises(ValueError):
            parse_context_spans("No target boundary marker")


class TokenRetentionTests(unittest.TestCase):
    def test_special_tokens_are_not_counted_and_partial_target_is_detected(self):
        context = "[PREVIOUS] before [TARGET] Target words [NEXT] after"
        parsed = parse_context_spans(context)
        full_offsets = [
            (0, 0), (0, 10), (11, 17), (18, 26),
            (27, 33), (34, 39), (40, 46), (47, 52), (0, 0),
        ]
        truncated_offsets = [
            (0, 0), (0, 10), (11, 17), (18, 26), (27, 33), (0, 0),
        ]

        positions = target_token_positions(
            full_offsets, parsed.target_start, parsed.target_end
        )
        full_count, retained_count = target_token_counts(
            full_offsets,
            truncated_offsets,
            parsed.target_start,
            parsed.target_end,
        )

        self.assertEqual(len(positions), 2)
        self.assertEqual(full_count, 2)
        self.assertEqual(retained_count, 1)

    def test_length_buckets_match_fold4_diagnostic_boundaries(self):
        self.assertEqual(target_length_bucket(70), "1-70")
        self.assertEqual(target_length_bucket(71), "71-98")
        self.assertEqual(target_length_bucket(98), "71-98")
        self.assertEqual(target_length_bucket(99), "99-130")
        self.assertEqual(target_length_bucket(130), "99-130")
        self.assertEqual(target_length_bucket(131), ">130")

        self.assertEqual(context_length_bucket(0), "0")
        self.assertEqual(context_length_bucket(64), "1-64")
        self.assertEqual(context_length_bucket(65), "65-128")
        self.assertEqual(context_length_bucket(129), "129-256")
        self.assertEqual(context_length_bucket(257), ">256")

    def test_aggregate_reports_contain_no_row_identifiers_or_text(self):
        frame = pd.DataFrame(
            [
                {
                    "split": "oof_validation", "fold": "4", "label_id": 0,
                    "target_length_bucket": ">130", "previous_context_bucket": "65-128",
                    "next_context_bucket": "129-256", "full_sequence_tokens": 310,
                    "truncated_sequence_tokens": 256, "target_tokens_full": 80,
                    "target_tokens_retained": 54, "target_retention_ratio": 0.675,
                    "previous_context_chars": 100, "next_context_chars": 200,
                    "text": "PRIVATE SENTENCE", "context_text": "PRIVATE CONTEXT",
                    "article_id": 123, "article_key": "private-key", "oof_row_id": 10,
                },
                {
                    "split": "oof_validation", "fold": "4", "label_id": 0,
                    "target_length_bucket": ">130", "previous_context_bucket": "65-128",
                    "next_context_bucket": "129-256", "full_sequence_tokens": 290,
                    "truncated_sequence_tokens": 256, "target_tokens_full": 60,
                    "target_tokens_retained": 60, "target_retention_ratio": 1.0,
                    "previous_context_chars": 90, "next_context_chars": 180,
                    "text": "ANOTHER PRIVATE SENTENCE", "context_text": "ANOTHER PRIVATE CONTEXT",
                    "article_id": 124, "article_key": "another-private-key", "oof_row_id": 11,
                },
            ]
        )

        summary = summarize_group(frame)
        fold_summary, stratified_summary = build_aggregate_tables(frame)

        self.assertEqual(summary["n"], 2)
        self.assertAlmostEqual(summary["target_truncation_rate"], 0.5)
        self.assertAlmostEqual(summary["mean_target_retention_ratio"], 0.8375)
        forbidden = {
            "text", "context_text", "article_id", "article_key",
            "oof_row_id", "novel_row_id",
        }
        self.assertTrue(forbidden.isdisjoint(fold_summary.columns))
        self.assertTrue(forbidden.isdisjoint(stratified_summary.columns))
        self.assertEqual(len(fold_summary), 2)
        self.assertEqual(len(stratified_summary), 3)


if __name__ == "__main__":
    unittest.main()
