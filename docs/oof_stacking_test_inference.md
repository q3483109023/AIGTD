# Frozen OOF Stacking: replay + private-server unlabeled inference

Status: **private-server Novel replay, 16-sentence GPU dry-run and full
500-article/4,264-sentence CUDA test inference completed; generic output
validator PASS** (2026-11-22). Teacher's official submission schema not yet
verified. **No retraining or threshold tuning.**

## Model identity and data protocol

This is the **five-fold OOF Stacking** candidate whose Fold-matched Novel
development Macro-F1 was 0.9174, **not** `scripts/predict.py --model-type
fusion` (an unrelated embedding-plus-features pipeline). Novel has been
used for development; the score is **not** an independent test metric.

For each *unlabeled target dataset*:
1. Parse original JSON `[{id, domain, sent: [...]}, ...]`; preserve
   article record order and sentence index. `id` can repeat across records.
2. Group by source `id`. Assign whole groups to one fold using the exact
   deterministic largest-first, least-loaded *label-free* assignment used
   for Novel Validation in `scripts/train_oof_fusion.py`.
3. For each target sentence, score the **assigned fold's** SVM
   `decision_function` and RoBERTa `logits[1]-logits[0]`, using
   the fold's original checkpoint and tokenizer (`max_length=256`).
4. Standardize each raw margin using the **saved fold-specific OOF
   validation** mean/std from `results/oof_fusion/score_normalization.json`.
   Features are `[svm_z, roberta_z]` in this exact order.
5. Apply the matching fold's cross-fitted Logistic Regression from
   `fusion_models.joblib` and its **OOF-selected threshold 0.585**.
   Do not average five separate logits, probabilities or meta-model
   decisions. Do not use the unrelated single-model SVM threshold 0.52.

Fold assignment depends on the size of the whole *target* dataset.
A two-article dry-run may assign folds differently than the full 500-article
set. This is expected and neither assignment uses labels.

## Before neural inference: replay already-saved Novel predictions

On the server, from project root with its Conda environment active:

```bash
python scripts/predict_oof_stacking.py \
  --replay-novel results/oof_fusion/novel_fusion_predictions.csv \
  --fusion-dir results/oof_fusion
```

Required: `"status": "PASS"`, `rows=11682`,
`threshold_labels_match=true`, max absolute score differences within
5e-7. This proves only that the **stored meta-model + z-score normalization
+ threshold** exactly reproduce historical saved predictions.
It does NOT execute or validate RoBERTa/SVM checkpoint forward passes.

## First actual checkpoint dry-run: two complete articles, 16 sentences

The two-article local-only input already exists from the SVM dry-run.
These commands do **not** modify any pretrained weights or data splits:

```bash
python scripts/predict_oof_stacking.py \
  --test data/stage1_local_only/test_2_articles.json \
  --expected-articles 2 \
  --expected-sentences 16 \
  --device auto --batch-size 16 \
  --output-dir results/stage1_local_audit/stacking_dry_run

python scripts/validate_test_predictions.py \
  --raw-test data/stage1_local_only/test_2_articles.json \
  --pred-json results/stage1_local_audit/stacking_dry_run/test_predictions.json \
  --pred-csv results/stage1_local_audit/stacking_dry_run/test_predictions.csv \
  --expected-articles 2 \
  --expected-sentences 16 \
  --threshold 0.585
```

Both the model-level completion and output validator's `PASS` are required.
The SVM dry-run used 0.52; check that separately if desired, never mix
thresholds between inference methods.

## Full unlabeled test: only after replay and dry-run pass

```bash
python scripts/predict_oof_stacking.py \
  --test data/raw/AIGTD_test_without_labels.json \
  --expected-articles 500 --expected-sentences 4264 \
  --device auto --batch-size 32 \
  --output-dir results/stage1_local_audit/stacking_full

python scripts/validate_test_predictions.py \
  --raw-test data/raw/AIGTD_test_without_labels.json \
  --pred-json results/stage1_local_audit/stacking_full/test_predictions.json \
  --pred-csv results/stage1_local_audit/stacking_full/test_predictions.csv \
  --expected-articles 500 --expected-sentences 4264 \
  --threshold 0.585 \
  --summary-out results/stage1_local_audit/stacking_full_validation.json
```

Outputs stay under the ignored `results/stage1_local_audit/` directory.
The generic output schema is not verified as the teacher's formal
submission schema. No test labels exist, hence **no test Accuracy/F1**.
No test-set threshold selection is permitted.

## Acceptance and limitations

- Neural checkpoint availability is verified locally before inference.
- Output order and source-group consistency are checked; source `id`
  duplicates are not collapsed into one article record.
- Inference uses saved OOF validation statistics, not statistics refitted
  on unlabeled test. It selects one matching fold per source.
- The CI synthetic test mocks neural forward passes to exercise output
  alignment. The CI Novel replay uses saved base-model margins; neither
  establishes end-to-end GPU correctness. The on-server 16-sentence
  checkpoint run is an indispensable separate acceptance step.
- If the assignment or checkpoint protocol differs from the training
  stage, fail and investigate rather than silently replacing scores or
  retuning the threshold.
- Novel source IDs were audited as disjoint from OOF training in the
  original pipeline. The unlabeled test may have different overlaps;
  any academic independence claim needs a separate audit.

## Actual full-test acceptance record (2026-11-22)

The user executed the commands above on the private server and supplied
aggregate stdout. All **five** saved fold checkpoints contributed to the
full test inference:

| Fold | Sentences |
|---|---:|
| 0 | 853 |
| 1 | 853 |
| 2 | 853 |
| 3 | 853 |
| 4 | 852 |
| **Total** | **4,264** |

The output contains **500 article records**, **1,548 Human** predictions,
and **2,716 Machine** predictions. The full output validator returned
`"status": "PASS"` with
`json_order_and_text_match=true`,
`csv_consistent=true`,
`threshold_checked=true`, and
`source_article_ids_may_repeat=true`.
The run used CUDA, the saved five-fold meta-models and threshold
**0.5850000000000001**.

On the same server, development replay passed on 11,682 records
(maximum probability difference 2.22e-16); the earlier real 16-sentence
checkpoint dry-run and its output schema validator also passed.

This evidence is restricted to **inference execution and generic schema
correctness**. It is not an independent accuracy estimate and does not
establish that the generic JSON is an official submission artifact.
Outputs and ground-truth-free predictions remain private and ignored by
Git. The Stage-1 executable model is now frozen; any later E2
improvement must use its own separate development-only evaluation
protocol and must not silently overwrite this baseline.
