# AIGTD Purged Group OOF - Stage 1

This package turns the audited OOF fold manifest into actual fold datasets and generates fold-wise OOF scores for sentence-only SVM and RoBERTa.

## Accepted split strategy

- All article records sharing the original `source_article_id` (`article_id` in the current train CSV) stay in one fold.
- For each fold, the candidate training rows are all rows outside that fold's validation rows.
- From that fold's candidate training rows, remove every row whose normalized sentence occurs in that fold's validation rows.
- Normalization is Unicode NFKC + `casefold()` + collapsed whitespace; punctuation and word order are retained.
- This is intended for sentence-only models. Do not reuse this purge for context-window training without a separate context-window leakage audit.

## Step 1: Materialize fold datasets

The audit script must already have produced `results/oof_purged_group_audit/row_fold_manifest.csv`.

```bash
python scripts/prepare_oof_folds.py \
  --train data/splits/train.csv \
  --validation data/splits/validation.csv \
  --row-manifest results/oof_purged_group_audit/row_fold_manifest.csv \
  --output-dir data/oof_folds \
  --n-splits 5
```

This writes `data/oof_folds/fold_0..fold_4/{train.csv,validation.csv,purged_train_rows.csv}`, `novel_validation.csv`, `fold_purge_balance.csv`, and `manifest.json`. It does not modify `data/splits`.

Expected Novel Validation count based on the current audit is 11,682. The script reports the actual count and validates it is text-disjoint from all OOF training rows.

## Step 2: Train SVM OOF first (cheap pipeline check)

```bash
python scripts/train_svm_oof.py \
  --config configs/experiment.yaml \
  --folds-dir data/oof_folds \
  --output-dir results/oof \
  --model-dir models/oof/svm \
  --n-splits 5
```

Outputs:
- `results/oof/svm_oof_predictions.csv`: every source training row receives exactly one OOF score.
- `results/oof/svm_novel_fold_predictions.csv`: each Novel Validation sentence is scored by each of the five fold-specific SVM models.
- `models/oof/svm/svm_fold_0.joblib` ... `svm_fold_4.joblib`.

Both `svm_decision_score` and its sigmoid transform `svm_p_machine` are preserved. The sigmoid score is not a calibrated probability.

## Step 3: Train RoBERTa OOF

```bash
python scripts/train_roberta_oof.py \
  --config configs/experiment.yaml \
  --folds-dir data/oof_folds \
  --output-dir results/oof \
  --model-dir models/oof/roberta \
  --n-splits 5 \
  --epochs 3
```

Important design choice: the outer OOF validation fold is never used for early stopping or checkpoint selection. A fixed 3-epoch schedule is used, based on the previously observed best single-split epoch. This is a fixed experimental setting, not fold-specific validation tuning. It may be compared with nested inner-validation training as a later ablation.

Outputs:
- `results/oof/roberta_oof_predictions.csv`: one OOF prediction per original training row.
- `results/oof/roberta_novel_fold_predictions.csv`: each Novel Validation sentence is scored by each of the five fold-specific RoBERTa models.
- fold checkpoints and training histories under `models/oof/roberta/`.

The script saves raw logit margin (`roberta_logit_margin`) as well as softmax score (`roberta_p_machine`), because the latter is currently highly saturated near 0 and 1.

## Why score Novel data with every fold model?

Each OOF base model is trained on about 70.6%-71.0% of the original training rows after fold-specific purging. Scoring Novel with the same five fold models preserves the fold-model score scale for the later meta-model, rather than mixing OOF scores from smaller models with scores from a single model refit on all rows. The later stacking stage should apply the meta-model to each fold's paired SVM/RoBERTa scores and average the five meta predictions.

## Important statistical notes

- The purge removes substantially more machine than human rows in this dataset. Fold files and summaries preserve class counts. Both current baseline implementations use class-weighted learning; retain and report fold-specific class counts.
- There are normalized duplicate-text groups appearing in validation portions of multiple folds. Per-fold train/validation text overlap is still zero, but the OOF meta-table can contain repeated text groups. The later meta-model stage should compare ordinary sentence-row weighting with an inverse-text-frequency weighting ablation; do not silently delete conflicting-label examples.
- The 45 conflicting-label normalized-text groups remain in data. Do not relabel or remove them automatically.
- This stage does not fit a meta-classifier or select a final fusion threshold.
