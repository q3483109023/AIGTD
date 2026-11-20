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

## Stage 2: Context-window OOF experiment (strict window purge)

This experiment compares two models on the EXACT same prepared fold rows:
- `sentence_matched`: RoBERTa receives only the target sentence.
- `context`: RoBERTa receives previous + target + next sentence.

Both use source-ID grouped 5-fold OOF, fixed 3 epochs (no outer-fold early stopping), the same model/configuration, class-weighted loss, and identical strict window-purged training rows. This matched control is necessary: comparing context against the old sentence-only OOF would confound context with the stricter purge/data-volume change.

### Step 1: Prepare and audit context windows

```bash
python scripts/prepare_context_oof_folds.py \
  --train data/splits/train.csv \
  --validation data/splits/validation.csv \
  --row-manifest results/oof_purged_group_audit/row_fold_manifest.csv \
  --output-dir data/context_oof_folds \
  --n-splits 5 \
  --context-window 1
```

The script builds windows on complete article records before selecting target rows. A candidate training row is purged if ANY normalized sentence component in its previous/target/next window appears in an outer validation window OR in a Novel Validation target window. It asserts zero component overlap, source-ID overlap, and article-key overlap. Novel target rows are selected using the existing exact normalized target-sentence definition, but their context is built from the full validation article sequence.

Inspect `data/context_oof_folds/fold_purge_balance.csv` before training. If a fold has very few rows, extreme purge rate, or class imbalance, stop and review rather than weakening the purge without documenting a new protocol.

### Step 2: Train the matched sentence-only control

```bash
python scripts/train_context_oof.py \
  --config configs/experiment.yaml \
  --folds-dir data/context_oof_folds \
  --output-dir results/context_oof \
  --model-dir models/oof/context_sentence_matched \
  --n-splits 5 --epochs 3 \
  --input-column text --run-name sentence_matched
```

### Step 3: Train the context model

```bash
python scripts/train_context_oof.py \
  --config configs/experiment.yaml \
  --folds-dir data/context_oof_folds \
  --output-dir results/context_oof \
  --model-dir models/oof/context \
  --n-splits 5 --epochs 3 \
  --input-column context_text --run-name context
```

Each run writes OOF predictions (one prediction per original train row), five-fold Novel Validation predictions, fold summaries, and fold checkpoints. Compare `sentence_matched` against `context` first; the old `roberta_oof_predictions.csv` is a secondary historical reference because its target-only purge is less strict.

No threshold tuning, stacking refit, or final test prediction is performed by these scripts. Evaluate paired OOF metrics and article/source-cluster bootstrap first. Novel fold predictions are diagnostic validation results, not an independent test set.
