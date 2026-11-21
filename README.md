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


## Stage 3: Data-quality and Context OOF integrity audits

These checks reuse existing split/fold CSVs and do **not** retrain models or
modify source data. Run them before interpreting the Context-window result as
the preferred model.

### Step 1: Locate cross-split conflicting duplicate texts by article

```bash
python scripts/analyze_conflict_article_distribution.py \
  --train data/splits/train.csv \
  --validation data/splits/validation.csv \
  --aggregate-output results/analysis_review/article_conflict_distribution_aggregate.json \
  --detail-dir results/duplicate_data_audit/article_distribution_local_only
```

The script recomputes normalized duplicate groups and reports the number of
conflict groups, the number of involved `article_key` / `article_id` values,
how many conflict groups and rows each article contributes, the distribution
of groups per article, and top-1/top-5/top-10 article concentration. It also
rechecks the handover's expected 38 groups rather than assuming that count.

The aggregate JSON contains counts only. The optional per-article detail CSVs
contain article identifiers and are **local-only**; they are covered by
`.gitignore` and must not be committed or pasted into public issues/PRs.
No raw or normalized sentence text is written by this script.

### Step 2: Independently verify Context OOF purge and Novel mapping

```bash
python scripts/audit_context_oof_independently.py \
  --train data/splits/train.csv \
  --validation data/splits/validation.csv \
  --row-manifest results/oof_purged_group_audit/row_fold_manifest.csv \
  --folds-dir data/context_oof_folds \
  --output-dir results/analysis_review \
  --n-splits 5 \
  --context-window 1 \
  --expected-novel-rows 11682
```

This auditor independently rebuilds previous/target/next windows and the
expected fold-specific training rows from the canonical split files and row
manifest. It then checks:
- exact row membership of each exported fold train/validation CSV;
- serialized context equality with independently reconstructed windows;
- zero overlap between train-window components and outer-validation windows;
- zero overlap between train-window components and Novel Validation windows;
- source-ID and article-key separation;
- both classes remain in each fold's training set;
- every training row appears in exactly one OOF validation fold;
- Novel Validation row IDs, target text, labels, article positions, and context
  match the independently recomputed Novel subset.

Only aggregate counts and PASS/FAIL checks are exported. The script supports
the already-used `context_window=1` protocol and does not change any data.

### Interpretation boundary

A PASS confirms that the materialized fold files match the current canonical
CSVs and manifest under this protocol. It does not prove label semantics are
correct, does not make Novel Validation an independent test set, and does not
explain the Fold 4 ranking-metric degradation. Any mismatch in source CSV
versions, normalization, manifest ordering, or fold exports must be resolved
before proceeding to a stricter de-duplication experiment.


### Synthetic regression tests

Run the synthetic checks without project data or model weights:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

The tests cover conflict-category accounting, text-free aggregate output, a
synthetic two-fold Context OOF materialization, Novel row mapping, and failure
detection after a deliberately corrupted context export. They do not replace
running both audit scripts against the real project split/fold files.


## Stage 4: Finish baseline review without exposing row-level diagnostics

### Step 1: Rebuild corrected Stacking aggregate analysis

The analysis script uses `stacker_p_machine` for Stacking ranking metrics and
per-fold standardized `roberta_z` / `svm_z` for the base models. Do not pool
fold-local raw RoBERTa/SVM probabilities for ROC-AUC or Average Precision.

Run the corrected analysis in aggregate-only mode:

```bash
python scripts/analyze_stacking_errors.py \\
  --input results/oof_fusion/novel_fusion_predictions.csv \\
  --output-dir results/stacking_error_analysis_corrected \\
  --fusion-config results/oof_fusion/fusion_config.json \\
  --bootstrap-reps 2000 \\
  --seed 42 \\
  --aggregate-only
```

This mode writes only aggregate metrics, length/domain summaries, paired
McNemar results, cluster-bootstrap differences, the Stacker/RoBERTa correctness
group summary, and a text summary. It suppresses sentence-level error files,
representative examples, duplicate-text exports, and per-article/per-source-ID
tables. The output directory is ignored by Git by default.

### Step 2: Audit Context target-token retention before any retraining

The Context OOF trainer tokenizes the serialized
`[PREVIOUS] ... [TARGET] ... [NEXT] ...` input with truncation enabled and
`transformer.max_length` (currently 256). The following audit uses fast-tokenizer
offset mappings to count target tokens before and after the same configured
truncation operation for each fold's training and validation windows, plus the
shared Novel Validation windows. It reports fold, split, label,
target-character-length, previous-context-length, and next-context-length
aggregates only.

```bash
python scripts/audit_context_target_token_retention.py \\
  --config configs/experiment.yaml \\
  --folds-dir data/context_oof_folds \\
  --model-dir models/oof/context \\
  --tokenizer-path models/oof/context/context_fold_0 \\
  --output-dir results/context_target_token_retention \\
  --n-splits 5 \\
  --batch-size 256
```

Outputs:
- `fold_summary.csv`: fold/split and Human/Machine target-retention summaries.
- `stratified_summary.csv`: target-length, previous-context-length, and
  next-context-length strata, split by fold and label.
- `run_metadata.json`: tokenizer class, truncation side, max length, counts,
  and privacy/export guarantees.

No row-level text, article identifiers, or sentence identifiers are written.
The target-length buckets (1-70, 71-98, 99-130, >130 characters) follow the
Fold 4 diagnosis boundaries. This audit diagnoses token retention; it does
not itself establish that truncation caused Fold 4's ranking-metric decline.

Do not retrain the five-fold models until the corrected Stacking aggregates and
the Context token-retention report have been reviewed. If the audit shows
material target-token loss, design a separate controlled truncation/window
ablation rather than silently changing the frozen experiment.


### Step 3: Build a protocol-aware unified baseline comparison

After the required aggregate CSVs exist, run:

```bash
python scripts/build_baseline_comparison.py \\
  --results-dir results \\
  --output-dir results/analysis_review
```

This writes:
- `baseline_comparison.csv`: SVM, RoBERTa, Stacking, Veto, Sentence-only,
  and Context metrics with explicit `protocol_id`, dataset, threshold source,
  and aggregation method. The current aggregate-source snapshot is also
  tracked as `results/analysis_review/baseline_comparison_reference.csv`.
- `context_fold_diagnostics.csv`: all five matched Context/Sentence-only
  OOF folds, retaining Fold 4 rather than hiding it in a pooled mean.

The table deliberately separates OOF cross-fit diagnostics, Fold-matched Novel
development results, five-fold mean-score ranking-only results, matched
Context pooled OOF, Context Novel mean-probability diagnostics, and single-
split Novel-threshold-selected results. Do not sort these rows as one
leaderboard. In particular, single-split SVM/RoBERTa thresholds were selected
on Novel Validation and must not be compared as if they were untouched test
results. Raw pooled Context AUC/AP and fold-standardized ranking AUC/AP are
labelled as distinct score aggregation protocols.


## Stage 1 handoff continuation (2026-11-22)

Read `docs/baseline_protocol_and_results.md` for the protocol-separated,
four-decimal baseline comparison, historical corrected-Stacking caveats,
paired error transitions and Fold-4/context limitations.

Read `docs/freeze_candidate.md` for the **candidate-not-yet-frozen**
decision, private-server artifact inventory, corrected Stacking verification,
two-article single-SVM dry-run, and full 500-article/4,264-sentence schema
validation. This is the next small acceptance step; **do not retrain models**.

New local-only checks:

```bash
python -m unittest tests/test_stage1_freeze.py -v
python scripts/check_stage1_artifacts.py
python scripts/verify_stacking_aggregates.py \
  --analysis-dir results/stacking_error_analysis_corrected
python scripts/validate_test_predictions.py \
  --raw-test data/raw/AIGTD_test_without_labels.json \
  --pred-json results/test_predictions.json \
  --pred-csv results/test_predictions.csv
```

The two verifiers fail closed on missing, misaligned, or inconsistent outputs.
These commands require the relevant local artifacts, which are intentionally
not all present in the public GitHub tree. Running the synthetic tests alone
does not prove the real data/model pipeline worked. The default `predict.py`
neural inference requires locally retained weight files; its `fusion` option
is **not** the OOF Stacking implementation. Unlabeled test predictions do
not contain ground truth and have no legitimate test Accuracy/F1.
