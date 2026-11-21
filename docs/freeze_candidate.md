# Stage 1 — Freeze candidate and test inference acceptance

Status: **frozen as the Stage-1 executable inference baseline** (2026-11-22), based on the user's reported complete private-server run and schema-validation PASS. **Not yet confirmed as the teacher's official submission format.**

## Model decision

| Role | Candidate | Current status |
|---|---|---|
| Best measured *development* fusion | Five-fold SVM + RoBERTa Logistic Stacking | Selected Stage-1 baseline: fold-matched Novel results verified; full 500-record/4,264-sentence unlabeled CUDA inference and generic schema validation PASS |
| Lowest-cost executable dry-run | Existing single-model SVM bundle in `models/best_model/svm.joblib` | Uses the existing generic inference entrypoint; threshold from its own bundle was selected on Novel and is **not** the OOF fusion threshold |
| Single-model neural alternative | RoBERTa `models/best_model/bert` | Metadata/tokenizer tracked; learned weight file excluded from Git, must be checked on server |

**Do not** mislabel the old `fusion` option in `scripts/predict.py` as the
five-fold Stacking experiment. It loads a separate `fusion.joblib` embedding
pipeline, not `results/oof_fusion/fusion_models.joblib`.

### Required operational freeze fields for the intended Stacking model

- Source: `results/oof_fusion/fusion_config.json`, seed=42, 5 folds,
  Logistic C=1.0, training groups by source article ID and exact-text purge
  in the meta cross-fitting stage.
- Saved thresholds: Stacking probability **0.585**; individual SVM
  standardized score **-0.34** and RoBERTa standardized score **+0.26**.
  Veto thresholds belong to a different rule, not the Stacking decision.
- Standardization: *per-fold* mean/std of OOF validation scores, reused on
  that fold's Novel scores. Never average raw RoBERTa probabilities across
  fold models as if their scales were calibrated.
- Required model dependencies: five SVM OOF bundles; five RoBERTa OOF
  checkpoint weights+tokenizers; fusion meta models; fold-specific
  normalization; source-group/novel-fold mapping logic; correct feature
  construction and inference order.
- **Established for the current local dataset:** `scripts/predict_oof_stacking.py`
  implements saved fold-matched inference and exact Novel meta-prediction
  replay. The user's private-server logs show a real 16-sentence checkpoint
  dry-run and full 500-article/4,264-sentence inference with all five folds.
  Teacher-specific final submission schema remains unverified.
- No independent blind-test score exists; Novel is repeatedly used for
  development.

## Stage 1 acceptance commands (run on the private server)

All commands assume the repository root:
`/home/zhuangyr/dl-workspace/projects/AIGTD_project_starter/AIGTD`.
Run on a new branch based on PR #4, not the historical `main` branch.

### 1. Inspect; no writes to datasets or weights

```bash
git status --short --branch
git log -1 --oneline
python scripts/check_stage1_artifacts.py
```

The inventory writes only boolean presence flags, a Git SHA/branch and an
aggregate warning to `results/stage1_local_audit/inventory.json`. GitHub
does not track large checkpoints or local raw data; remote absence alone
cannot prove server absence. If model weights are absent, stop; do not
automatically retrain.

### 2. Verify Stacking and protocol table

Follow the two commands in `docs/baseline_protocol_and_results.md` to
generate aggregate corrected statistics and run its fail-closed consistency
verifier. The historical PR #4 output did not include the new corrected
aggregate directory because it is intentionally ignored.

### 3. Safe single-SVM inference dry-run

The current inference script parses the requested raw JSON directly and does
not reuse or overwrite `data/processed/test_sentences.csv`. For a dry-run,
create a **local-only** JSON subset of the first 2 complete article records,
preserving the original `id`, `sent` and optional `domain` fields.
Do not commit the subset.

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path('data/raw/AIGTD_test_without_labels.json')
items = json.loads(root.read_text(encoding='utf-8'))
assert isinstance(items, list) and len(items) == 500
sample = items[:2]
assert all(isinstance(x.get('sent'), list) and x['sent'] for x in sample)
path = Path('data/stage1_local_only')
path.mkdir(parents=True, exist_ok=True)
(path/'test_2_articles.json').write_text(json.dumps(sample, ensure_ascii=False), encoding='utf-8')
print('Sample: 2 article records /', sum(len(x['sent']) for x in sample), 'sentences')
PY

python scripts/predict.py \
  --model-type svm \
  --test data/stage1_local_only/test_2_articles.json \
  --output-dir results/stage1_local_audit/dry_run
```

For the two-article subset, the validator's default expected counts (500 and
4,264) are **not appropriate**; pass the actual sentence count explicitly.
For example:

```bash
python scripts/validate_test_predictions.py \
  --raw-test data/stage1_local_only/test_2_articles.json \
  --pred-json results/stage1_local_audit/dry_run/test_predictions.json \
  --pred-csv results/stage1_local_audit/dry_run/test_predictions.csv \
  --expected-articles 2 \
  --expected-sentences "$(python - <<'PY'
import json
from pathlib import Path
x = json.loads(Path('data/stage1_local_only/test_2_articles.json').read_text(encoding='utf-8'))
print(sum(len(r['sent']) for r in x))
PY
)"
```

No raw text or IDs appear in verification output. This only proves the
*single SVM* dry-run, **not** that the five-fold Stacking candidate can predict.

### 4. Final full test acceptance (only after selecting/fixing actual final model)

```bash
python scripts/predict.py --model-type svm \
  --test data/raw/AIGTD_test_without_labels.json \
  --output-dir results/stage1_local_audit/full_svm

python scripts/validate_test_predictions.py \
  --raw-test data/raw/AIGTD_test_without_labels.json \
  --pred-json results/stage1_local_audit/full_svm/test_predictions.json \
  --pred-csv results/stage1_local_audit/full_svm/test_predictions.csv \
  --summary-out results/stage1_local_audit/full_svm_validation.json
```

Expected: **500 article records, 4,264 sentences**, exactly one
`human`/`machine` label per sentence, stable article and sentence order,
duplicate source IDs allowed, no missing/invalid probability, no extra rows.
This produces a local *generic* output only. The teacher has not supplied
a required final submission schema; adapt before formal submission.

**Current stop/go decision (2026-11-22):** Five-fold OOF Stacking is accepted
as the frozen **Stage-1 engineering baseline**. The saved normalization,
meta-model probabilities and threshold are replay-validated. The server GPU
ran all five base folds and wrote the full unlabeled test predictions; the
JSON/CSV validator passed. The teacher-specific submission contract remains
a separate unresolved deliverable. Do not use unlabeled test-set predictions
for selection, threshold tuning or reporting Accuracy/F1.

**CI update (2026-11-22):** Corrected Stacking aggregate recomputation and the new fail-closed verifier passed on the already-committed GitHub snapshot (run [#44924113524](https://github.com/q3483109023/AIGTD/actions/runs/44924113524)). This verifies the published development aggregate, **not** the local server checkpoints or test-time inference. Operational freeze status is unchanged.


## Recorded private-server acceptance — 2026-11-22

User-provided local command output (aggregate only; no raw artifacts copied into Git):

| Check | Result |
|---|---|
| Local Novel replay | PASS, 11,682 sentences, 5 folds |
| SVM and RoBERTa z-score replay | Maximum absolute delta = 0.0 for both |
| Stacking probability replay | Maximum delta = 2.220446049250313e-16 |
| Frozen Stacking threshold label replay | All labels match |
| Real CUDA two-article dry-run | 2 article records / 16 sentences; Fold 0=8, Fold 1=8; validator PASS |
| Real CUDA full unlabeled test | 500 article records / 4,264 sentences; all five folds invoked |
| Full test fold sentence counts | Fold 0=853, 1=853, 2=853, 3=853, 4=852 |
| Full test predictions | Human=1,548; Machine=2,716 |
| Full output contract checks | PASS: raw JSON order/text, CSV consistency, threshold=0.585 |

The historical Novel Stacking **Macro-F1 0.917374** (vs RoBERTa
0.909330) is **development validation** only. The unlabeled test has no
known ground-truth labels. The saved full-run files are local-only:

- `results/stage1_local_audit/stacking_full/test_predictions.json`
- `results/stage1_local_audit/stacking_full/test_predictions.csv`
- `results/stage1_local_audit/stacking_full_validation.json`

Frozen executable protocol: source-group LPT assignment on the complete
target dataset; matching fold SVM decision score and RoBERTa logit margin;
OOF-fold z score standardization in order `[svm_z, roberta_z]`;
matching saved Logistic Regression meta-model; threshold **0.585**.
**Do not substitute** single-model SVM threshold 0.52, five-fold averaging,
or the unrelated `scripts/predict.py --model-type fusion`.

Remaining before course submission: compare the teacher's actual required
file format against this generic output; create an adapter only if required.
No unsupported claim of unseen test performance should be made.
