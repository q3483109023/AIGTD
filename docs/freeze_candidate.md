# Stage 1 — Freeze candidate and test inference acceptance

Status: **candidate selected; NOT yet operationally frozen** (2026-11-22).

## Model decision

| Role | Candidate | Current status |
|---|---|---|
| Best measured *development* fusion | Five-fold SVM + RoBERTa Logistic Stacking | Preferred accuracy/Macro-F1 candidate on fold-matched Novel; **no complete unlabeled-test entrypoint yet** |
| Lowest-cost executable dry-run | Existing single-model SVM bundle in \`models/best_model/svm.joblib\` | Uses the existing generic inference entrypoint; threshold from its own bundle was selected on Novel and is **not** the OOF fusion threshold |
| Single-model neural alternative | RoBERTa \`models/best_model/bert\` | Metadata/tokenizer tracked; learned weight file excluded from Git, must be checked on server |

**Do not** mislabel the old \`fusion\` option in \`scripts/predict.py\` as the
five-fold Stacking experiment. It loads a separate \`fusion.joblib\` embedding
pipeline, not \`results/oof_fusion/fusion_models.joblib\`.

### Required operational freeze fields for the intended Stacking model

- Source: \`results/oof_fusion/fusion_config.json\`, seed=42, 5 folds,
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
- **Not established:** a Stacking test-time assignment/inference and exact
  replay contract for unseen test articles. A new inference path must be
  implemented and verified before freezing this candidate.
- No independent blind-test score exists; Novel is repeatedly used for
  development.

## Stage 1 acceptance commands (run on the private server)

All commands assume the repository root:
\`/home/zhuangyr/dl-workspace/projects/AIGTD_project_starter/AIGTD\`.
Run on a new branch based on PR #4, not the historical \`main\` branch.

### 1. Inspect; no writes to datasets or weights

\`\`\`bash
git status --short --branch
git log -1 --oneline
python scripts/check_stage1_artifacts.py
\`\`\`

The inventory writes only boolean presence flags, a Git SHA/branch and an
aggregate warning to \`results/stage1_local_audit/inventory.json\`. GitHub
does not track large checkpoints or local raw data; remote absence alone
cannot prove server absence. If model weights are absent, stop; do not
automatically retrain.

### 2. Verify Stacking and protocol table

Follow the two commands in \`docs/baseline_protocol_and_results.md\` to
generate aggregate corrected statistics and run its fail-closed consistency
verifier. The historical PR #4 output did not include the new corrected
aggregate directory because it is intentionally ignored.

### 3. Safe single-SVM inference dry-run

The current inference script parses the requested raw JSON directly and does
not reuse or overwrite \`data/processed/test_sentences.csv\`. For a dry-run,
create a **local-only** JSON subset of the first 2 complete article records,
preserving the original \`id\`, \`sent\` and optional \`domain\` fields.
Do not commit the subset.

\`\`\`bash
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
\`\`\`

For the two-article subset, the validator's default expected counts (500 and
4,264) are **not appropriate**; pass the actual sentence count explicitly.
For example:

\`\`\`bash
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
\`\`\`

No raw text or IDs appear in verification output. This only proves the
*single SVM* dry-run, **not** that the five-fold Stacking candidate can predict.

### 4. Final full test acceptance (only after selecting/fixing actual final model)

\`\`\`bash
python scripts/predict.py --model-type svm \
  --test data/raw/AIGTD_test_without_labels.json \
  --output-dir results/stage1_local_audit/full_svm

python scripts/validate_test_predictions.py \
  --raw-test data/raw/AIGTD_test_without_labels.json \
  --pred-json results/stage1_local_audit/full_svm/test_predictions.json \
  --pred-csv results/stage1_local_audit/full_svm/test_predictions.csv \
  --summary-out results/stage1_local_audit/full_svm_validation.json
\`\`\`

Expected: **500 article records, 4,264 sentences**, exactly one
\`human\`/\`machine\` label per sentence, stable article and sentence order,
duplicate source IDs allowed, no missing/invalid probability, no extra rows.
This produces a local *generic* output only. The teacher has not supplied
a required final submission schema; adapt before formal submission.

**Current stop/go decision:** Stacking remains the research front-runner;
the pipeline is not yet accepted for test-time use. Do not call the model
"finally frozen" until its exact inference path, weights, thresholds,
standardization, and final output are replay-validated. If insufficient time,
choose a proven executable baseline, with its distinct evaluation protocol
clearly disclosed. No test-set metric can be reported without labels.
