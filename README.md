# EmoAudio_DataAnalysis

Initial data visualisation plus a manual 5-fold evaluation workflow.

## Manual 5-Fold Evaluation

The fold pipeline only uses evaluated samples from:

- `data/user_emotion_responses.json`: user-rated targets
- `data/song_emotion_ground_truth.csv`: song-level input vectors

The workflow is manual by design:

1. prepare all 5 deterministic folds once
2. run only Fold 1
3. review Fold 1 in Streamlit
4. explicitly approve Fold 2
5. run Fold 2 manually
6. repeat until Fold 5

There is no automatic loop that runs all folds.

### 1. Prepare Folds

```bash
python3 -m evaluation.cli prepare
```

This creates:

- `splits/eligible_samples.csv`
- `splits/fold_1/train.csv` ... `splits/fold_5/test.csv`
- `state/manual_cv_state.json`
- validation reports under `state/agent_reports/`

### 2. Run Fold 1

```bash
python3 -m evaluation.cli run-fold --fold 1
```

This saves Fold 1 outputs in:

- `results/fold_1/predictions.csv`
- `results/fold_1/test_items.csv`
- `results/fold_1/model_summary.json`
- `results/fold_1/metrics_summary.json`

### 3. Review Fold 1

Launch Streamlit:

```bash
streamlit run app.py
```

Open the `Manual Fold Review` section to inspect:

- the current fold
- its 20% test split
- predictions vs ground truth
- aggregate metrics

Use the review buttons to:

- mark the fold as reviewed
- explicitly approve proceeding to the next fold

### 4. Proceed to Fold 2

Only after Fold 1 has been reviewed and approved:

```bash
python3 -m evaluation.cli run-fold --fold 2
```

Then repeat the same review-and-approve cycle for Folds 2 to 5.

### Useful Commands

Show current state:

```bash
python3 -m evaluation.cli status
```

Mark a fold reviewed from the CLI if needed:

```bash
python3 -m evaluation.cli review --fold 1 --approve-next
```

Generate validation summary:

```bash
python3 -m evaluation.cli validate
```
