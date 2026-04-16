# EmoAudio

Streamlit dashboard plus a fold-based LLM analysis workflow for music-emotion annotations.

## Data Sources

The LLM fold workflow uses only these checked-in files as source data:

- `data/user_emotion_responses.json`
- `data/song_emotion_ground_truth.csv`

Test users are selected from the JSON. Songs are included only if they appear in those users' responses and also have a matching row in the ground-truth CSV.

## Main App

Run the app with:

```bash
streamlit run app.py
```

The sidebar contains:

- `Panoramica Dataset`
- `🕷️ Spider Charts`
- `Similarity Analysis`
- `LLM Analysis`

## Final LLM Workflow

The current fold workflow is the `LLM Analysis` flow in the Streamlit app.

Each completed fold saves:

- annotation CSVs in `data/annotations/fold_N/`
- a run manifest in `data/annotations/fold_N/run_manifest.json`
- a fold summary in `state/fold_N_summary.json`
- agent reports in `state/agent_reports/`
- persisted fold metrics in `state/llm_analysis/fold_N_metrics.json`
- persisted aggregate metrics in `state/llm_analysis/aggregate_metrics.json`

That means the fold outputs and analysis remain available the next time you start Streamlit.

## Switching From Mock To Live

Before a real run:

1. Set `USE_MOCK = False` in `annotation/llm_clients.py`.
2. Provide `OPENROUTER_API_KEY` through Streamlit secrets or an environment variable.
3. Remove old mock fold outputs before starting a live run.

Recommended cleanup targets:

```bash
rm -rf data/annotations/fold_* state/fold_workflow.json state/fold_*_summary.json state/user_folds.json state/llm_analysis state/agent_reports/fold_*_report.json
```

The app now blocks re-running a fold if existing annotation CSVs do not have a matching run manifest or if the saved fold was created in a different mode (`mock` vs `live`).

## Running Each Fold

1. Launch `streamlit run app.py`.
2. Open `LLM Analysis -> Overview`.
3. Click `Run Next Fold`.
4. Review the saved fold output in:
   - `LLM Analysis -> Overview`
   - `LLM Analysis -> Fold Comparison`
   - `LLM Analysis -> Cross-Model Analysis`
   - `LLM Analysis -> Agent Reports`
5. If the fold is acceptable, click `Approve Fold N`.
6. Repeat for the next fold.

Folds are intentionally manual. Fold `N+1` stays locked until Fold `N` is completed and approved.

## Per-Fold Checks

Each fold automatically runs the supervisor after annotation generation. The supervisor runs:

- `security_agent`
- `quality_agent`
- `consistency_agent`

If these checks fail, the fold run fails.

## Statistical Analysis

Per-fold and aggregate analysis includes:

- MAE
- RMSE
- Pearson correlation
- Spearman correlation
- cosine similarity
- top-emotion accuracy
- Krippendorff alpha

These metrics compare:

- `human_test`
- `human_consensus`
- `deepseek`
- `gemini`
- `mistral`
- `ground_truth`

## Important Note About The Prompt

The live LLM prompt does not use audio files. It gives the model the ground-truth 8-emotion song vector and asks it to predict average listener ratings from that information.

If you want an audio-based system instead, the prompting and annotation design would need to change.
