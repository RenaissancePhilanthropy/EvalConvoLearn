# Paper Results

Two files reproduce the evaluation results from the paper.

## `eedi_fitted_learner_evals.py`

Runs the full evaluation suite on the Eedi dataset-fitted benchmark. It sweeps all combinations of:

- **Learner type** — `BinarySkillLearner` and `ConversationHistoryLearner`
- **Model pair** — `(model_learner, model_tutor_evals)` entries in `_MODEL_COMBINATIONS`
- **Few-shot count** — set per model pair

Before running, set the `EEDI_SAMPLED_CONVERSATIONS_PATH` env variable (or edit `_CONVERSATIONS_JSONL`) to point to your Eedi conversations JSONL file.

```bash
python examples/paper_results/eedi_fitted_learner_evals.py
```

Outputs land in `outputs/dataset_fitted_evals/<run_label>__<timestamp>/`, one subdirectory per configuration.

## `analyze_evals_results.ipynb`

Notebook for analyzing the output of the eval script. Point `EVAL_OUTPUT_DIR` at your output directory and run the cells to get:

| Section | What it does |
|---|---|
| **Results table** | Comparison of ECL / LB / CONV scores across all configurations |
| **Score plots** | Bar chart with standard error bars, per configuration |
| **Re-score** | Recompute scores with custom metric weights (no re-run needed) |
| **LaTeX tables** | Generates Tables 1 and 2 as in the paper |
| **Per-scenario breakdown** | Score decomposition by mastery scenario |
| **Conversation browser** | Inspect individual dialogues |
| **Human validation sample** | Random sample of conversations + LLM-as-judge labels for manual grading, exportable to CSV |
| **Distribution plots** | Real vs. simulated error-type and talk-move distributions per scenario |
