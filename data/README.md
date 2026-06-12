# data/

Source data and generation scripts for EvalConvoLearn.

> **Note:** Example CSV files (practice items, tutor responses) are distributed with this repository. Follow the data preparation steps in the main [README](../README.md) to update them locally.

---

## `data_utils/`

Shared utility scripts.

- **`add_tutor_learner_mock_responses.py`** — Reads a practice-items CSV and fills in `helpful_response`, `unhelpful_response`, `learner_response_helpful`, and `learner_response_unhelpful` via LLM calls. Writes back after every row so progress survives interruption.

  > **When to use:** Only needed when running multi-turn assessments with `max_assessment_turns = 2`. The default evaluation flow uses single-turn tutor responses only; the `learner_response_*` columns are not required for standard benchmarks.

---

## `eedi_tutoring/`

Scripts for extracting, tagging, and reviewing real Eedi tutoring conversations to showcase EvalConvoLearn learner evaluations on an existing, open-source dataset.

- **`extract_tag_store_eedi_tutoring_conversations.py`** — Full pipeline: loads the [Eedi HuggingFace dataset](https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k/viewer/dq-question-metadata), filters dialogues by learner-turn ratio, samples tutors, uses an LLM to align each question to the Florida DOE skill space, infers learning outcomes, and saves matched conversations as JSONL. Key functions: `select_sampled_dialogues`, `tag_question_to_skill`, `infer_learning_outcome`, `save_jsonl`.

- **`tag_eedi_sampled_convs_with_metrics.py`** — Reads the output JSONL from the script above and adds `conversation_metrics` (error types, talk moves, turn length) computed by `compute_conversation_metrics` from the benchmark module. Key function: `main`.

- **`export_eedi_tagging_review_samples.py`** — Samples tagged conversations and writes three markdown review packets for manual quality checks: skill-tagging, learner mastery, and talk-move/error-type labeling. Key functions: `_render_skill_review`, `_render_mastery_review`, `_render_behavior_review`.

---

## `florida-doe/`

[Florida DOE BEST curriculum](https://www.fldoe.org/academics/standards/subject-areas/math-science/mathematics/bestmath.stml) skill space and practice-item data for 6th-grade mathematics, used as the default curriculum ontology (skill prerequisites, practice items, misconceptions).

### Files

- **`skill-space.csv`** *(example skills and problems - can be extended)* — Master skill catalogue with columns `skill_id`, `skill_description`, `prerequisite_skills`, `problem_1`, `problem_2`, `misconceptions`. The `skill_id`, `skill_description` and `prerequisite_skills` columns are extracted from the Florida DOE BEST curriculum example. The `problem_1`, `problem_2`, `misconceptions` columns are examples of skill-aligned problems and possible student misconceptions. This data is a working example but users are encouraged to use their own skill ontology and aligned problems.

- **`tagged-practice-items-with-responses.csv`** *(generated — not distributed)* — Practice items with `helpful_response` and `unhelpful_response` columns. Produced by running the two `data_cleaning/` scripts below in order.

#### `oversampled_items/`

- **`oversampled-items-x10.csv`** *(not distributed)* — Example placeholder for expanded practice-item pool with ~10× more problems per skill than `skill-space.csv`. Columns: `problem`, `answer`, `incorrect_answer_1`, `incorrect_answer_2`, `incorrect_answer_3`, `skill_id`, `prerequisite_skills`, `generated`. Can be created by generating new problems that are similar to the examples in `skill-space.csv`, covering the same skill and misconception space. Set `generated=true` on all synthesized rows to distinguish them from hand-authored items.

### `data_cleaning/`

Run these scripts in sequence to build the practice-item pool from scratch.

**1. `skills-to-practice-items.py`** — Pivots `skill-space.csv` into `tagged-practice-items.csv` (one row per example problem per skill). Run from the repo root:

```bash
python data/florida-doe/data_cleaning/skills-to-practice-items.py
```

**2. `generate_tutor_responses.py`** — Calls an OpenAI model to generate `helpful_response` and `unhelpful_response` for every row in the practice-items CSV. Supports `--resume` to skip already-processed rows. Key function: `_generate_response`.

```bash
python data/florida-doe/data_cleaning/generate_tutor_responses.py \
    --input  data/florida-doe/tagged-practice-items.csv \
    --output data/florida-doe/tagged-practice-items-with-responses.csv \
    --model  gpt-4.1-mini
```

---

## `simulated_datasets/`

Scripts for generating synthetic FlexLearner conversation datasets in order to test EvalConvoLearn on simulated tutoring conversations (to control for generated learner features).

- **`simulate_flexlearner_dataset.py`** — Runs end-to-end simulation: creates a learner pool, assigns random skill sets (with prerequisite closure), samples personas and misconceptions, then runs N conversations per learner against a helpful (90%) or unhelpful (10%) LLM tutor. Saves sessions to a pool directory and writes a learning-sequence summary. Key functions: `run_simulation`, `build_initial_skill_set`.

- **`simulation_utils.py`** — Post-simulation analysis helpers. Key functions: `load_conversations` (parse `all_conversations.jsonl`), `group_by_learner` (sort by conv index), `generate_learning_sequence_summary` (write a Markdown progression report). Also runnable as a CLI with `--conversations-file` or `--pool-dir`.
