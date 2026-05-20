# `utils` — Shared Utilities

## `llm_evaluator.py`

LLM-based judges for learner evaluation using structured output responses.

| Symbol | What it does |
|---|---|
| `evaluate_response_correctness` | Judges whether a learner's response correctly answers a problem (single turn). |
| `did_learner_find_solution_in_turns` | Judges whether a learner found a correct solution across all their conversation turns. |
| `evaluate_knowledge_sufficiency` | Judges whether a learner's knowledge state is sufficient to answer a problem. |
| `classify_conversation_behaviors` | Labels conversation-level learner error types and talk moves. |
| `CorrectnessVerdict`, `SolutionFoundVerdict`, `KnowledgeSufficiencyVerdict`, `ConversationBehaviorLabelsVerdict` | Pydantic response models returned by the judges above. |

## `alignment_matrices.py`

Generates expected-outcome matrices used to score binary benchmark evals.

| Symbol | What it does |
|---|---|
| `generate_placement_test_alignment_matrix` | Returns `{level: {correct: set, incorrect: set}}` — which skills a learner at each proficiency level should answer correctly on a placement test. |
| `generate_learning_alignment_matrix` | Returns `{level: {skill_id: {helpful_response: bool, unhelpful_response: bool}}}` — whether each skill should be learned given a helpful vs. unhelpful tutor response. |

## `benchmark_metrics.py`

Metric helpers consumed by benchmark runners.

| Symbol | What it does |
|---|---|
| `calculate_placement_test_alignment` | Given a `PlacementTestResult` and the expected correct/incorrect skill sets, returns `(expected_correct, is_aligned)`. |

## `benchmark_results.py`

Result tracking and console output for benchmark runs.

| Symbol | What it does |
|---|---|
| `PlacementTestResult` | Collects per-run placement test records and serialises them to JSONL. |
| `print_placement_test_results` | Pretty-prints a single placement test run dict to stdout. |
| `print_lfc_results` | Pretty-prints all Learning-From-Conversation benchmark records from a JSONL file, grouped by check mode. |

## `learner_configs.py`

Preset learner configurations and skill-level definitions used across benchmarks.

| Symbol | What it does |
|---|---|
| `get_placement_test_skill_levels` | Returns the canonical `{level: [skill_ids]}` map for beginner / intermediate / expert learners. |
| `get_beginner_config` / `get_intermediate_config` / `get_expert_config` | Convenience wrappers returning `{"mastered_skills": [...]}` for each proficiency tier. |
| `get_blank_config` | Returns a config with no mastered skills (tabula-rasa learner). |

## `data_loaders.py`

File-system helpers for locating datasets and loading benchmark inputs.

| Symbol | What it does |
|---|---|
| `get_data_dir` | Returns the project-root `data/` directory as a `Path`. |
| `get_florida_doe_data_dir` | Returns `data/florida-doe/`. |
| `load_tagged_skill_ids` | Loads the set of skill IDs from the tagged practice items CSV. |
| `get_tutor_responses_csv_path` | Returns the path to the tutor responses CSV (accepts an optional override). |
| `load_tutor_responses_mapping` | Loads the tutor responses CSV into a `{problem_text: {helpful_response, unhelpful_response, ...}}` dict. |
| `get_benchmark_output_dir` | Returns (and creates if needed) `data/benchmark_evaluations/`. |
| `render_conversation_messages` | Formats a list of message dicts or LangGraph message objects into a readable string for LLM prompts. |
