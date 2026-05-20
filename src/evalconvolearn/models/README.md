# `models/` — Domain models and data structures

This package contains all Pydantic models and data structures used across the library.

## Knowledge & content

### `skill.py` — `Skill`, `SkillSpace`

- **`Skill`** — A unit of learnable knowledge with an ID, description, and list of prerequisite skill IDs.
- **`SkillSpace`** — An ordered, validated collection of `Skill` objects. Internally backed by a directed acyclic graph (NetworkX) for prerequisite traversal. Key methods: `get_all_prerequisites`, `get_bfs_skill_order`, `get_all_subgraphs_of_skill_prerequisites`, `choose_skills_for_item` (LLM-assisted skill tagging), `load_skills_from_csv`.

### `practice_item.py` — `PracticeItem`, `PracticeItemPool`

- **`PracticeItem`** — A problem text with associated skill IDs, a correct answer, and optional multiple-choice distractors.
- **`PracticeItemPool`** — A validated collection of `PracticeItem` objects tied to a `SkillSpace`. Key methods: `get_items_with_unique_skill`, `get_items_for_skill_scenario`, `load_items_from_json`, `load_items_from_csv`.

## Learner implementations

### `binary_skills_flexlearner.py` — `BinarySkillsFlexLearner`, `StudentPool`

- **`BinarySkillsFlexLearner`** — The default `FlexLearner` implementation. Knowledge is represented purely as the list of mastered skill IDs; no external knowledge store is used.
- **`StudentPool`** — A named group of `FlexLearner` instances sharing a practice-conversation file. Supports creation, persistence, and CSV-based loading of learner state.

## Conversation runners

### `flexlearner_conversation.py` — `ConversationGraph`

LangGraph-backed multi-turn conversation engine for `FlexLearner` simulations. Manages the practice → confusion → solution loop, calls the learner's prompt-generation hooks, integrates skill-guardrail learning updates, and persists conversation state in a SQLite checkpoint.

### `base_learner_conversation.py` — `run_base_learner_conversation`, `BaseConversationResult`

Lightweight, dependency-free conversation runner for `BaseLearner` subclasses. Drives multi-turn exchanges using only `start_or_continue_conversation` / `end_conversation`, with optional scripted or live tutor responses.

## Tutors

### `tutor.py` — `Tutor`, `TutorResponse`, `LLMTutorStrategy`, `HumanInterfaceTutorStrategy`

- **`TutorResponse`** — Standard response envelope (`message` + `metadata` dict).
- **`BaseTutorStrategy`** — Abstract strategy interface (implement `generate_strategy_response`).
- **`LLMTutorStrategy`** — OpenAI-backed tutor with configurable helpfulness, response length, few-shot grounding, and optional conversation-end detection.
- **`HumanInterfaceTutorStrategy`** — Stub strategy for human-in-the-loop tutoring.
- **`Tutor`** — Pydantic model wrapping a strategy; supports `return_only` mode (no HTTP) or `http` mode (calls a running learner API server).

## Evaluation

### `evaluation.py` — `EvaluationConfig`, `LearnerEvalConfig`

Configuration models for evaluation runs.

- **`LearnerEvalConfig`** — Describes one learner archetype: its class, initial skill set or proficiency level, optional knowledge-init kwargs, and which benchmarks to run.
- **`EvaluationConfig`** — Top-level config grouping one or more `LearnerEvalConfig` entries with benchmark selection, runs-per-scenario, and optional output directory.

### `evaluation_results.py` — `EvaluationResults`, `EvalSetResults`, `BenchmarkRunSummary`

Result dataclasses returned by the SDK's `run_evaluation` and `aggregate_results` methods.

- **`BenchmarkRunSummary`** — Status and output for a single (benchmark × learner config) run.
- **`EvaluationResults`** — Aggregated result from one `run_evaluation` call; exposes `all_passed`, `failed_summaries`, `output_paths`.
- **`EvalSetResults`** — Cross-run aggregate from `aggregate_results`; groups metrics by benchmark × learner type. Supports `print_summary()` and `save()`.

### `placement_test.py` — `PlacementTest`, `PracticeItemResult`

Placement-test harness used by placement-test benchmarks.

- **`PracticeItemResult`** — Result of administering one practice item (correct/incorrect, skill check, LLM-generated answer details).
- **`PlacementTest`** — Administers items to a `FlexLearner`, handles multiple-choice and open-ended formats, and optionally uses an LLM to generate and evaluate answers.
