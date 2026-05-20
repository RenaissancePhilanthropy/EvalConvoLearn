# Services

Orchestration layer between the SDK entry point and the benchmark/model layer.

## `evaluation_service.py` — `EvaluationService`

Runs benchmarks for **FlexLearner** simulations (those built on `BinarySkillsFlexLearner`).

- Accepts an `EvaluationConfig` describing which learner archetypes and benchmarks to run.
- Validates that all skill IDs in learner configs and skill levels exist in the provided `SkillSpace`.
- Builds one `StudentPool` per learner config under a timestamped run directory.
- Dispatches to `PlacementTestBenchmark`, `LearningFromConversationBenchmark`, and `MultiConversationsPracticeBenchmark`.
- Writes a consolidated `evaluation_summary.json` alongside each benchmark's output artifacts.

## `base_learner_evaluation_service.py` — `BaseLearnerEvaluationService`

Mirrors `EvaluationService` but targets **black-box `BaseLearner` subclasses** — learners that implement the `start_or_continue_conversation` / `end_conversation` interface without depending on `StudentPool` or `BinarySkillsFlexLearner`.

Supported benchmarks (lazily imported to avoid circular dependencies):
- `BaseLinePlacementTestBenchmark`
- `BaseLineLearningFromConversationBenchmark`
- `BaselineMultiConversationsBenchmark`
- `DatasetFittedConversationalBenchmark`

## `session_service.py` — `SessionService`, `ConversationSession`, `BaseConversationSession`

Manages interactive tutoring sessions.

- **`ConversationSession`** — drives a `FlexLearner` through a multi-turn conversation with a custom tutor using the `ConversationGraph` engine. Auto-saves session state and student pool practice history after each turn.
- **`SessionService`** — creates, saves, and loads `ConversationSession` instances via `FileSessionStorage` and `FileStudentPoolStorage`.
- **`BaseConversationSession`** — lighter-weight session driver for `BaseLearner` subclasses. Uses the learner's `start_or_continue_conversation` / `end_conversation` API directly, without the `ConversationGraph`.

## `conversation_service.py` — `ConversationService`

Thin wrapper holding an `EvalConvoLearnConfig`. Reserved for future conversation-level orchestration logic.
