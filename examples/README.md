# Examples

End-to-end examples for the `evalconvolearn` library. Run all scripts from the project root.

---

## `base_learner/`

Implementations of `BaseLearner` — a black-box learner interface evaluated purely on input/output behavior rather than internal knowledge representation.

| File | Description |
|------|-------------|
| `binary_skill_learner.py` | Tracks mastery as a binary set of skill IDs. Tags each practice item with relevant skills via LLM, behaves as a struggling or competent student accordingly, and marks demonstrated skills as mastered after each conversation. |
| `conversation_history_learner.py` | Stores knowledge as natural-language summaries extracted from past conversations. Uses these summaries as in-context knowledge when generating student responses. |

---

## `flexlearner/`

Implementations of `FlexLearner` — a more flexible learner interface that exposes its knowledge representation to the benchmarking framework.

| File | Description |
|------|-------------|
| `flexlearner_basic_usage.py` | Minimal example: create a learner pool, run a conversation with a custom rule-based tutor. |
| `flexlearner_conversation_history.py` | `ConversationHistoryLearner` — knowledge stored as a list of natural-language summaries, updated after each conversation via LLM summarisation. |
| `flexlearner_knowledge_graph.py` | `KnowledgeGraphLearner` — knowledge stored as a property graph + vector store. Triplets are extracted from conversations and relevant knowledge is retrieved via cosine-similarity lookup. |

---

## `evaluations/`

Evaluation scripts. Each script loads data, configures one or more learner archetypes, and runs the specified benchmarks.

### Base Learner evaluations (`sdk.run_base_learner_evaluation`)

| File | Benchmarks | Learners tested |
|------|-----------|-----------------|
| `base_learner_placement_test.py` | `BaseLinePlacementTestBenchmark` | Both base learners |
| `base_learner_learning_from_conversation.py` | `BaseLineLearningFromConversationBenchmark` | Both base learners |
| `base_learner_multi_conv.py` | `BaselineMultiConversationsBenchmark` | Both base learners |
| `base_learner_student_metrics.py` | `DatasetFittedConversationalBenchmark` | Both base learners × model × few-shot count |

### FlexLearner evaluations (`sdk.run_evaluation`)

| File | Benchmarks | Learners tested |
|------|-----------|-----------------|
| `flexlearner_placement_test.py` | `PlacementTestBenchmark` | All 3 FlexLearner implementations |
| `flexlearner_learning_from_conversation.py` | `LearningFromConversationBenchmark` | All 3 FlexLearner implementations |
| `flexlearner_multi_conv.py` | `MultiConversationsPracticeBenchmark` | All 3 FlexLearner implementations |

---

## `learner_utils/`

Utilities for managing learner pools and running interactive sessions.

| File | Description |
|------|-------------|
| `create_and_load_learner_pool.py` | Create a pool, add multiple learners, run sessions, save and reload the pool. |
| `manual_tutor.py` | Interactive command-line tutoring session where you type the tutor responses. |
