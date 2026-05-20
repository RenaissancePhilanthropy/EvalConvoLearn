# Benchmarks

Three families of benchmarks for evaluating learner simulations.

---

## 1. BaseLearner benchmarks (`base_learners/`)

Black-box benchmarks that treat the learner as an opaque agent — no access to internal knowledge state. They call the `BaseLearner` interface only.

| Class | What it measures |
|---|---|
| `BaseLinePlacementTestBenchmark` | Does the learner answer correctly for skills it has, and incorrectly for skills it lacks? |
| `BaseLineLearningFromConversationBenchmark` | Does the learner learn a skill after a helpful tutor response, and *not* learn after an unhelpful one? Requires pre-loaded mocked tutor responses. |
| `BaselineMultiConversationsBenchmark` | Can the learner progressively master a target skill by climbing its prerequisite graph through successive tutoring conversations? |

All three produce a JSONL results file and a JSON summary under the configured `output_dir`.

---

## 2. FlexLearner benchmarks (`flexlearners/`)

White-box benchmarks that have access to the learner's internal mastered-skill set. They use `FlexLearner` and `BinarySkillsFlexLearner` to drive learning and assess knowledge state directly.

| Class | What it measures |
|---|---|
| `PlacementTestBenchmark` | Alignment between learner answers and expected skill mastery, with and without a prior knowledge check. |
| `LearningFromConversationBenchmark` | Alignment between skills learned from helpful/unhelpful mock conversations and the expected learning matrix. Supports both skill-set alignment mode and pre/post-test mode. |
| `MultiConversationsPracticeBenchmark` | Progressive mastery of a target skill by climbing its prerequisite graph, followed by consolidation runs. Reports average turns per skill and consolidation solution rate. |

`FlexLearnerBenchmark` is the abstract base class shared by all three.

---

## 3. Dataset-fitted benchmarks (`realistic_benchmarks_from_conversation_data/`)

Benchmarks that compare simulated conversation behavior to a real tutoring-conversation dataset loaded from a JSONL file.

| Class | What it measures |
|---|---|
| `DatasetFittedConversationalBenchmark` | Distance between simulated and real conversation metrics (learner turn length, question rate, error types, talk moves), plus a learning-behavior score based on solution-found rates. Produces a composite **EvalConvoLearn score** (`0.6 × LB + 0.4 × Conv`). |

The benchmark samples real conversations by skill and mastery group, runs the learner on each, computes per-conversation metrics (optionally cached), and aggregates distances per scenario (mastered/unmastered × prerequisites met/not met).

---

## Common patterns

- Every benchmark exposes `run_all_evaluations() -> Path`, which writes results to `output_dir` and returns the primary output path.
- Every benchmark exposes a static `compute_structured_metrics(output_file)` method for post-hoc analysis of a saved results file.
- Benchmark-specific options are passed via `benchmark_extra_args: dict`.
