# EvalConvoLearn

An open-source evaluation library to assess the quality of learner simulations in conversational tutoring systems.

Learner simulations are AI agents that model how a student learns and responds during tutoring sessions. EvalConvoLearn provides standardized benchmarks to measure whether a simulated learner behaves realistically: placing accurately by prior knowledge, demonstrating skill acquisition after tutoring, and progressing coherently across multiple sessions.

## Code structure

```
src/evalconvolearn/
├── core/               # Core abstractions and SDK entry point
│   ├── sdk.py          # EvalConvoLearn — main interface
│   ├── base_learner.py # BaseLearner (ABC) — black-box learner interface
│   ├── flexlearner.py  # FlexLearner (ABC) — transparent, skill-list-based learner
│   ├── base_tutor.py   # BaseTutor (ABC) — tutor interface
│   └── config.py       # Global configuration
├── models/             # Pydantic data models
│   ├── skill.py        # Skill, SkillSpace (DAG with prerequisite validation)
│   ├── practice_item.py        # PracticeItem, PracticeItemPool
│   ├── binary_skills_flexlearner.py  # BinarySkillsFlexLearner, StudentPool
│   ├── tutor.py                # LLMTutorStrategy, HumanInterfaceTutorStrategy
│   ├── evaluation.py           # LearnerEvalConfig, EvaluationConfig
│   └── evaluation_results.py   # EvaluationResults, EvalSetResults
├── benchmarks/         # Evaluation benchmarks
│   ├── base_learners/          # Black-box learner benchmarks
│   ├── flexlearners/           # Transparent learner benchmarks
│   └── realistic_benchmarks_from_conversation_data/  # Dataset-fitted benchmarks
├── services/           # Orchestration layer (conversations, evaluations, sessions)
├── storage/            # Persistence (CSV-based StudentPool storage)
└── utils/              # LLM grading, metrics, alignment matrices, data loaders
```

**Four benchmark families:**

| Benchmark | What it measures |
|---|---|
| Placement Test | Does the learner answer problems correctly given its initial skill level? |
| Learning from Conversation | Does the learner demonstrate skill gains after a tutored exchange? |
| Multi-Conversation Practice | Does the learner progress coherently across multiple sessions? |
| Dataset-Fitted | Does the learner's conversation behavior match real tutoring data? |

## Installation

```bash
pip install evalconvolearn
# or with uv:
uv add evalconvolearn
```

## Environment variables

Copy `.env-example` to `.env` and fill in the values:

```bash
cp .env-example .env
```

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key used for generating tutor responses and LLM-based grading |
| `SKILL_SPACE_PATH` | No | Path to the skill-space CSV — lets you call `sdk.load_skill_space()` with no argument |
| `TAGGED_PRACTICE_ITEMS_WITH_RESPONSES_CSV` | No | Path to the tagged practice-items CSV — lets you call `sdk.load_practice_items(skill_space)` with no argument |
| `OVERSAMPLED_ITEMS_CSV` | No | Path to the oversampled items CSV — lets you call `sdk.load_oversampled_items(skill_space)` with no argument |

The path variables are optional: if not set, you can pass explicit paths directly to `sdk.load_skill_space(path)`, `sdk.load_practice_items(skill_space, path)`, and `sdk.load_oversampled_items(skill_space, path)`.

## Usage

### 1. Evaluate your own learner simulation

Subclass `BaseLearner` and implement three methods:

```python
from evalconvolearn.core.base_learner import BaseLearner

class MyLearner(BaseLearner):
    def has_skill(self, skill) -> bool:
        # return True if the learner has mastered this skill
        ...

    def start_or_continue_conversation(self, message: str, practice_item=None) -> str:
        # respond to a tutor message; update internal state as needed
        ...

    def end_conversation(self) -> None:
        # finalize session (save state, update knowledge, etc.)
        ...
```

**If you have a dataset of real tutoring conversations**, format it to match the expected schema and run the dataset-fitted benchmark to compare your learner's behavior against real data:

```python
from evalconvolearn.benchmarks.realistic_benchmarks_from_conversation_data import DatasetFittedConversationalBenchmark

benchmark = DatasetFittedConversationalBenchmark(skill_space, practice_item_pool)
results = benchmark.run(MyLearner, dataset=your_formatted_dataset)
```

**If you don't have a dataset**, you first need to prepare the skill space and practice items from the Florida DOE BEST curriculum, then run the standard benchmarks.

#### Step 1 — Complete the skill space

`data/florida-doe/skill-space.csv` ships with skill IDs and descriptions but **no practice problems or misconceptions**. The `problem_1`, `problem_2`, and `misconceptions` columns for each skill can be filled using the [Florida DOE BEST Mathematics curriculum](https://www.fldoe.org/academics/standards/subject-areas/math-science/mathematics/bestmath.stml) for example. This set of skills serves as an example but can be replaced with any set of skills including prerequisite relationships, aligned practice items and example misconceptions.

#### Step 2 — Pivot into a practice-item pool

```bash
python data/florida-doe/data_cleaning/skills-to-practice-items.py
# outputs: data/florida-doe/tagged-practice-items.csv
```

#### Step 3 — Generate tutor responses

```bash
python data/florida-doe/data_cleaning/generate_tutor_responses.py \
    --input  data/florida-doe/tagged-practice-items.csv \
    --output data/florida-doe/tagged-practice-items-with-responses.csv \
    --model  gpt-4.1-mini
# add --resume to pick up after an interruption
```

This calls an OpenAI model (set `OPENAI_API_KEY` in your environment) to produce a `helpful_response` and an `unhelpful_response` for every practice item.

#### Step 4 — Run the benchmarks

```python
from evalconvolearn.core.sdk import EvalConvoLearn
from evalconvolearn.models.evaluation import EvaluationConfig, LearnerEvalConfig
from evalconvolearn.benchmarks.base_learners import (
    BaseLinePlacementTestBenchmark,
    BaseLineLearningFromConversationBenchmark,
    BaselineMultiConversationsBenchmark,
)

sdk = EvalConvoLearn()
skill_space = sdk.load_skill_space("data/florida-doe/skill-space.csv")
item_pool = sdk.load_practice_items("data/florida-doe/tagged-practice-items-with-responses.csv", skill_space)

config = EvaluationConfig(
    learners=[
        LearnerEvalConfig(
            learner_class=MyLearner,
            init_kwargs={"skill_space": skill_space},
            benchmarks=[
                BaseLinePlacementTestBenchmark,
                BaseLineLearningFromConversationBenchmark,
                BaselineMultiConversationsBenchmark,
            ],
        )
    ]
)

results = sdk.run_evaluation(config)
```

### 2. Simulate a tutoring conversation dataset

Use the learner class `FlexLearner` to simulate a dataset of tutoring conversations on a set of practice items and aligned skills. `FlexLearner` implements explicit skill states and learner <> tutor conversations with stateful execution graphs. The framework also allows to subclass `FlexLearner` to test different implementations of it, in particular with different 'knowledge' technical structures.

Run the simulation script directly (requires a practice-item pool from Step 3 above):

```bash
python data/simulated_datasets/simulate_flexlearner_dataset.py \
    --nb-learners 20 \
    --max-init-skills 3 \
    --nb-conversations 7 \
    --seed 42 \
    --pool-id my_simulation
```

Or call it from Python:

```python
from data.simulated_datasets.simulate_flexlearner_dataset import run_simulation

run_simulation(
    nb_learners=20,
    max_init_skills=3,
    nb_conversations=7,
    seed=42,
    pool_id="my_simulation",
)
```

Each learner is initialized with a random sample of skills (prerequisites are automatically closed), assigned a persona and a set of misconceptions, then runs `nb_conversations` sessions against a helpful (90%) or unhelpful (10%) LLM tutor. Sessions are persisted to disk under the pool directory, and a Markdown learning-sequence summary is written alongside them.

See [examples/](examples/) for complete implementations including binary-skill, conversation-history, and knowledge-graph variants.

## Contributors

- [Baptiste Moreau-Pernet](mailto:baptiste@levi.digitalharbor.org)
- [AJ Strauman-Scott](mailto:aj.scott@renphil.org)
