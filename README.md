# EvalConvoLearn

An open-source evaluation library to assess the quality of learner simulations in conversational tutoring systems.

A learner simulation is any custom code that can respond to tutor questions during problem-solving tutoring sessions. To be evaluated, it should subclass the `BaseLearner` class (see below).

EvalConvoLearn evaluates the realism of learner simulations by comparing their responses to real student responses extracted from real tutoring conversations datasets.

It also provides standardized scenario-based evaluations to measure whether a learner simulation behaves as expected: responding appropriately to reflect some prior knowledge, demonstrating skill acquisition from tutoring, or progressing coherently across multiple sessions.

## Citation

This library accompanies the [Evaluating Learner Simulations with EvalConvoLearn paper](Evaluating_learner_simulations_with_EvalConvoLearn.pdf) presented at the [IRAISE](https://safeinsights.github.io/iraise26/) '26 workshop at the Festival of Learning:

```
Upcoming citation.
```

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
| `ANTHROPIC_API_KEY` | No | Anthropic API key — required when using Claude models (e.g. `claude-sonnet-4-6`) as the tutor or evaluator |
| `SKILL_SPACE_PATH` | No | Path to the skill-space CSV — lets you call `sdk.load_skill_space()` with no argument |
| `TAGGED_PRACTICE_ITEMS_WITH_RESPONSES_CSV` | No | Path to the tagged practice-items CSV — lets you call `sdk.load_practice_items(skill_space)` with no argument |
| `OVERSAMPLED_ITEMS_CSV` | No | Path to the oversampled items CSV — lets you call `sdk.load_oversampled_items(skill_space)` with no argument |
| `EEDI_SAMPLED_CONVERSATIONS_PATH` | No | Path to the Eedi conversations JSONL file — used by `examples/paper_results/eedi_fitted_learner_evals.py` |

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

**If you have a dataset of real tutoring conversations**, format it as a JSONL file (one JSON object per line) and run the dataset-fitted benchmark to compare your learner's behavior against real data:

```python
from evalconvolearn.benchmarks.realistic_benchmarks_from_conversation_data import (
    DatasetFittedConversationalBenchmark,
)
from evalconvolearn.models.evaluation import LearnerEvalConfig

learner_config = LearnerEvalConfig(
    learner_class=MyLearner,
    init_kwargs={"skill_space": skill_space},
)
benchmark = DatasetFittedConversationalBenchmark(
    skill_space=skill_space,
    practice_item_pool=practice_item_pool,
    learner_config=learner_config,
    skill_levels={},
    benchmark_extra_args={
        "conversations_jsonl_path": "path/to/your/conversations.jsonl",
    },
)
summary_path = benchmark.run_all_evaluations()
```

Each line of the JSONL file must be a JSON object with at minimum:

| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | Unique conversation identifier |
| `item_skills` | `list[str]` | Skill IDs aligned to the practice item (must match your skill space) |
| `practice_item_text` | `str` | The problem presented to the learner |
| `dialogue_history` | `list[dict]` | Turns as `{"role": "assistant"/"user", "content": "..."}` (tutor = `"assistant"`, learner = `"user"`) |
| `mastered_skills_before_conversation` | `list[str]` | Skill IDs the learner had mastered before this session |
| `mastered_skills_from_conversation` | `list[str]` | Skill IDs the learner mastered *during* this session |

Optional fields: `learner_id`, `tutor_id`, `correct_answer`, `item_skill_prerequisites`.

**If you don't have a dataset**, you first need to prepare the skill space and practice items from the Florida DOE BEST curriculum, then run the standard benchmarks.

#### Step 1 — Complete the skill space

`data/florida-doe/skill-space.csv` ships with skill IDs, descriptions, and **generated worked example problems for all 15 skills**, so you can run the steps below immediately. You can extend or replace this content using the [Florida DOE BEST Mathematics curriculum](https://www.fldoe.org/academics/standards/subject-areas/math-science/mathematics/bestmath.stml) or substitute your own skill set — any CSV with `skill_id`, `skill_description`, `prerequisite_skills`, and `problem_1...N` columns will work.

#### Step 2 — Pivot into a practice-item pool

> **Pre-generated file included:** `data/florida-doe/tagged-practice-items.csv` ships with the repo. Skip this step unless you modified `skill-space.csv`.

```bash
python data/florida-doe/data_cleaning/skills-to-practice-items.py
# outputs: data/florida-doe/tagged-practice-items.csv
```

#### Step 3 — Generate tutor responses

> **Pre-generated file included:** `data/florida-doe/tagged-practice-items-with-responses.csv` ships with the repo. Skip this step unless you want to regenerate responses with a different model.

```bash
python data/florida-doe/data_cleaning/generate_tutor_responses.py \
    --input  data/florida-doe/tagged-practice-items.csv \
    --output data/florida-doe/tagged-practice-items-with-responses.csv \
    --model  gpt-4.1-mini
# add --resume to pick up after an interruption
```

This calls an OpenAI model (set `OPENAI_API_KEY` in your environment) to produce a `helpful_response` and an `unhelpful_response` for every practice item. Note that the responses are generated based on the **problem text and skill description alone** — they are model tutor responses to the problem, not reactions to any specific learner message.

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

> **This section is independent from the evaluation benchmarks above.** You can use the simulation pipeline on its own to generate synthetic tutoring datasets without running any benchmarks, and you can run the benchmarks with your own real data without ever using the simulation.

Use the learner class `FlexLearner` to simulate a dataset of tutoring conversations on a set of practice items and aligned skills. `FlexLearner` implements explicit skill states and learner <> tutor conversations with stateful execution graphs. The framework also allows to subclass `FlexLearner` to test different implementations of it, in particular with different 'knowledge' technical structures.

> **Prerequisite:** this script requires a completed skill space and practice-item pool. Complete Steps 1–3 from the section above first, or ensure `skill-space.csv` has `problem_1`, `problem_2`, and `misconceptions` filled in (the file ships with several worked examples to get you started).

Run the simulation script directly:

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

data/                   # Scripts for generating and managing datasets (not raw data files)
├── florida-doe/        # Florida DOE BEST curriculum: skill space CSV and data-cleaning scripts
├── eedi_tutoring/      # Pipeline for tagging and reviewing real Eedi tutoring conversations
├── simulated_datasets/ # Script for simulating learner-tutor conversation datasets with FlexLearner
└── data_utils/         # Shared helpers (e.g. adding mock tutor/learner responses)

examples/               # End-to-end usage examples
├── base_learner/       # BaseLearner implementations (binary-skill, conversation-history)
├── flexlearner/        # FlexLearner implementations (binary, history, knowledge-graph)
├── evaluations/        # Evaluation scripts for all four benchmark families
├── paper_results/      # Scripts and notebook to reproduce the paper's evaluation results
└── learner_utils/      # Utilities: learner pool creation, manual tutor session

```

## Reproducing paper results

`examples/paper_results/` contains two files that reproduce the evaluation results from the paper.

**`eedi_fitted_learner_evals.py`** — runs the full evaluation suite on the Eedi dataset-fitted benchmark, sweeping all combinations of learner type (`BinarySkillLearner`, `ConversationHistoryLearner`) and model pair defined in `_MODEL_COMBINATIONS`. Set `EEDI_SAMPLED_CONVERSATIONS_PATH` in your `.env` to point to your Eedi conversations JSONL file, then:

```bash
python examples/paper_results/eedi_fitted_learner_evals.py
```

Outputs land in `outputs/dataset_fitted_evals/<run_label>__<timestamp>/`, one subdirectory per configuration.

**`analyze_evals_results.ipynb`** — notebook for analyzing the output of the eval script. Point `EVAL_OUTPUT_DIR` at your output directory and run the cells to get results tables, score plots, LaTeX tables (Tables 1 and 2 from the paper), per-scenario breakdowns, a conversation browser, and distribution plots.

### Claude model support

`src/evalconvolearn/utils/llm_client.py` exposes a `make_client(model)` helper that returns an OpenAI-compatible client for both OpenAI and Claude models. Pass a Claude model name (e.g. `"claude-sonnet-4-6"`) and the client automatically routes requests to the Anthropic API using `ANTHROPIC_API_KEY`. This is used by the tutor and evaluator components — set `ANTHROPIC_API_KEY` in your `.env` when running with Claude models.

## Running tests

The test suite uses [pytest](https://docs.pytest.org/). Install the dev dependencies first:

```bash
# with uv (recommended):
uv sync --group dev
# or with pip:
pip install -e ".[dev]"
```

**Unit tests** (no API key required):

```bash
pytest tests/unit
```

**Integration tests** (require `OPENAI_API_KEY`):

```bash
pytest tests/integration
```

**All tests:**

```bash
pytest
```

Useful flags:
- Skip slow tests: `pytest -m "not slow"`
- Skip integration tests: `pytest -m "not integration"`
- Run with coverage: `pytest --cov=src/evalconvolearn --cov-report=term-missing`

## Contributors

- [Baptiste Moreau-Pernet](mailto:baptiste@levi.digitalharbor.org)
- [AJ Strauman-Scott](mailto:aj.scott@renphil.org)
