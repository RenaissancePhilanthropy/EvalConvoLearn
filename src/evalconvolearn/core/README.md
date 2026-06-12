# `core/` — Abstractions and SDK entry point

This package contains the base abstractions and the main SDK class.

## `sdk.py` — `EvalConvoLearn`

The main entry point for the library. Provides high-level methods to:

- Load a `SkillSpace` and `PracticeItemPool` from CSV/JSON files.
- Create and load `StudentPool` instances.
- Create conversation sessions for any learner type.
- Run evaluations (`run_evaluation`, `run_base_learner_evaluation`) and aggregate results (`aggregate_results`).

## `config.py` — `EvalConvoLearnConfig`

Pydantic settings model that controls global defaults (data directories, conversation turn limits, evaluation output paths). Pass an instance to `EvalConvoLearn(config=...)` to override defaults.

The following environment variables are read automatically via `pydantic-settings`:

| Variable | Config field | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required by LLM-based grading and tutor-response generation |
| `SKILL_SPACE_PATH` | `skill_space_path` | Default path for `sdk.load_skill_space()` |
| `TAGGED_PRACTICE_ITEMS_WITH_RESPONSES_CSV` | `tagged_practice_items_with_responses_csv` | Default path for `sdk.load_practice_items()` |
| `OVERSAMPLED_ITEMS_CSV` | `oversampled_items_csv` | Default path for `sdk.load_oversampled_items()` |

Copy `.env-example` at the repo root to `.env` and fill in the values before running any scripts or benchmarks.

## `base_learner.py` — `BaseLearner`

Abstract base class for **black-box** learner simulations. Subclass this when your learner has its own internal knowledge representation and you want the framework to evaluate it without controlling its prompts.

Key abstract methods to implement:
- `start_or_continue_conversation` — respond to a tutor message and signal whether the conversation is done.
- `end_conversation` — finalize the session and allow the learner to update its knowledge.

Concrete helpers provided:
- `has_skill` — probe skill mastery by running assessment problems.
- `initialize_from_skills` / `upskill_learner_to_skills` — bring the learner to a target knowledge state via tutored conversations.
- `save_practice_conversation` / `load_practice_conversations` — persist and retrieve conversation history.

## `flexlearner.py` — `FlexLearner`

> **Note:** `FlexLearner` currently supports **OpenAI models only**. The internal LLM calls in `learns_from_conversation` and `answer_practice_item` hardcode `gpt-4.1-mini` and use an `OpenAI()` client directly. Claude model support requires threading a `model` parameter through these methods and routing via `utils/llm_client.py`.

Abstract extension of `BaseLearner` for **transparent, skill-guardrail-based** simulations. The learner's true mastery state is always tracked as a list of skill IDs; the *visible* knowledge representation (what gets injected into prompts) is left to subclasses.

Key abstract methods to implement:
- `get_knowledge_description` / `get_knowledge_for_problem` / `get_required_knowledge_to_answer_practice_item` — expose the learner's knowledge in prompt-ready form.
- `update_knowledge_from_conversation` — update the internal representation after a tutored session.
- `initialize_learner_knowledge` — set up the internal representation from the initial mastered-skill list.

Concrete helpers provided:
- `learns_from_conversation` — LLM-based skill mastery update with guardrail enforcement.
- `answer_practice_item` — generate a placement-test answer conditioned on mastery state.
- `get_practice_prompt` / `get_solution_prompt` — default prompt builders (override for custom styles).
- `master_new_skill` / `can_learn_skill` / `get_learnable_skills` — skill-graph guardrail utilities.

## `base_tutor.py` — `BaseTutor`

Abstract base class for tutors used in evaluation conversations. Implement `generate_response(dialogue_history, **kwargs) -> TutorResponse` to plug in any tutor (LLM-based, rule-based, human-in-the-loop, etc.).

Also contains two standalone helpers for working with tutoring conversation datasets:
- `load_effective_conversations` — load JSONL records where the learner demonstrably learned.
- `format_conversation_as_few_shot` — render a conversation record as a few-shot example block.
