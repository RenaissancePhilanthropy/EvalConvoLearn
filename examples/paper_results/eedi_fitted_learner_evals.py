"""Full evaluation suite for the Eedi dataset-fitted conversational benchmark.

Runs all combinations of:
  - Learner type  : BinarySkillLearner, ConversationHistoryLearner
  - (model_learner, model_tutor_evals) pairs defined in _MODEL_COMBINATIONS below
  - Tutor few-shot: configurable per run

model_learner controls the learner's internal LLM calls (response generation,
knowledge classification, skill tagging).
model_tutor_evals controls the tutor responses (both during evaluation and initialization),
and all LLM-as-judge evaluations (error/talk-move tagging).

Run from the project root:
    python examples/paper_results/eedi_fitted_learner_evals.py
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base_learner.binary_skill_learner import BinarySkillLearner  # noqa: E402
from base_learner.conversation_history_learner import ConversationHistoryLearner  # noqa: E402

from evalconvolearn import EvalConvoLearn, EvaluationConfig, LearnerEvalConfig  # noqa: E402
from evalconvolearn.models.practice_item import PracticeItemPool  # noqa: E402
from evalconvolearn.models.skill import SkillSpace  # noqa: E402

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# --------------------------------------------------------------------------- #
#  Paths — update these to point to your dataset
# --------------------------------------------------------------------------- #

_CONVERSATIONS_JSONL = Path(
    os.getenv("EEDI_SAMPLED_CONVERSATIONS_PATH", "path_to_your_saved_eedi_dataset"),
)
_OUTPUT_BASE = Path("outputs/dataset_fitted_evals")
_RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
#  Evaluation parameters
# --------------------------------------------------------------------------- #

_LEARNER_VARIANTS = [
    (BinarySkillLearner, "binary_skills", _OUTPUT_BASE / "binary_skills_learner"),
    (
        ConversationHistoryLearner,
        "conv_history",
        _OUTPUT_BASE / "conversation_history_learner",
    ),
]

# Each entry: (model_learner, model_tutor_evals, few_shot_count)
#   model_learner — learner's internal LLM (response generation, classification)
#   model_tutor_evals — tutor responses + LLM-as-judge evals (error/talk-move tagging)
_MODEL_COMBINATIONS: list[tuple[str, str, int]] = [
    ("gpt-4.1-mini", "claude-sonnet-4-6", 3),
    # ("gpt-4.1-mini", "claude-sonnet-4-6", 0),
]

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _build_eval_config(
    *,
    learner_class: type,
    learner_short_name: str,
    base_eval_dir: Path,
    model_learner: str,
    model_tutor_evals: str,
    few_shot_count: int,
    conversations_jsonl_path: Path,
    run_id: str,
) -> EvaluationConfig:
    m1_tag = model_learner.replace(".", "_")
    m2_tag = model_tutor_evals.replace(".", "_")
    few_shot_tag = f"fs{few_shot_count}"
    run_label = f"{learner_short_name}__m1_{m1_tag}__m2_{m2_tag}__{few_shot_tag}"
    eval_dir = base_eval_dir / f"{run_label}__{run_id}"

    return EvaluationConfig(
        label=f"Dataset-fitted eval — {run_label}",
        output_dir=eval_dir,
        learner_configs=[
            LearnerEvalConfig(
                learner_class=learner_class,
                label=f"{run_label}__{run_id}",
                mastered_skills=[],
                init_knowledge_kwargs={
                    "knowledge_cache_dir": str(eval_dir / "knowledge_cache"),
                    "model": model_learner,
                    "tutor_model": model_tutor_evals,
                    "num_few_shot_examples": few_shot_count,
                    "conversations_jsonl_path": str(conversations_jsonl_path),
                },
                benchmarks=["DatasetFittedConversationalBenchmark"],
            ),
        ],
        benchmarks=["DatasetFittedConversationalBenchmark"],
        benchmarks_custom_args={
            "DatasetFittedConversationalBenchmark": {
                "conversations_jsonl_path": conversations_jsonl_path,
                "conversation_metrics_cache_path": eval_dir / "conversation_metrics_cache.json",
                "max_records_per_skill_mastery": 8,
                "max_conversation_turns": 7,
                "num_example_conversations_for_tutor_response_generation": few_shot_count,
                "tutor_model": model_tutor_evals,
                "classification_model": model_tutor_evals,
                "random_seed": _RANDOM_SEED,
                "runs": 3,
                "use_capped_dialogues": True,
                "max_skills": 3,
            },
        },
    )


def _run_eval(
    *,
    sdk: EvalConvoLearn,
    skill_space: SkillSpace,
    practice_item_pool: PracticeItemPool,
    eval_config: EvaluationConfig,
    run_label: str,
) -> None:
    print("=" * 70)
    print(f"Running: {run_label}")
    results = sdk.run_base_learner_evaluation(
        eval_config,
        skill_space,
        practice_item_pool,
    )
    print(f"Run directory : {results.run_dir}")
    print(f"All passed    : {results.all_passed}")
    for summary in results.summaries:
        output_file = summary.output.get("output_file")
        print(
            f"  {summary.benchmark_name}: passed={summary.passed}; output={output_file}",
        )


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #


def main(run_id: str | None = None) -> None:
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H-%M-%S")

    sdk = EvalConvoLearn()
    skill_space = sdk.load_skill_space()
    practice_item_pool = sdk.load_oversampled_items(skill_space)

    for (learner_class, learner_short_name, base_eval_dir), (
        model_learner,
        model_tutor_evals,
        few_shot_count,
    ) in product(
        _LEARNER_VARIANTS,
        _MODEL_COMBINATIONS,
    ):
        m1_tag = model_learner.replace(".", "_")
        m2_tag = model_tutor_evals.replace(".", "_")
        run_label = f"{learner_short_name}__m1_{m1_tag}__m2_{m2_tag}__fs{few_shot_count}"

        eval_config = _build_eval_config(
            learner_class=learner_class,
            learner_short_name=learner_short_name,
            base_eval_dir=base_eval_dir,
            model_learner=model_learner,
            model_tutor_evals=model_tutor_evals,
            few_shot_count=few_shot_count,
            conversations_jsonl_path=_CONVERSATIONS_JSONL,
            run_id=run_id,
        )
        _run_eval(
            sdk=sdk,
            skill_space=skill_space,
            practice_item_pool=practice_item_pool,
            eval_config=eval_config,
            run_label=run_label,
        )


if __name__ == "__main__":
    main()
