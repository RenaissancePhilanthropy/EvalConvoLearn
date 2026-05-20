"""Evaluate base learners on the DatasetFittedConversationalBenchmark.

Runs all combinations of:
  - Learner type  : BinarySkillLearner, ConversationHistoryLearner
  - Model         : gpt-4.1-mini
  - Tutor few-shot: 3 examples, 0 examples

The model controls ALL LLM calls inside the learner (response generation,
end-of-conversation classification, knowledge update) as well as the
benchmark's conversation-behavior classification.

Run from the project root:
    python examples/evaluations/base_learner_student_metrics.py
"""

from __future__ import annotations

import logging
import random
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base_learner.binary_skill_learner import BinarySkillLearner
from base_learner.conversation_history_learner import ConversationHistoryLearner

from evalconvolearn import EvalConvoLearn, EvaluationConfig, LearnerEvalConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# --------------------------------------------------------------------------- #
#  Paths — update these to point to your dataset
# --------------------------------------------------------------------------- #

_CONVERSATIONS_JSONL = Path(
    "data/evaluations/source_data/tutoring_conversations.jsonl",
)
_PRACTICE_ITEMS_CSV = Path(
    "data/florida-doe/oversampled_items/oversampled-items-x10.csv",
)
_OUTPUT_BASE = Path("outputs/dataset_fitted_evals")

# --------------------------------------------------------------------------- #
#  Evaluation parameters
# --------------------------------------------------------------------------- #

_LEARNER_VARIANTS = [
    (BinarySkillLearner, "binary_skills", _OUTPUT_BASE / "binary_skills_learner"),
    (ConversationHistoryLearner, "conv_history", _OUTPUT_BASE / "conversation_history_learner"),
]

_MODELS = ["gpt-4.1-mini"]
_FEW_SHOT_COUNTS = [3, 0]


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def _build_eval_config(
    *,
    learner_class,
    learner_short_name: str,
    base_eval_dir: Path,
    model: str,
    few_shot_count: int,
    conversations_jsonl_path: Path,
    run_id: str,
) -> EvaluationConfig:
    model_tag = model.replace(".", "_")
    few_shot_tag = f"fs{few_shot_count}"
    run_label = f"{learner_short_name}__{model_tag}__{few_shot_tag}"
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
                    "model": model,
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
                "classification_model": model,
                "random_seed": random.randint(0, 1000),
                "runs": 2,
                "use_capped_dialogues": True,
                "max_skills": 3,
            },
        },
    )


def _run_eval(
    *,
    sdk: EvalConvoLearn,
    skill_space,
    practice_item_pool,
    eval_config: EvaluationConfig,
    run_label: str,
) -> None:
    print("=" * 70)
    print(f"Running: {run_label}")
    results = sdk.run_base_learner_evaluation(eval_config, skill_space, practice_item_pool)
    print(f"Run directory : {results.run_dir}")
    print(f"All passed    : {results.all_passed}")
    for summary in results.summaries:
        output_file = summary.output.get("output_file")
        print(f"  {summary.benchmark_name}: passed={summary.passed}; output={output_file}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #


def main(run_id: str | None = None) -> None:
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H-%M-%S")

    sdk = EvalConvoLearn()
    skill_space = sdk.load_skill_space(Path("data/florida-doe/skill-space.csv"))
    practice_item_pool = sdk.load_practice_items(_PRACTICE_ITEMS_CSV, skill_space)

    for (learner_class, learner_short_name, base_eval_dir), model, few_shot_count in product(
        _LEARNER_VARIANTS, _MODELS, _FEW_SHOT_COUNTS
    ):
        model_tag = model.replace(".", "_")
        run_label = f"{learner_short_name}__{model_tag}__fs{few_shot_count}"

        eval_config = _build_eval_config(
            learner_class=learner_class,
            learner_short_name=learner_short_name,
            base_eval_dir=base_eval_dir,
            model=model,
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
