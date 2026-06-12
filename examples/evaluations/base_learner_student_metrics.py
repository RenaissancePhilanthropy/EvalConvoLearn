"""Evaluate a base learner on the DatasetFittedConversationalBenchmark.

Minimal example: BinarySkillLearner with gpt-4.1-mini and 3 few-shot tutor examples.
For the full paper comparison (both learner types, models, few-shot counts) see
examples/paper_results/eedi_fitted_learner_evals.py.

Run from the project root:
    python examples/evaluations/base_learner_student_metrics.py
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base_learner.binary_skill_learner import BinarySkillLearner

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
    os.getenv("EEDI_SAMPLED_CONVERSATIONS_PATH", "path_to_your_saved_eedi_dataset"),
)
_OUTPUT_DIR = Path("outputs/dataset_fitted_evals/binary_skills_learner")
_RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    model = "gpt-4.1-mini"
    few_shot_count = 3

    run_label = f"binary_skills__gpt-4_1-mini__fs{few_shot_count}"
    eval_dir = _OUTPUT_DIR / f"{run_label}__{run_id}"

    eval_config = EvaluationConfig(
        label=f"Dataset-fitted eval — {run_label}",
        output_dir=eval_dir,
        learner_configs=[
            LearnerEvalConfig(
                learner_class=BinarySkillLearner,
                label=f"{run_label}__{run_id}",
                mastered_skills=[],
                init_knowledge_kwargs={
                    "knowledge_cache_dir": str(eval_dir / "knowledge_cache"),
                    "model": model,
                    "num_few_shot_examples": few_shot_count,
                    "conversations_jsonl_path": str(_CONVERSATIONS_JSONL),
                },
                benchmarks=["DatasetFittedConversationalBenchmark"],
            ),
        ],
        benchmarks=["DatasetFittedConversationalBenchmark"],
        benchmarks_custom_args={
            "DatasetFittedConversationalBenchmark": {
                "conversations_jsonl_path": _CONVERSATIONS_JSONL,
                "conversation_metrics_cache_path": eval_dir
                / "conversation_metrics_cache.json",
                "max_records_per_skill_mastery": 8,
                "max_conversation_turns": 7,
                "num_example_conversations_for_tutor_response_generation": few_shot_count,
                "classification_model": model,
                "random_seed": _RANDOM_SEED,
                "runs": 2,
                "use_capped_dialogues": True,
                "max_skills": 3,
            },
        },
    )

    sdk = EvalConvoLearn()
    skill_space = sdk.load_skill_space()
    practice_item_pool = sdk.load_oversampled_items(skill_space)

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


if __name__ == "__main__":
    main()
