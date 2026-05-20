"""Evaluate base learners on the BaseLinePlacementTestBenchmark.

Tests both BinarySkillLearner and ConversationHistoryLearner.

Run from the project root:
    python examples/evaluations/base_learner_placement_test.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base_learner.binary_skill_learner import BinarySkillLearner
from base_learner.conversation_history_learner import ConversationHistoryLearner

from evalconvolearn import EvalConvoLearn, EvaluationConfig, LearnerEvalConfig

_OUTPUT_DIR = Path("outputs/base_learner/placement_test")


def main() -> None:
    sdk = EvalConvoLearn()
    skill_space = sdk.load_skill_space(Path("data/florida-doe/skill-space.csv"))
    items = sdk.load_practice_items(
        Path("data/florida-doe/tagged-practice-items-with-responses.csv"),
        skill_space,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    output_dir = _OUTPUT_DIR / timestamp

    eval_config = EvaluationConfig(
        label="BaseLearner — placement test",
        output_dir=output_dir,
        learner_configs=[
            LearnerEvalConfig(
                learner_class=BinarySkillLearner,
                label=f"binary_skill_{timestamp}",
                mastered_skills=["MA.6.NSO.2.1"],
                benchmarks=["BaseLinePlacementTestBenchmark"],
            ),
            LearnerEvalConfig(
                learner_class=ConversationHistoryLearner,
                label=f"conv_history_{timestamp}",
                mastered_skills=["MA.6.NSO.2.1"],
                init_knowledge_kwargs={
                    "knowledge_cache_dir": str(output_dir / "knowledge_cache"),
                },
                benchmarks=["BaseLinePlacementTestBenchmark"],
            ),
        ],
        benchmarks=["BaseLinePlacementTestBenchmark"],
        benchmarks_custom_args={
            "BaseLinePlacementTestBenchmark": {
                "max_items": 5,
                "runs": 2,
            },
        },
    )

    results = sdk.run_base_learner_evaluation(eval_config, skill_space, items)
    results.print_summary()


if __name__ == "__main__":
    main()
