"""Evaluate base learners on the BaselineMultiConversationsBenchmark.

Tests both BinarySkillLearner and ConversationHistoryLearner on multi-turn
upskilling sequences: the learner is expected to climb a skill tree through
a series of practice conversations and consolidate mastery at each level.

Run from the project root:
    python examples/evaluations/base_learner_multi_conv.py
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

_OUTPUT_DIR = Path("outputs/base_learner/multi_conv")


def main() -> None:
    sdk = EvalConvoLearn()
    skill_space = sdk.load_skill_space()
    items = sdk.load_practice_items(skill_space)

    timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    output_dir = _OUTPUT_DIR / timestamp

    eval_config = EvaluationConfig(
        label="BaseLearner — multi-conversation benchmark",
        output_dir=output_dir,
        learner_configs=[
            LearnerEvalConfig(
                learner_class=BinarySkillLearner,
                label=f"binary_skill_{timestamp}",
                mastered_skills=["MA.6.NSO.2.3"],
                benchmarks=["BaselineMultiConversationsBenchmark"],
            ),
            LearnerEvalConfig(
                learner_class=ConversationHistoryLearner,
                label=f"conv_history_{timestamp}",
                mastered_skills=["MA.6.NSO.2.3"],
                init_knowledge_kwargs={
                    "knowledge_cache_dir": str(output_dir / "knowledge_cache"),
                },
                benchmarks=["BaselineMultiConversationsBenchmark"],
            ),
        ],
        benchmarks=["BaselineMultiConversationsBenchmark"],
        benchmarks_custom_args={
            "BaselineMultiConversationsBenchmark": {
                "consolidation_runs": 3,
                "max_conversation_turns": 6,
                "max_climb_items_per_skill": 3,
            },
        },
    )

    results = sdk.run_base_learner_evaluation(eval_config, skill_space, items)
    results.print_summary()


if __name__ == "__main__":
    main()
