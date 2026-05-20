"""Evaluate FlexLearner implementations on the LearningFromConversationBenchmark.

Tests three learner implementations:
  - BinarySkillsFlexLearner (default, binary skill mastery)
  - ConversationHistoryLearner (natural-language knowledge items)
  - KnowledgeGraphLearner (property graph + vector store)

The benchmark runs a tutoring conversation on a target skill and checks whether
the learner's knowledge representation reflects the learning that took place.

Run from the project root:
    python examples/evaluations/flexlearner_learning_from_conversation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flexlearner.flexlearner_conversation_history import ConversationHistoryLearner
from flexlearner.flexlearner_knowledge_graph import (
    KnowledgeGraphLearner,
    build_initial_kg_snapshot,
)

from evalconvolearn import BinarySkillsFlexLearner, EvalConvoLearn
from evalconvolearn.models.evaluation import EvaluationConfig, LearnerEvalConfig
from evalconvolearn.utils.benchmark_results import print_lfc_results

_KG_SEED_TRIPLETS: list[dict] = [
    {
        "entity1": "multi-digit decimal multiplication",
        "entity1_label": "operation",
        "relation": "applying to",
        "entity2": "standard algorithm",
        "entity2_label": "concept",
    },
    {
        "entity1": "fraction division",
        "entity1_label": "operation",
        "relation": "applying to",
        "entity2": "multiply by reciprocal",
        "entity2_label": "concept",
    },
    {
        "entity1": "mixed number division",
        "entity1_label": "operation",
        "relation": "suggesting sequential relationship",
        "entity2": "convert to improper fraction first",
        "entity2_label": "concept",
    },
]


def main() -> None:
    sdk = EvalConvoLearn()

    data_root = Path("data") / "florida-doe"
    skill_space = sdk.load_skill_space(data_root / "skill-space.csv")
    items = sdk.load_practice_items(
        data_root / "tagged-practice-items-with-responses.csv", skill_space
    )

    print("Pre-building initial KG snapshot...")
    kg_initial_state = build_initial_kg_snapshot(_KG_SEED_TRIPLETS)

    eval_config = EvaluationConfig(
        learner_configs=[
            LearnerEvalConfig(
                learner_class=BinarySkillsFlexLearner,
                label="binary_skills_flexlearner",
                mastered_skills=["MA.6.NSO.1.1", "MA.6.NSO.1.2"],
                benchmarks=["LearningFromConversationBenchmark"],
            ),
            LearnerEvalConfig(
                learner_class=ConversationHistoryLearner,
                label="conv_history_flexlearner",
                mastered_skills=["MA.6.NSO.1.1", "MA.6.NSO.1.2"],
                benchmarks=["LearningFromConversationBenchmark"],
            ),
            LearnerEvalConfig(
                learner_class=KnowledgeGraphLearner,
                label="kg_flexlearner",
                mastered_skills=["MA.6.NSO.2.1", "MA.6.NSO.2.2"], # use the skills defined in the knowledge graph above
                benchmarks=["LearningFromConversationBenchmark"],
                init_knowledge_kwargs={"prebuilt_kg_state": kg_initial_state},
            ),
        ],
        benchmarks=["LearningFromConversationBenchmark"],
        runs_per_scenario=1,
        label="flexlearner_learning_from_conversation",
        benchmarks_custom_args={
            "LearningFromConversationBenchmark": {
                "evaluate_learning_with_pre_post_tests": False,
            },
        },
    )

    results = sdk.run_evaluation(
        eval_config=eval_config, skill_space=skill_space, practice_item_pool=items
    )

    results.print_summary()

    for bench_summary in results.summaries:
        if bench_summary.benchmark_name != "LearningFromConversationBenchmark":
            continue
        output_file_str = (bench_summary.output or {}).get("output_file", "")
        if output_file_str:
            print_lfc_results(
                output_file=Path(output_file_str),
                label=bench_summary.learner_config_label,
            )


if __name__ == "__main__":
    main()
