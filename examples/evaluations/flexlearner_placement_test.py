"""Evaluate FlexLearner implementations on the PlacementTestBenchmark.

Tests three learner implementations:
  - BinarySkillsFlexLearner (default, binary skill mastery)
  - ConversationHistoryLearner (natural-language knowledge items)
  - KnowledgeGraphLearner (property graph + vector store)

The benchmark verifies that learners answer items correctly when they master
the required skills and incorrectly otherwise.

Run from the project root:
    python examples/evaluations/flexlearner_placement_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flexlearner.flexlearner_conversation_history import (  # noqa: E402
    ConversationHistoryLearner,
)
from flexlearner.flexlearner_knowledge_graph import KnowledgeGraphLearner  # noqa: E402

from evalconvolearn import BinarySkillsFlexLearner, EvalConvoLearn  # noqa: E402
from evalconvolearn.models.evaluation import (  # noqa: E402
    EvaluationConfig,
    LearnerEvalConfig,
)
from evalconvolearn.utils.benchmark_results import print_placement_results  # noqa: E402

# KG seed triplets for MA.6.NSO.1.1 and MA.6.NSO.1.2
_KG_SEED_TRIPLETS: list[dict] = [
    {
        "entity1": "multi-digit decimal multiplication",
        "entity1_label": "operation",
        "relation": "applying to",
        "entity2": "standard algorithm",
        "entity2_label": "concept",
    },
    {
        "entity1": "multi-digit decimal division",
        "entity1_label": "operation",
        "relation": "applying to",
        "entity2": "long division algorithm",
        "entity2_label": "concept",
    },
    {
        "entity1": "decimal place count in factors",
        "entity1_label": "concept",
        "relation": "defining the meaning",
        "entity2": "decimal place count in product",
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
]  # illustrate the seed triplets for the knowledge graph initialization


def main() -> None:
    """Run PlacementTestBenchmark for all three FlexLearner implementations."""
    sdk = EvalConvoLearn()

    data_root = Path("data") / "florida-doe"
    skill_space = sdk.load_skill_space(data_root / "skill-space.csv")
    items = sdk.load_practice_items(
        data_root / "tagged-practice-items-with-responses.csv",
        skill_space,
    )

    eval_config = EvaluationConfig(
        output_dir=Path("outputs/flexlearner/placement_test"),
        learner_configs=[
            LearnerEvalConfig(
                learner_class=BinarySkillsFlexLearner,
                label="binary_skills_flexlearner",
                mastered_skills=["MA.6.NSO.1.1", "MA.6.NSO.1.2"],
                benchmarks=["PlacementTestBenchmark"],
            ),
            LearnerEvalConfig(
                learner_class=ConversationHistoryLearner,
                label="conv_history_flexlearner",
                mastered_skills=["MA.6.NSO.1.1", "MA.6.NSO.1.2"],
                benchmarks=["PlacementTestBenchmark"],
            ),
            LearnerEvalConfig(
                learner_class=KnowledgeGraphLearner,
                label="kg_flexlearner",
                mastered_skills=[
                    "MA.6.NSO.2.1",
                    "MA.6.NSO.2.2",
                ],  # align with KG seed triplets above
                benchmarks=["PlacementTestBenchmark"],
                init_knowledge_kwargs={
                    "initial_triplets": _KG_SEED_TRIPLETS,  # illustrate passing the initial KG triplets directly
                },
            ),
        ],
        benchmarks=["PlacementTestBenchmark"],
        runs_per_scenario=1,
        label="flexlearner_placement_test",
    )

    results = sdk.run_evaluation(
        eval_config=eval_config,
        skill_space=skill_space,
        practice_item_pool=items,
    )
    results.print_summary()

    for bench_summary in results.summaries:
        if bench_summary.benchmark_name != "PlacementTestBenchmark":
            continue
        output_file_str = (bench_summary.output or {}).get("output_file", "")
        if output_file_str:
            print_placement_results(
                Path(output_file_str),
                bench_summary.learner_config_label,
            )


if __name__ == "__main__":
    main()
