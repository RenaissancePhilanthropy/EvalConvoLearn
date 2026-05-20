"""Evaluate FlexLearner implementations on the MultiConversationsPracticeBenchmark.

Tests three learner implementations:
  - BinarySkillsFlexLearner (default, binary skill mastery)
  - ConversationHistoryLearner (natural-language knowledge items)
  - KnowledgeGraphLearner (property graph + vector store)

The benchmark runs a full skill-tree climb: the learner practices prerequisite
skills and consolidates mastery at each level via multiple conversations.

Run from the project root:
    python examples/evaluations/flexlearner_multi_conv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flexlearner.flexlearner_conversation_history import ConversationHistoryLearner  # noqa: E402
from flexlearner.flexlearner_knowledge_graph import (  # noqa: E402
    KnowledgeGraphLearner,
    build_initial_kg_snapshot,
)

from evalconvolearn import BinarySkillsFlexLearner, EvalConvoLearn  # noqa: E402
from evalconvolearn.models.evaluation import EvaluationConfig, LearnerEvalConfig  # noqa: E402
from evalconvolearn.utils.benchmark_results import print_mcp_results  # noqa: E402

# Seed triplets for MA.6.NSO.2.1 (decimals) and MA.6.NSO.2.2 (fractions)
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
]

# illustrate the Knowledge Graph initialization using the skill_id_to_triplets mapping
_SKILL_ID_TO_TRIPLETS: dict[str, list[dict]] = {
    "MA.6.NSO.2.1": _KG_SEED_TRIPLETS[:3],
    "MA.6.NSO.2.2": _KG_SEED_TRIPLETS[3:],
}


def main() -> None:
    """Run MultiConversationsPracticeBenchmark for all three FlexLearner implementations."""
    sdk = EvalConvoLearn()

    data_root = Path("data") / "florida-doe"
    skill_space = sdk.load_skill_space(data_root / "skill-space.csv")
    items = sdk.load_practice_items(
        data_root / "tagged-practice-items-with-responses.csv", skill_space,
    )
    oversampled_items = sdk.load_practice_items(
        data_root / "oversampled_items" / "oversampled-items-x10.csv", skill_space,
    )

    eval_config = EvaluationConfig(
        learner_configs=[
            LearnerEvalConfig(
                learner_class=BinarySkillsFlexLearner,
                label="binary_skills_flexlearner",
                mastered_skills=["MA.6.NSO.1.1", "MA.6.NSO.1.2"],
                benchmarks=["MultiConversationsPracticeBenchmark"],
            ),
            LearnerEvalConfig(
                learner_class=ConversationHistoryLearner,
                label="conv_history_flexlearner",
                mastered_skills=["MA.6.NSO.1.1", "MA.6.NSO.1.2"],
                benchmarks=["MultiConversationsPracticeBenchmark"],
            ),
            LearnerEvalConfig(
                learner_class=KnowledgeGraphLearner,
                label="kg_flexlearner",
                mastered_skills=["MA.6.NSO.2.1", "MA.6.NSO.2.2"],  # align with KG triplets above
                benchmarks=["MultiConversationsPracticeBenchmark"],
                init_knowledge_kwargs={
                    "skill_id_to_triplets": _SKILL_ID_TO_TRIPLETS,  # selects relevant triplets at init time
                },
            ),
        ],
        benchmarks=["MultiConversationsPracticeBenchmark"],
        runs_per_scenario=1,
        label="flexlearner_multi_conv",
        benchmarks_custom_args={
            "MultiConversationsPracticeBenchmark": {
                "oversampled_item_pool": oversampled_items,
            },
        },
    )

    results = sdk.run_evaluation(
        eval_config=eval_config, skill_space=skill_space, practice_item_pool=items,
    )
    results.print_summary()

    for bench_summary in results.summaries:
        if bench_summary.benchmark_name != "MultiConversationsPracticeBenchmark":
            continue
        output_file_str = (bench_summary.output or {}).get("output_file", "")
        if output_file_str:
            print_mcp_results(Path(output_file_str), bench_summary.learner_config_label)


if __name__ == "__main__":
    main()
