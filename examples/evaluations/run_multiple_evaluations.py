"""Unified evaluation runner — runs all benchmark x learner combinations and
produces an aggregated summary JSON in ``outputs/``.

The script orchestrates (in order):

1. **FlexLearner benchmarks** (``LearningFromConversationBenchmark``,
   ``MultiConversationsPracticeBenchmark``) with:
   - The default skill-binary ``Learner`` (binary-skill-skills)
   - The ``KnowledgeGraphLearner`` (KG-backed)

2. **Base-learner benchmarks** (``BaseLineLearningFromConversationBenchmark``,
   ``BaselineMultiConversationsBenchmark``) with the
   ``ConversationHistoryLearner`` implementation.

All results are gathered into a single summary JSON via ``sdk.aggregate_results``
with per-benchmark x per-learner-type alignment metrics and pointers to
individual result dirs.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so local imports resolve
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from base_learner.conversation_history_learner import ConversationHistoryLearner  # noqa: E402
from flexlearner.flexlearner_knowledge_graph import (  # noqa: E402
    KnowledgeGraphLearner,
    build_initial_kg_snapshot,
)

from evalconvolearn import EvalConvoLearn, EvaluationResults  # noqa: E402
from evalconvolearn.models.binary_skills_flexlearner import BinarySkillsFlexLearner  # noqa: E402
from evalconvolearn.models.evaluation import EvaluationConfig, LearnerEvalConfig  # noqa: E402

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = PROJECT_ROOT / "data" / "florida-doe"
MULTI_LEARNER_EVALS_DIR = PROJECT_ROOT / "outputs" / "multi_learner_evals"
BL_KNOWLEDGE_CACHE_DIR = PROJECT_ROOT / "outputs" / "base_learner" / "learning_from_conversation"

RUNS_PER_SCENARIO = 2
MAX_ITEMS = 3


# ====================================================================== #
#  KG seed triplets
# ====================================================================== #

KG_SEED_TRIPLETS: list[dict] = [
    # MA.6.NSO.2.1 — multi-digit decimal multiply / divide
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
        "entity1": "$20.76 divided by $3.46",
        "entity1_label": "operation",
        "relation": "suggesting sequential relationship",
        "entity2": "6 bags of chips",
        "entity2_label": "whole numbers",
    },
    {
        "entity1": "6 bags",
        "entity1_label": "whole numbers",
        "relation": "applying to",
        "entity2": "3 servings per bag",
        "entity2_label": "whole numbers",
    },
    {
        "entity1": "18 total servings",
        "entity1_label": "whole numbers",
        "relation": "suggesting structural relationship",
        "entity2": "6 bags of chips with 3 servings each",
        "entity2_label": "whole numbers",
    },
    {
        "entity1": "6.75 bags",
        "entity1_label": "fraction",
        "relation": "applying to",
        "entity2": "13.125 ounces per bag",
        "entity2_label": "fraction",
    },
    {
        "entity1": "6.75 multiplied by 13.125",
        "entity1_label": "operation",
        "relation": "suggesting sequential relationship",
        "entity2": "88.59375 ounces total",
        "entity2_label": "fraction",
    },
    {
        "entity1": "decimal place count in factors",
        "entity1_label": "concept",
        "relation": "defining the meaning",
        "entity2": "decimal place count in product",
        "entity2_label": "concept",
    },
    {
        "entity1": "decimal division",
        "entity1_label": "operation",
        "relation": "identifying the property",
        "entity2": "place value alignment in quotient",
        "entity2_label": "concept",
    },
    {
        "entity1": "decimal less than 1 multiplied by another number",
        "entity1_label": "fraction",
        "relation": "differentiating",
        "entity2": "product smaller than original factor",
        "entity2_label": "concept",
    },
    # MA.6.NSO.2.2 — fraction multiply / divide
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
    {
        "entity1": "6 and 2/3 tons in container",
        "entity1_label": "fraction",
        "relation": "applying to",
        "entity2": "1 and 1/2 tons juiced per day",
        "entity2_label": "fraction",
    },
    {
        "entity1": "6 and 2/3 divided by 1 and 1/2",
        "entity1_label": "operation",
        "relation": "suggesting sequential relationship",
        "entity2": "4 and 4/9 days to empty container",
        "entity2_label": "fraction",
    },
    {
        "entity1": "20/3 divided by 3/2",
        "entity1_label": "operation",
        "relation": "applying to",
        "entity2": "20/3 multiplied by 2/3 equals 40/9",
        "entity2_label": "fraction",
    },
    {
        "entity1": "35/5",
        "entity1_label": "fraction",
        "relation": "suggesting sequential relationship",
        "entity2": "simplify to 7 before dividing",
        "entity2_label": "simplify",
    },
    {
        "entity1": "35/5 divided by 8",
        "entity1_label": "operation",
        "relation": "suggesting sequential relationship",
        "entity2": "7/8 as result",
        "entity2_label": "fraction",
    },
    {
        "entity1": "multiplying or dividing fractions",
        "entity1_label": "operation",
        "relation": "differentiating",
        "entity2": "common denominator not required",
        "entity2_label": "concept",
    },
    {
        "entity1": "dividing by a fraction less than 1",
        "entity1_label": "operation",
        "relation": "differentiating",
        "entity2": "quotient larger than dividend",
        "entity2_label": "concept",
    },
    # MA.6.NSO.4.2 — Apply and extend previous understandings of operations with whole numbers to multiply and divide integers with procedural fluency
    {
        "entity1": "integer multiplication and division",
        "entity1_label": "operation",
        "relation": "applying to",
        "entity2": "whole numbers with attention to signs",
        "entity2_label": "concept",
    },
    {
        "entity1": "negatives multiplied or divided",
        "entity1_label": "operation",
        "relation": "defining the meaning",
        "entity2": "positive result when signs match, negative when different",
        "entity2_label": "concept",
    },
    {
        "entity1": "-52 / 4",
        "entity1_label": "operation",
        "relation": "suggesting sequential relationship",
        "entity2": "-13",
        "entity2_label": "integer",
    },
    {
        "entity1": "-4 times -6",
        "entity1_label": "operation",
        "relation": "suggesting sequential relationship",
        "entity2": "first compute absolute value",
        "entity2_label": "integer",
    },
    {
        "entity1": "-2 times -43",
        "entity1_label": "operation",
        "relation": "suggesting sequential relationship",
        "entity2": "after value find the resulting sign",
        "entity2_label": "integer",
    },
    {
        "entity1": "12 / -3",
        "entity1_label": "operation",
        "relation": "suggesting sequential relationship",
        "entity2": "-4",
        "entity2_label": "integer",
    },
    {
        "entity1": "negative times positive is negative",
        "entity1_label": "operation",
        "relation": "applying to",
        "entity2": "-8 times 7",
        "entity2_label": "integer",
    },
]

SKILL_ID_TO_TRIPLETS: dict[str, list[dict]] = {
    "MA.6.NSO.2.1": KG_SEED_TRIPLETS[:10],
    "MA.6.NSO.2.2": KG_SEED_TRIPLETS[10:20],
    "MA.6.NSO.4.2": KG_SEED_TRIPLETS[20:],
}


# ====================================================================== #
#  Config builders
# ====================================================================== #


def build_base_learner_configs(
    timestamp: str,
    bl_items_path: Path,
) -> tuple[EvaluationConfig, ...]:
    """Return EvaluationConfigs for base-learner benchmarks."""
    common_cache_kw = {
        "knowledge_cache_dir": BL_KNOWLEDGE_CACHE_DIR / f"knowledge_cache_{timestamp}",
    }
    lfc_configs = [
        LearnerEvalConfig(
            learner_class=ConversationHistoryLearner,
            label=f"bl_lfc_intermediate_{timestamp}",
            mastered_skills=["MA.6.NSO.4.2"],
            init_knowledge_kwargs=common_cache_kw,
            benchmarks=["BaseLineLearningFromConversationBenchmark"],
        ),
    ]
    lfc_eval = EvaluationConfig(
        learner_configs=lfc_configs,
        benchmarks=["BaseLineLearningFromConversationBenchmark"],
        runs_per_scenario=RUNS_PER_SCENARIO,
        label=f"bl_lfc_{timestamp}",
        benchmarks_custom_args={
            "BaseLineLearningFromConversationBenchmark": {
                "mocked_tutor_responses_csv_path": bl_items_path,
                "runs": RUNS_PER_SCENARIO,
                "max_items": MAX_ITEMS,
            },
        },
    )

    return (lfc_eval,)


def build_flexlearner_configs(
    timestamp: str,
    kg_initial_state: dict,
    oversampled_items: Any,
) -> tuple[EvaluationConfig, ...]:
    """Return EvaluationConfigs for FlexLearner (binary-skill-skills + KG) benchmarks."""
    binary_skill_lfc = LearnerEvalConfig(
        learner_class=BinarySkillsFlexLearner,
        label=f"binary_skill_lfc_intermediate_{timestamp}",
        mastered_skills=["MA.6.NSO.4.2"],
        benchmarks=["LearningFromConversationBenchmark"],
    )
    binary_skill_multi = LearnerEvalConfig(
        learner_class=BinarySkillsFlexLearner,
        label=f"binary_skill_multi_intermediate_{timestamp}",
        mastered_skills=["MA.6.NSO.4.2"],
        benchmarks=["MultiConversationsPracticeBenchmark"],
    )

    kg_init_kw = {
        "prebuilt_kg_state": kg_initial_state,
        "skill_id_to_triplets": SKILL_ID_TO_TRIPLETS,
    }
    kg_lfc = LearnerEvalConfig(
        learner_class=KnowledgeGraphLearner,
        label=f"kg_lfc_intermediate_{timestamp}",
        mastered_skills=["MA.6.NSO.4.2"],
        init_knowledge_kwargs=kg_init_kw,
        benchmarks=["LearningFromConversationBenchmark"],
    )
    kg_multi = LearnerEvalConfig(
        learner_class=KnowledgeGraphLearner,
        label=f"kg_multi_intermediate_{timestamp}",
        mastered_skills=["MA.6.NSO.4.2"],
        init_knowledge_kwargs=kg_init_kw,
        benchmarks=["MultiConversationsPracticeBenchmark"],
    )

    lfc_eval = EvaluationConfig(
        learner_configs=[binary_skill_lfc, kg_lfc],
        benchmarks=["LearningFromConversationBenchmark"],
        runs_per_scenario=RUNS_PER_SCENARIO,
        label=f"fl_lfc_{timestamp}",
        benchmarks_custom_args={
            "LearningFromConversationBenchmark": {
                "evaluate_learning_with_pre_post_tests": True,
                "check_if_should_learn_modes": [(True, "with_skill_check")],
                "max_items": MAX_ITEMS,
            },
        },
    )

    multi_eval = EvaluationConfig(
        learner_configs=[binary_skill_multi, kg_multi],
        benchmarks=["MultiConversationsPracticeBenchmark"],
        runs_per_scenario=RUNS_PER_SCENARIO,
        label=f"fl_multiconv_{timestamp}",
        benchmarks_custom_args={
            "MultiConversationsPracticeBenchmark": {
                "oversampled_item_pool": oversampled_items,
            },
        },
    )

    return lfc_eval, multi_eval


# ====================================================================== #
#  Main
# ====================================================================== #


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
    evalset_label = f"evalset_{timestamp}"

    evalset_dir = MULTI_LEARNER_EVALS_DIR / evalset_label
    evalset_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 64)
    logger.info("  Evalset label : %s", evalset_label)
    logger.info("  Output dir    : %s", evalset_dir)
    logger.info("=" * 64)

    sdk = EvalConvoLearn()

    # ── Load shared data ─────────────────────────────────────────────── #
    skill_space = sdk.load_skill_space()
    logger.info("Skill space loaded: %d skills", len(skill_space.skills))

    fl_items = sdk.load_practice_items(
        skill_space,
        DATA_ROOT / "tagged-practice-items-with-answers-and-incorrect.csv",
    )
    logger.info("FlexLearner practice items: %d", len(fl_items.items))

    oversampled_items = sdk.load_oversampled_items(skill_space)
    logger.info("Oversampled items: %d", len(oversampled_items.items))

    bl_items = sdk.load_practice_items(skill_space)
    logger.info("Base-learner practice items: %d", len(bl_items.items))

    all_results: list[EvaluationResults] = []
    all_eval_configs: list[EvaluationConfig] = []

    # ── 1. FlexLearner benchmarks ────────────────────────────────────── #
    logger.info("  PHASE 1: FlexLearner benchmarks")
    kg_initial_state = build_initial_kg_snapshot(KG_SEED_TRIPLETS)
    for ec in build_flexlearner_configs(timestamp, kg_initial_state, oversampled_items):
        ec.output_dir = evalset_dir / "flexlearner"
        logger.info("Running: %s", ec.label)
        try:
            all_results.append(sdk.run_evaluation(ec, skill_space, fl_items))
            all_eval_configs.append(ec)
        except Exception:
            logger.exception("  FlexLearner eval '%s' failed", ec.label)

    # ── 2. Base-learner benchmarks ───────────────────────────────────── #
    logger.info("  PHASE 2: Base-learner benchmarks")
    for ec in build_base_learner_configs(
        timestamp,
        sdk.config.tagged_practice_items_with_responses_csv,
    ):
        ec.output_dir = evalset_dir / "base_learner"
        logger.info("Running: %s", ec.label)
        try:
            all_results.append(
                sdk.run_base_learner_evaluation(ec, skill_space, bl_items),
            )
            all_eval_configs.append(ec)
        except Exception:
            logger.exception("  Base-learner eval '%s' failed", ec.label)

    # ── 3. Aggregate & display results ──────────────────────────────── #
    evalset_results = sdk.aggregate_results(
        all_results,
        eval_configs=all_eval_configs,
        evalset_label=evalset_label,
        output_dir=evalset_dir,
    )
    evalset_results.print_summary()
    logger.info("Summary saved to: %s", evalset_dir / "evalset_summary.json")


if __name__ == "__main__":
    main()
