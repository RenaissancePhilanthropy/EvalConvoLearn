"""Shared utilities for BaseLearner benchmark modules."""

from __future__ import annotations

import logging
from pathlib import Path

from evalconvolearn.core.base_learner import BaseLearner, LearnerInitializationError
from evalconvolearn.models.evaluation import LearnerEvalConfig
from evalconvolearn.models.practice_item import PracticeItem, PracticeItemPool
from evalconvolearn.models.skill import SkillSpace
from evalconvolearn.models.tutor import Tutor

logger = logging.getLogger(__name__)


def _create_learner_for_scenario(
    learner_config: LearnerEvalConfig,
    skill_space: SkillSpace,
    mastered_skill_ids: list[str],
    learner_id: str,
    practice_conversations_file: Path | str | None = None,
    practice_item_pool: PracticeItemPool | None = None,
    tutor: Tutor | None = None,
) -> BaseLearner:
    """Instantiate and initialise a BaseLearner for one benchmark scenario."""
    learner: BaseLearner = learner_config.learner_class(
        id=learner_id,
        skill_space=skill_space,
        practice_conversations_file=practice_conversations_file,
    )
    try:
        learner.initialize_from_skills(
            mastered_skill_ids=mastered_skill_ids,
            practice_item_pool=practice_item_pool,
            tutor=tutor,
            **learner_config.init_knowledge_kwargs,
        )
    except LearnerInitializationError:
        raise
    except Exception as exc:
        msg = f"Learner {learner_id!r} initialization failed: {exc}"
        raise LearnerInitializationError(msg) from exc
    return learner


def _items_for_skill_scenario(
    pool: PracticeItemPool,
    mastered_ids: set[str],
    want_mastered: bool,
    skill_space: SkillSpace,
    max_items: int = 4,
    retrieve_all_learner_skill_prerequisites: bool = True,
    select_items_near_mastery_boundary_first: bool = True,
    item_prerequisites_should_be_mastered: bool = False,
) -> list[PracticeItem]:
    """Select practice items whose skills are all mastered (or not).

    Thin wrapper around ``PracticeItemPool.get_items_for_skill_scenario``.
    The ``skill_space`` argument is accepted for backward compatibility but
    ignored — the pool's own ``skill_space`` is used internally.
    """
    return pool.get_items_for_skill_scenario(
        mastered_ids=mastered_ids,
        want_mastered=want_mastered,
        max_items=max_items,
        retrieve_all_learner_skill_prerequisites=retrieve_all_learner_skill_prerequisites,
        select_items_near_mastery_boundary_first=select_items_near_mastery_boundary_first,
        item_prerequisites_should_be_mastered=item_prerequisites_should_be_mastered,
    )


DEFAULT_BL_CONSOLIDATION_RUNS = 3
DEFAULT_BL_MAX_CONVERSATION_TURNS = 6
DEFAULT_BL_MAX_CLIMB_ITEMS_PER_SKILL = 3


__all__ = [
    "DEFAULT_BL_CONSOLIDATION_RUNS",
    "DEFAULT_BL_MAX_CLIMB_ITEMS_PER_SKILL",
    "DEFAULT_BL_MAX_CONVERSATION_TURNS",
    "_create_learner_for_scenario",
    "_items_for_skill_scenario",
]
