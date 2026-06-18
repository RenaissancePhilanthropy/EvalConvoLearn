"""Alignment matrix generation for benchmarks."""

from collections.abc import Iterable, Mapping

from ..models.practice_item import PracticeItemPool
from ..models.skill import SkillSpace
from .data_loaders import load_tagged_skill_ids
from .learner_configs import get_placement_test_skill_levels


def generate_placement_test_alignment_matrix() -> dict:
    """Generate alignment matrix from tagged practice item skills and level skill outlines.

    For each learner level, determines which skills should be answered correctly
    (mastered) and which should be answered incorrectly (not mastered).

    Returns
    -------
        dict: Mapping from learner level to {'correct': set, 'incorrect': set}

    """
    all_skill_ids = load_tagged_skill_ids()
    placement_test_skill_levels = get_placement_test_skill_levels()

    matrix = {}
    for level, skills in placement_test_skill_levels.items():
        correct = set(skills)
        incorrect = all_skill_ids - correct
        matrix[level] = {"correct": correct, "incorrect": incorrect}

    return matrix


def generate_learning_alignment_matrix(
    skill_space: SkillSpace,
    practice_item_pool: PracticeItemPool,
    placement_test_skill_levels: Mapping[str, Iterable[str]] | None = None,
) -> dict:
    """Generate alignment matrix for learning from conversation evaluation.

    For each learner level and practice item skill, determines if the skill should be learned
    in each of the 2 scenarios:
    - helpful_response: Should learn if prerequisites met and not already mastered
    - unhelpful_response: Should NOT learn (unhelpful response prevents learning)

    Args:
    ----
        skill_space: SkillSpace object containing skill definitions and prerequisites
        practice_item_pool: PracticeItemPool containing all practice items

    Returns:
    -------
        dict: Nested structure {learner_level: {skill_id: {scenario: should_learn}}}

    """
    if not placement_test_skill_levels:
        placement_test_skill_levels = get_placement_test_skill_levels()
    matrix = {}

    all_practice_item_skills = set()
    for item in practice_item_pool.items:
        all_practice_item_skills.update(item.associated_skills)

    for level, mastered_skills in placement_test_skill_levels.items():
        matrix[level] = {}
        # Expand mastered_set to include all transitive prerequisites,
        # matching the expansion that Learner/SimulatedLearner applies at init time.
        mastered_set = set(mastered_skills)
        for sid in set(mastered_skills):
            if sid in skill_space:
                mastered_set.update(
                    skill_space.get_all_prerequisites(sid, return_as_ids=True),
                )

        for skill_id in all_practice_item_skills:
            prerequisites = set()
            if skill_id in skill_space:
                skill = skill_space[skill_id]
                if skill.prerequisites:
                    prerequisites = set(skill.prerequisites)

            already_mastered = skill_id in mastered_set
            prerequisites_met = prerequisites.issubset(mastered_set)
            is_learnable = prerequisites_met and not already_mastered

            matrix[level][skill_id] = {
                "helpful_response": is_learnable,
                "unhelpful_response": False,
            }

    return matrix
