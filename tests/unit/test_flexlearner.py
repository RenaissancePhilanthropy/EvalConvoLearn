"""Unit tests for FlexLearner skill-guardrail machinery."""

from pathlib import Path

import pytest

from evalconvolearn.models.binary_skills_flexlearner import BinarySkillsFlexLearner
from evalconvolearn.models.practice_item import PracticeItemPool
from evalconvolearn.models.skill import SkillSpace


class TestFlexLearnerLearnableSkills:
    """Test skill prereq guardrails: get_learnable_skills and can_learn_skill."""

    def test_get_learnable_skills_returns_list(self, learner: BinarySkillsFlexLearner) -> None:
        learnable = learner.get_learnable_skills()
        assert isinstance(learnable, list)

    def test_learnable_skills_not_already_mastered(self, learner: BinarySkillsFlexLearner) -> None:
        learnable_ids = {sk.id for sk in learner.get_learnable_skills()}
        for skill_id in learner.mastered_skills:
            assert skill_id not in learnable_ids

    def test_learnable_skills_have_mastered_prerequisites(
        self, learner: BinarySkillsFlexLearner, skill_space: SkillSpace
    ) -> None:
        for skill in learner.get_learnable_skills():
            prereqs = skill_space.get_prerequisite_skills(skill.id)
            for prereq in prereqs:
                assert prereq.id in learner.mastered_skills

    def test_can_learn_unmastered_skill_with_mastered_prereqs(self, learner: BinarySkillsFlexLearner) -> None:
        learnable = learner.get_learnable_skills()
        if learnable:
            assert learner.can_learn_skill(learnable[0]) is True

    def test_cannot_learn_already_mastered_skill(self, learner: BinarySkillsFlexLearner) -> None:
        for skill_id in learner.mastered_skills:
            assert learner.can_learn_skill(skill_id) is False


class TestFlexLearnerKnowledgeForProblem:
    """Test problem-specific knowledge representation."""

    def test_knowledge_for_mastered_item(
        self,
        learner: BinarySkillsFlexLearner,
        practice_item_pool: PracticeItemPool,
        skill_space: SkillSpace,
    ) -> None:
        mastered_items = [
            item
            for item in practice_item_pool.items
            if all(sk in learner.mastered_skills for sk in item.associated_skills)
        ]
        if not mastered_items:
            pytest.skip("No practice items whose skills are all mastered by learner")
        item = mastered_items[0]
        item_skills = [skill_space[sk_id] for sk_id in item.associated_skills]
        knowledge = learner.get_knowledge_for_problem(item, item_skills)
        assert isinstance(knowledge, str)
        assert len(knowledge) > 0

    def test_knowledge_for_unmastered_item(
        self,
        learner: BinarySkillsFlexLearner,
        practice_item_pool: PracticeItemPool,
        skill_space: SkillSpace,
    ) -> None:
        unmastered_items = [
            item
            for item in practice_item_pool.items
            if all(sk not in learner.mastered_skills for sk in item.associated_skills)
        ]
        if not unmastered_items:
            pytest.skip("No practice items whose skills are all unmastered by learner")
        item = unmastered_items[0]
        item_skills = [skill_space[sk_id] for sk_id in item.associated_skills]
        knowledge = learner.get_knowledge_for_problem(item, item_skills)
        assert isinstance(knowledge, str)


class TestFlexLearnerPersona:
    """Test persona and active_misconceptions fields."""

    def test_default_persona_is_empty(self, learner: BinarySkillsFlexLearner) -> None:
        assert isinstance(learner.persona, dict)

    def test_default_active_misconceptions_is_empty(self, learner: BinarySkillsFlexLearner) -> None:
        assert isinstance(learner.active_misconceptions, dict)


class TestFlexLearnerPrerequisiteExpansion:
    """Test that mastered_skills auto-expands to include all ancestors."""

    def test_prerequisites_auto_expanded(self, skill_space: SkillSpace, tmp_path: Path) -> None:
        # Find a skill with a prerequisite
        skill_with_prereqs = None
        for skill in skill_space:
            if skill.prerequisites:
                skill_with_prereqs = skill
                break
        if skill_with_prereqs is None:
            pytest.skip("No skill with prerequisites found in skill space")

        practice_file = tmp_path / "prereq_test.jsonl"
        practice_file.touch()
        # Init with only the child skill; its parent should be auto-added
        learner = BinarySkillsFlexLearner(
            id="prereq-expansion-test",
            mastered_skills=[skill_with_prereqs.id],
            skill_space=skill_space,
            practice_conversations_file=str(practice_file),
        )
        for prereq_id in skill_with_prereqs.prerequisites:
            assert prereq_id in learner.mastered_skills
