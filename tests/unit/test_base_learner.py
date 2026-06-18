"""Unit tests for BaseLearner / BinarySkillsFlexLearner basic API."""

from pathlib import Path

import pytest

from evalconvolearn.models.binary_skills_flexlearner import BinarySkillsFlexLearner
from evalconvolearn.models.skill import SkillSpace


class TestBaseLearnerInitialization:
    """Test learner construction and basic attribute access."""

    def test_learner_has_id(self, learner: BinarySkillsFlexLearner) -> None:
        assert learner.id is not None
        assert isinstance(learner.id, str)

    def test_learner_has_skill_space(self, learner: BinarySkillsFlexLearner, skill_space: SkillSpace) -> None:
        assert learner.skill_space is not None
        assert learner.skill_space == skill_space

    def test_learner_has_mastered_skills(self, learner: BinarySkillsFlexLearner, selected_config: dict) -> None:
        assert isinstance(learner.mastered_skills, list)
        assert len(learner.mastered_skills) > 0
        for skill_id in selected_config["mastered_skills"]:
            assert skill_id in learner.mastered_skills

    def test_learner_practice_history_logged_on_init(self, learner: BinarySkillsFlexLearner) -> None:
        assert len(learner.practice_history) == 1
        assert learner.practice_history[0]["session_id"] == "initialization"

    def test_learner_practice_file_exists(self, learner: BinarySkillsFlexLearner) -> None:
        assert Path(learner.practice_conversations_file).exists()

    def test_duplicate_skills_rejected(self, skill_space: SkillSpace, tmp_path: Path) -> None:
        practice_file = tmp_path / "test.jsonl"
        practice_file.touch()
        with pytest.raises(ValueError, match="duplicate skill"):
            BinarySkillsFlexLearner(
                id="dup-test",
                mastered_skills=["MA.6.NSO.1.1", "MA.6.NSO.1.1"],
                skill_space=skill_space,
                practice_conversations_file=str(practice_file),
            )

    def test_unknown_skill_rejected(self, skill_space: SkillSpace, tmp_path: Path) -> None:
        practice_file = tmp_path / "test.jsonl"
        practice_file.touch()
        with pytest.raises(
            ValueError,
            match="Skill with id NONEXISTENT.SKILL.ID in Learner skills is not part of the defined SkillSpace.",
        ):
            BinarySkillsFlexLearner(
                id="unknown-test",
                mastered_skills=["NONEXISTENT.SKILL.ID"],
                skill_space=skill_space,
                practice_conversations_file=str(practice_file),
            )


class TestBaseLearnerHasSkill:
    """Test has_skill() method."""

    def test_has_mastered_skill_returns_true(self, learner: BinarySkillsFlexLearner, selected_config: dict) -> None:
        for skill_id in selected_config["mastered_skills"]:
            assert learner.has_skill(skill_id) is True

    def test_has_unmastered_skill_returns_false(
        self, learner: BinarySkillsFlexLearner, skill_space: SkillSpace
    ) -> None:
        unmastered = [sk.id for sk in skill_space if sk.id not in learner.mastered_skills]
        if unmastered:
            assert learner.has_skill(unmastered[0]) is False

    def test_has_nonexistent_skill_returns_false(self, learner: BinarySkillsFlexLearner) -> None:
        assert learner.has_skill("DOES.NOT.EXIST") is False


class TestBaseLearnerKnowledge:
    """Test knowledge description methods."""

    def test_get_knowledge_description_returns_string(self, learner: BinarySkillsFlexLearner) -> None:
        desc = learner.get_knowledge_description()
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_knowledge_description_mentions_mastered_skills(self, learner: BinarySkillsFlexLearner) -> None:
        desc = learner.get_knowledge_description()
        for skill_id in learner.mastered_skills:
            assert skill_id in desc

    def test_empty_learner_knowledge_description(self, skill_space: SkillSpace, tmp_path: Path) -> None:
        practice_file = tmp_path / "empty.jsonl"
        practice_file.touch()
        learner = BinarySkillsFlexLearner(
            id="empty-learner",
            mastered_skills=[],
            skill_space=skill_space,
            practice_conversations_file=str(practice_file),
        )
        desc = learner.get_knowledge_description()
        assert isinstance(desc, str)
