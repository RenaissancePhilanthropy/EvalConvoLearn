import uuid
from pathlib import Path

import pytest

from evalconvolearn.models.binary_skills_flexlearner import (
    BinarySkillsFlexLearner,
    StudentPool,
)
from evalconvolearn.models.practice_item import PracticeItem, PracticeItemPool
from evalconvolearn.models.skill import SkillSpace
from evalconvolearn.models.tutor import Tutor


@pytest.fixture()
def skills_csv_path():
    """Path to example skills CSV file."""
    return Path(__file__).parent.parent / "data" / "florida-doe" / "skill-space.csv"


@pytest.fixture()
def tagged_items_csv_path():
    """Path to example tagged items CSV file."""
    return (
        Path(__file__).parent.parent
        / "data"
        / "florida-doe"
        / "tagged-practice-items-with-responses.csv"
    )


@pytest.fixture()
def skill_space(skills_csv_path):
    """SkillSpace loaded from example CSV."""
    skill_space = SkillSpace()
    skill_space.load_skills_from_csv(skills_csv_path)
    return skill_space


@pytest.fixture()
def practice_item_pool(tagged_items_csv_path, skill_space):
    """PracticeItemPool loaded from example CSV file."""
    pool = PracticeItemPool(items=[], skill_space=skill_space)
    pool.load_items_from_csv(tagged_items_csv_path)
    return pool


@pytest.fixture()
def beginner_config():
    """Beginner learner config with a single root skill."""
    return {
        "mastered_skills": ["MA.6.NSO.1.1"],
        "description": "Beginner learner with only a single skill",
    }


@pytest.fixture()
def intermediate_config():
    """Intermediate learner config with a few skills."""
    return {
        "mastered_skills": ["MA.6.NSO.1.1", "MA.6.NSO.1.2", "MA.6.NSO.1.3"],
        "description": "Intermediate learner with 3 skills.",
    }


@pytest.fixture()
def chosen_learner_type():
    """Default learner type for tests."""
    return "beginner"


@pytest.fixture()
def selected_config(beginner_config, intermediate_config, chosen_learner_type):
    """Configuration for the chosen learner type."""
    configs = {
        "beginner": beginner_config,
        "intermediate": intermediate_config,
    }
    return configs[chosen_learner_type]


@pytest.fixture()
def learner(skill_space, tmp_path, selected_config):
    """BinarySkillsFlexLearner initialized with selected skill config."""
    practice_file = tmp_path / "test_practice_conversations.jsonl"
    practice_file.touch()
    return BinarySkillsFlexLearner(
        id=str(uuid.uuid4()),
        mastered_skills=selected_config["mastered_skills"].copy(),
        skill_space=skill_space,
        practice_conversations_file=str(practice_file),
    )


@pytest.fixture()
def student_pool(learner):
    """StudentPool with one learner."""
    return StudentPool(id="test-pool", learners=[learner])


@pytest.fixture()
def random_practice_item(practice_item_pool) -> PracticeItem:
    """A random practice item from the pool."""
    return practice_item_pool.get_random_item()


@pytest.fixture()
def session_id():
    """Unique session ID."""
    return str(uuid.uuid4())[:8]


@pytest.fixture()
def helpful_tutor(practice_item_pool):
    """A Tutor configured to be helpful (strategy not initialized)."""
    return Tutor(
        id=str(uuid.uuid4()),
        tutor_type="llm",
        tutor_characteristics={"helpfulness": True},
        practice_item_pool=practice_item_pool,
        response_interaction_mode="return_only",
    )


@pytest.fixture()
def unhelpful_tutor(practice_item_pool):
    """A Tutor configured to be unhelpful (strategy not initialized)."""
    return Tutor(
        id=str(uuid.uuid4()),
        tutor_type="llm",
        tutor_characteristics={"helpfulness": False},
        practice_item_pool=practice_item_pool,
        response_interaction_mode="return_only",
    )
