import uuid
from pathlib import Path

import pytest

from evalconvolearn.models.binary_skills_flexlearner import (
    BinarySkillsFlexLearner,
    StudentPool,
)
from evalconvolearn.models.practice_item import PracticeItemPool
from evalconvolearn.models.skill import SkillSpace
from evalconvolearn.utils import (
    generate_learning_alignment_matrix,
    generate_placement_test_alignment_matrix,
    get_benchmark_output_dir,
    get_placement_test_skill_levels,
    get_tutor_responses_csv_path,
    load_tutor_responses_mapping,
)


@pytest.fixture(scope="session")
def benchmark_output_dir() -> Path:
    """Directory for storing benchmark evaluation results."""
    return get_benchmark_output_dir()


@pytest.fixture(scope="session")
def placement_test_skill_levels() -> dict[str, list[str]]:
    """Skill levels for placement test learners at different proficiency levels."""
    return get_placement_test_skill_levels()


@pytest.fixture(scope="session")
def placement_test_alignment_matrix() -> dict:
    """Alignment matrix from tagged practice item skills and level skill outlines."""
    return generate_placement_test_alignment_matrix()


@pytest.fixture(scope="session")
def tutor_responses_csv_path() -> Path:
    """Path to CSV file with generated tutor responses (helpful and unhelpful)."""
    return get_tutor_responses_csv_path()


@pytest.fixture(scope="session")
def tutor_responses_mapping() -> dict:
    """Mapping from problem text to helpful/unhelpful tutor responses."""
    return load_tutor_responses_mapping()


@pytest.fixture()
def run_id() -> str:
    """Unique run ID for test execution."""
    return str(uuid.uuid4())[:8]


@pytest.fixture()
def populated_student_pool(
    skill_space: SkillSpace,
    selected_config: dict,
    tmp_path_factory: pytest.TempPathFactory,
) -> StudentPool:
    """StudentPool pre-populated with 3 beginner learners."""
    pool_id = f"benchmark_{uuid.uuid4()}"
    pool = StudentPool(id=pool_id, learners=[])
    for i in range(3):
        practice_file = tmp_path_factory.mktemp("populated_student_pool") / f"{pool_id}_{i}.jsonl"
        practice_file.touch()
        learner = BinarySkillsFlexLearner(
            id=f"learner_{i}",
            mastered_skills=selected_config["mastered_skills"].copy(),
            skill_space=skill_space,
            practice_conversations_file=practice_file,
        )
        pool.add_learner(learner)
    return pool


@pytest.fixture()
def learning_from_conversation_alignment_matrix(skill_space: SkillSpace, practice_item_pool: PracticeItemPool) -> dict:
    """Alignment matrix for learning from conversation evaluation."""
    return generate_learning_alignment_matrix(skill_space, practice_item_pool)
