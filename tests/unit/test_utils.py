"""Unit tests for utility functions."""

from pathlib import Path

from evalconvolearn.utils import (
    get_beginner_config,
    get_benchmark_output_dir,
    get_blank_config,
    get_data_dir,
    get_expert_config,
    get_florida_doe_data_dir,
    get_intermediate_config,
    get_placement_test_skill_levels,
)


class TestDataPaths:
    """Test data path utilities."""

    def test_get_data_dir_returns_path(self):
        data_dir = get_data_dir()
        assert data_dir is not None
        assert isinstance(data_dir, Path)

    def test_get_florida_doe_data_dir_returns_path(self):
        florida_dir = get_florida_doe_data_dir()
        assert florida_dir is not None
        assert isinstance(florida_dir, Path)
        assert florida_dir.name == "florida-doe"

    def test_get_benchmark_output_dir_returns_path(self):
        output_dir = get_benchmark_output_dir()
        assert output_dir is not None
        assert isinstance(output_dir, Path)


class TestLearnerConfigs:
    """Test learner configuration utilities."""

    def test_placement_test_skill_levels_structure(self):
        skill_levels = get_placement_test_skill_levels()
        assert isinstance(skill_levels, dict)
        assert "beginner" in skill_levels
        assert "intermediate" in skill_levels
        assert "expert" in skill_levels
        for skills in skill_levels.values():
            assert isinstance(skills, list)

    def test_beginner_config_structure(self):
        config = get_beginner_config()
        assert isinstance(config, dict)
        assert "mastered_skills" in config
        assert isinstance(config["mastered_skills"], list)
        assert len(config["mastered_skills"]) > 0

    def test_intermediate_has_more_skills_than_beginner(self):
        beginner = get_beginner_config()
        intermediate = get_intermediate_config()
        assert len(intermediate["mastered_skills"]) >= len(beginner["mastered_skills"])

    def test_expert_has_more_skills_than_intermediate(self):
        intermediate = get_intermediate_config()
        expert = get_expert_config()
        assert len(expert["mastered_skills"]) >= len(intermediate["mastered_skills"])

    def test_blank_config_has_no_skills(self):
        config = get_blank_config()
        assert isinstance(config, dict)
        assert "mastered_skills" in config
        assert config["mastered_skills"] == []
