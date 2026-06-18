"""Integration tests for PlacementTestBenchmark with FlexLearner simulations."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from evalconvolearn.benchmarks.flexlearners.placement_test_benchmark import (
    PlacementTestBenchmark,
)
from evalconvolearn.models.practice_item import PracticeItemPool
from evalconvolearn.models.skill import SkillSpace


@pytest.mark.benchmark()
class TestPlacementTestBenchmark:
    """Simulate PlacementTestBenchmark runs with BinarySkillsFlexLearner."""

    def test_benchmark_initializes(
        self,
        skill_space: SkillSpace,
        practice_item_pool: PracticeItemPool,
        placement_test_skill_levels: dict,
        tmp_path: Path,
    ) -> None:
        benchmark = PlacementTestBenchmark(
            skill_space=skill_space,
            practice_item_pool=practice_item_pool,
            runs_per_level=1,
            output_dir=tmp_path,
            skill_levels={"beginner": set(placement_test_skill_levels["beginner"])},
        )
        assert benchmark is not None
        assert benchmark.skill_space is not None
        assert benchmark.runs_per_level == 1
        assert "beginner" in benchmark.skill_levels

    def test_benchmark_runs_one_level(
        self,
        skill_space: SkillSpace,
        practice_item_pool: PracticeItemPool,
        placement_test_skill_levels: dict,
        tmp_path: Path,
    ) -> None:
        benchmark = PlacementTestBenchmark(
            skill_space=skill_space,
            practice_item_pool=practice_item_pool,
            runs_per_level=1,
            output_dir=tmp_path,
            skill_levels={"beginner": set(placement_test_skill_levels["beginner"])},
        )
        with TemporaryDirectory() as tmp_dir:
            result = benchmark.run_evaluation_for_level("beginner", 0, Path(tmp_dir))

        assert isinstance(result, dict)
        assert "test_name" in result
        assert "alignment_accuracy" in result
        assert "items" in result
        assert isinstance(result["items"], list)
        assert 0.0 <= result["alignment_accuracy"] <= 1.0
