"""Integration tests for LearningFromConversationBenchmark with FlexLearner simulations."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from evalconvolearn.benchmarks.flexlearners.learning_from_conversation_benchmark import (
    LearningFromConversationBenchmark,
)
from evalconvolearn.models.practice_item import PracticeItemPool
from evalconvolearn.models.skill import SkillSpace


@pytest.mark.benchmark()
class TestLearningFromConversationBenchmark:
    """Simulate LearningFromConversationBenchmark runs with BinarySkillsFlexLearner."""

    def test_benchmark_initializes(
        self,
        skill_space: SkillSpace,
        practice_item_pool: PracticeItemPool,
        placement_test_skill_levels: dict,
        tutor_responses_mapping: dict,
        tmp_path: Path,
    ) -> None:
        if not tutor_responses_mapping:
            pytest.skip("No tutor responses available")

        benchmark = LearningFromConversationBenchmark(
            skill_space=skill_space,
            practice_item_pool=practice_item_pool,
            runs_per_level=1,
            output_dir=tmp_path,
            skill_levels={"beginner": set(placement_test_skill_levels["beginner"])},
            max_items=2,
        )
        assert benchmark is not None
        assert benchmark.skill_space is not None
        assert "beginner" in benchmark.skill_levels

    def test_benchmark_runs_helpful_responses(
        self,
        skill_space: SkillSpace,
        practice_item_pool: PracticeItemPool,
        placement_test_skill_levels: dict,
        tutor_responses_mapping: dict,
        tmp_path: Path,
    ) -> None:
        if not tutor_responses_mapping:
            pytest.skip("No tutor responses available")

        benchmark = LearningFromConversationBenchmark(
            skill_space=skill_space,
            practice_item_pool=practice_item_pool,
            runs_per_level=1,
            output_dir=tmp_path,
            skill_levels={"beginner": set(placement_test_skill_levels["beginner"])},
            max_items=2,
        )
        with TemporaryDirectory() as tmp_dir:
            results = benchmark.run_evaluation_for_level(
                "beginner",
                "helpful",
                0,
                Path(tmp_dir),
            )

        assert isinstance(results, list)

    def test_benchmark_runs_unhelpful_responses(
        self,
        skill_space: SkillSpace,
        practice_item_pool: PracticeItemPool,
        placement_test_skill_levels: dict,
        tutor_responses_mapping: dict,
        tmp_path: Path,
    ) -> None:
        if not tutor_responses_mapping:
            pytest.skip("No tutor responses available")

        benchmark = LearningFromConversationBenchmark(
            skill_space=skill_space,
            practice_item_pool=practice_item_pool,
            runs_per_level=1,
            output_dir=tmp_path,
            skill_levels={"beginner": set(placement_test_skill_levels["beginner"])},
            max_items=2,
        )
        with TemporaryDirectory() as tmp_dir:
            results = benchmark.run_evaluation_for_level(
                "beginner",
                "unhelpful",
                0,
                Path(tmp_dir),
            )

        assert isinstance(results, list)
