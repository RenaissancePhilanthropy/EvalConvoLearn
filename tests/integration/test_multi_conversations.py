"""Integration tests for MultiConversationsPracticeBenchmark with FlexLearner simulations."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from evalconvolearn.benchmarks.flexlearners.multi_conversations_practice_benchmark import (
    MultiConversationsPracticeBenchmark,
)
from evalconvolearn.models.practice_item import PracticeItemPool
from evalconvolearn.models.skill import SkillSpace


@pytest.mark.benchmark()
class TestMultiConversationsBenchmark:
    """Simulate MultiConversationsPracticeBenchmark runs with BinarySkillsFlexLearner."""

    def test_benchmark_initializes(
        self,
        skill_space: SkillSpace,
        practice_item_pool: PracticeItemPool,
        placement_test_skill_levels: dict,
        tmp_path: Path,
    ) -> None:
        benchmark = MultiConversationsPracticeBenchmark(
            skill_space=skill_space,
            practice_item_pool=practice_item_pool,
            output_dir=tmp_path,
            skill_levels={"beginner": set(placement_test_skill_levels["beginner"])},
            consolidation_runs=1,
            max_conversation_turns=3,
            max_climb_items_per_skill=1,
        )
        assert benchmark is not None
        assert benchmark.skill_space is not None
        assert "beginner" in benchmark.target_skills

    def test_benchmark_runs_one_skill(
        self,
        skill_space: SkillSpace,
        practice_item_pool: PracticeItemPool,
        tmp_path: Path,
    ) -> None:
        benchmark = MultiConversationsPracticeBenchmark(
            skill_space=skill_space,
            practice_item_pool=practice_item_pool,
            output_dir=tmp_path,
            target_skills={"beginner": ["MA.6.NSO.1.2"]},
            consolidation_runs=1,
            max_conversation_turns=3,
            max_climb_items_per_skill=1,
        )
        with TemporaryDirectory() as tmp_dir:
            result = benchmark.run_evaluation_for_target_skill(
                tier="beginner",
                target_skill_id="MA.6.NSO.1.2",
                tmp_dir=Path(tmp_dir),
            )
        assert isinstance(result, dict)
        assert "tier" in result
        assert "target_skill" in result
