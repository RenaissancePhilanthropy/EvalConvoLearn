"""Base benchmark class for all flexlearner benchmark evaluators."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from pathlib import Path

from ...models.binary_skills_flexlearner import StudentPool
from ...models.evaluation import LearnerEvalConfig
from ...models.practice_item import PracticeItemPool
from ...models.skill import SkillSpace
from ...utils import get_benchmark_output_dir


class FlexLearnerBenchmark(ABC):
    """Abstract base class for all benchmark evaluators.

    Provides shared initialization logic.
    """

    def __init__(
        self,
        skill_space: SkillSpace,
        practice_item_pool: PracticeItemPool,
        output_dir: Path | None = None,
        learner_pool: StudentPool | None = None,
        learner_config: LearnerEvalConfig | None = None,
        benchmark_extra_args: dict | None = None,
    ) -> None:
        """Initialize shared benchmark state.

        Args:
        ----
            skill_space: SkillSpace object defining all skills.
            practice_item_pool: Pool of practice items used for evaluation.
            output_dir: Directory to save results.
            learner_pool: Optional StudentPool used to create learners during evaluation.
            learner_config: Optional LearnerEvalConfig.
            benchmark_extra_args: Optional dict of benchmark-specific extra
                arguments (e.g. paths to mocked response files).

        """
        self.skill_space = skill_space
        self.practice_item_pool = practice_item_pool
        self.output_dir = output_dir or get_benchmark_output_dir()
        self.learner_pool = learner_pool
        self.learner_config = learner_config
        self.benchmark_extra_args = benchmark_extra_args or {}
        self.logger = logging.getLogger(__name__)

        for key, value in self.benchmark_extra_args.items():
            setattr(self, key, value)

    # ------------------------------------------------------------------
    # Skill-levels helpers
    # ------------------------------------------------------------------

    def _resolve_skill_levels(
        self,
        skill_levels: Mapping[str, Iterable[str]],
    ) -> dict[str, set[str]]:
        """Apply ``learner_config`` overrides to a ``skill_levels`` mapping.

        Resolution order (highest priority last):

        1. Use ``skill_levels`` as the base.
        2. If ``learner_config.mastered_skills`` is set, replace the whole
           mapping with a single ``"default"`` entry.
        3. If ``learner_config.learner_level`` is set, keep only that entry
           from the (possibly already narrowed) mapping.

        Args:
        ----
            skill_levels: Base mapping of level names to sets of skill IDs.

        Returns:
        -------
            Resolved mapping.

        Raises:
        ------
            ValueError: If a ``learner_level`` specified in ``learner_config``
                does not exist in ``skill_levels``.

        """
        resolved: dict[str, set[str]] = {k: set(v) for k, v in skill_levels.items()}

        if self.learner_config and self.learner_config.mastered_skills:
            resolved = {"default": set(self.learner_config.mastered_skills)}

        if self.learner_config and self.learner_config.learner_level:
            specified_level = self.learner_config.learner_level
            if specified_level in resolved:
                resolved = {specified_level: resolved[specified_level]}
            else:
                raise ValueError(
                    f"Learner level '{specified_level}' specified in learner_config "
                    f"not found in skill_levels. Available levels: {list(resolved.keys())}",
                )

        return resolved

    def _validate_skill_levels(self, skill_levels: dict[str, set[str]]) -> None:
        """Validate that every skill referenced in ``skill_levels`` exists in the skill space.

        Args:
        ----
            skill_levels: Mapping of level names to sets of skill IDs to check.

        Raises:
        ------
            ValueError: If any skill ID is absent from the skill space.

        """
        all_skill_ids = {skill.id for skill in self.skill_space.skills}
        for level, skills in skill_levels.items():
            missing_skills = set(skills) - all_skill_ids
            if missing_skills:
                raise ValueError(
                    f"Skills defined for learner level '{level}' are missing from skill space: {missing_skills}",
                )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run_all_evaluations(self) -> Path:
        """Run all evaluations and return the path to the results output.

        Returns
        -------
            Path to the primary output file or directory produced by the benchmark.

        """
