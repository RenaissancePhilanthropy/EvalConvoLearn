"""Main SDK entry point for EvalConvoLearn."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from evalconvolearn.models.binary_skills_flexlearner import (
    BinarySkillsFlexLearner,
    StudentPool,
)
from evalconvolearn.models.evaluation_results import (
    BenchmarkRunSummary,
    EvalSetResults,
    EvaluationResults,
    build_evalset_results,
)
from evalconvolearn.models.practice_item import PracticeItemPool
from evalconvolearn.models.skill import SkillSpace
from evalconvolearn.services.base_learner_evaluation_service import (
    BaseLearnerEvaluationService,
)
from evalconvolearn.services.conversation_service import ConversationService
from evalconvolearn.services.session_service import (
    BaseConversationSession,
    ConversationSession,
)
from evalconvolearn.storage.file_storage import FileStudentPoolStorage

from .base_learner import BaseLearner
from .config import EvalConvoLearnConfig
from .flexlearner import FlexLearner

if TYPE_CHECKING:
    from evalconvolearn.models.evaluation import EvaluationConfig

logger = logging.getLogger(__name__)

__all__ = [
    "BenchmarkRunSummary",
    "EvaluationResults",
    "EvalSetResults",
    "EvalConvoLearn",
]


class EvalConvoLearn:
    """Main SDK interface for EvalConvoLearn.

    Examples
    --------
        >>> from evalconvolearn import EvalConvoLearn
        >>> sdk = EvalConvoLearn()
        >>>
        >>> # Load skill space (path from SKILL_SPACE_PATH env var or explicit)
        >>> skill_space = sdk.load_skill_space()
        >>>
        >>> # Load practice items (path from TAGGED_PRACTICE_ITEMS_WITH_RESPONSES_CSV env var or explicit)
        >>> items = sdk.load_practice_items(skill_space)
        >>>
        >>> # Create learner pool
        >>> pool = sdk.create_learner_pool("pool_1", skill_space)
        >>> learner = pool.create_learner("learner_1", mastered_skills=["skill_1"])
        >>>
        >>> # Run conversation with custom tutor
        >>> session = sdk.create_session(pool, learner)
        >>> for response in session.conversation(items[0], my_tutor):
        ...     print(response)

    """

    def __init__(self, config: EvalConvoLearnConfig | None = None) -> None:
        """Initialize EvalConvoLearn SDK."""
        self.config = config or EvalConvoLearnConfig()
        self._conversation_service = ConversationService(self.config)
        self._pool_storage = FileStudentPoolStorage()

    def load_skill_space(self, csv_path: str | Path | None = None) -> SkillSpace:
        """Load skill space from CSV file.

        If ``csv_path`` is omitted the value of the ``SKILL_SPACE_PATH`` env var
        (exposed via :attr:`~EvalConvoLearnConfig.skill_space_path`) is used.
        """
        path = csv_path or self.config.skill_space_path
        if path is None:
            raise ValueError(
                "csv_path must be provided or SKILL_SPACE_PATH env var must be set.",
            )
        skill_space = SkillSpace()
        skill_space.load_skills_from_csv(str(path))
        return skill_space

    def load_practice_items(
        self,
        skill_space: SkillSpace,
        json_path: str | Path | None = None,
    ) -> PracticeItemPool:
        """Load tagged practice items from CSV file.

        If ``json_path`` is omitted the value of the
        ``TAGGED_PRACTICE_ITEMS_WITH_RESPONSES_CSV`` env var is used.
        """
        path = json_path or self.config.tagged_practice_items_with_responses_csv
        if path is None:
            raise ValueError(
                "json_path must be provided or TAGGED_PRACTICE_ITEMS_WITH_RESPONSES_CSV env var must be set.",
            )
        pool = PracticeItemPool(items=[], skill_space=skill_space)
        pool.load_items_from_csv(str(path))
        return pool

    def load_oversampled_items(
        self,
        skill_space: SkillSpace,
        csv_path: str | Path | None = None,
    ) -> PracticeItemPool:
        """Load oversampled practice items from CSV file.

        If ``csv_path`` is omitted the value of the ``OVERSAMPLED_ITEMS_CSV``
        env var is used.
        """
        path = csv_path or self.config.oversampled_items_csv
        if path is None:
            raise ValueError(
                "csv_path must be provided or OVERSAMPLED_ITEMS_CSV env var must be set.",
            )
        pool = PracticeItemPool(items=[], skill_space=skill_space)
        pool.load_items_from_csv(str(path))
        return pool

    def create_learner_pool(
        self,
        pool_id: str,
        skill_space: SkillSpace,
        learner_class: type = BinarySkillsFlexLearner,
    ) -> StudentPool:
        """Create a new student pool with timestamped directory."""
        return StudentPool(
            id=pool_id,
            learner_class=learner_class,
            skill_space=skill_space,
            base_directory=self.config.student_pools_dir,
        )

    def load_student_pool_exact(
        self,
        pool_directory: str | Path,
        skill_space: SkillSpace,
    ) -> StudentPool:
        """Load student pool from exact directory path.

        Args:
        ----
            pool_directory: Exact path to the pool directory (e.g., 'pool_id_20240101_120000')
            skill_space: The skill space to use

        """
        pool_path = Path(pool_directory) if isinstance(pool_directory, str) else pool_directory

        if pool_path.is_absolute() or pool_path.exists():
            pass  # Use pool_path as-is
        else:
            pool_path = self.config.student_pools_dir / pool_path

        practice_csv = pool_path / "practice.csv"

        if not pool_path.exists():
            raise FileNotFoundError(f"Student pool directory not found: {pool_path}")

        if not practice_csv.exists():
            raise FileNotFoundError(f"Practice history file not found: {practice_csv}")

        return self._pool_storage.load_pool(practice_csv, skill_space)

    def load_student_pool_most_recent(
        self,
        pool_id: str,
        skill_space: SkillSpace,
    ) -> StudentPool:
        """Load the most recent student pool instance with the given ID.

        Searches for directories matching pattern 'pool_id_YYYYMMDD_HHMMSS' and loads
        the one with the most recent timestamp.

        Args:
        ----
            pool_id: The base pool ID (without timestamp)
            skill_space: The skill space to use

        """
        pools_dir = self.config.student_pools_dir

        if not pools_dir.exists():
            raise FileNotFoundError(f"Student pools directory not found: {pools_dir}")

        matching_dirs = []
        for item in pools_dir.iterdir():
            if item.is_dir() and item.name.startswith(f"{pool_id}_"):
                matching_dirs.append(item)

        if not matching_dirs:
            raise FileNotFoundError(
                f"No student pool instances found for ID '{pool_id}' in {pools_dir}",
            )

        most_recent = sorted(matching_dirs, key=lambda x: x.name)[-1]
        return self.load_student_pool_exact(most_recent, skill_space)

    def create_session(
        self,
        student_pool: StudentPool,
        learner: BaseLearner | FlexLearner,
        session_id: str | None = None,
    ) -> ConversationSession | BaseConversationSession:
        """Create a conversation session for a learner."""
        if isinstance(learner, BaseLearner) and not isinstance(learner, FlexLearner):
            return self.create_base_learner_session(learner, session_id=session_id)

        from ..services.session_service import SessionService

        session_service = SessionService(self.config)
        return session_service.create_session(
            student_pool=student_pool,
            learner=learner,
            session_id=session_id,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def run_evaluation(
        self,
        eval_config: EvaluationConfig,
        skill_space: SkillSpace,
        practice_item_pool: PracticeItemPool,
        skill_misconceptions: dict[str, str] | None = None,
    ) -> EvaluationResults:
        """Run a full benchmark evaluation and return an :class:`EvaluationResults`.

        Parameters
        ----------
        eval_config:
            Top-level evaluation configuration describing which learner
            archetypes and benchmarks to run.
        skill_space:
            The :class:`~evalconvolearn.models.skill.SkillSpace` used by all
            benchmarks.
        practice_item_pool:
            The :class:`~evalconvolearn.models.practice_item.PracticeItemPool`
            used by all benchmarks.
        skill_misconceptions:
            Optional mapping of skill ID → misconception description string

        Returns
        -------
        EvaluationResults
            Aggregated results including per-benchmark pass/fail summaries and
            paths to all output artifacts.

        Examples
        --------
            >>> from evalconvolearn import EvalConvoLearn
            >>> from evalconvolearn.models.evaluation import EvaluationConfig, LearnerEvalConfig
            >>> from evalconvolearn.models.binary_skills_flexlearner import BinarySkillsFlexLearner
            >>> sdk = EvalConvoLearn()
            >>> skill_space = sdk.load_skill_space("data/skills.csv")
            >>> items = sdk.load_practice_items("data/items.csv", skill_space)
            >>> eval_config = EvaluationConfig(
            ...     learner_configs=[LearnerEvalConfig(learner_class=BinarySkillsFlexLearner, label="default")],
            ...     benchmarks=["PlacementTestBenchmark"],
            ... )
            >>> results = sdk.run_evaluation(eval_config, skill_space, items)
            >>> print(results.all_passed, results.output_paths)

        """
        if any(
            issubclass(lc.learner_class, BaseLearner) and not issubclass(lc.learner_class, BinarySkillsFlexLearner)
            for lc in eval_config.learner_configs
        ):
            return self.run_base_learner_evaluation(
                eval_config=eval_config,
                skill_space=skill_space,
                practice_item_pool=practice_item_pool,
                skill_misconceptions=skill_misconceptions,
            )

        from ..services.evaluation_service import EvaluationService

        service = EvaluationService(
            eval_config=eval_config,
            skill_space=skill_space,
            practice_item_pool=practice_item_pool,
            sdk_config=self.config,
            skill_misconceptions=skill_misconceptions,
        )
        raw = service.run()
        return EvaluationResults._from_raw(raw)

    def create_base_learner_session(
        self,
        learner: BaseLearner,
        session_id: str | None = None,
        max_turns: int | None = None,
    ) -> BaseConversationSession:
        """Create a conversation session for a `BaseLearner`."""
        return BaseConversationSession(
            learner=learner,
            session_id=session_id,
            max_turns=max_turns or self.config.max_conversation_turns,
        )

    def run_base_learner_evaluation(
        self,
        eval_config: EvaluationConfig,
        skill_space: SkillSpace,
        practice_item_pool: PracticeItemPool,
        skill_misconceptions: dict[str, str] | None = None,
    ) -> EvaluationResults:
        """Run benchmarks for `BaseLearner` subclasses.

        This method mirrors `run_evaluation` but uses a separate
        evaluation service that does **not** depend on
        `StudentPool` or `BinarySkillsFlexLearner`.
        Only ``learner_configs`` whose ``learner_class`` extends
        `BaseLearner` (and *not* `BinarySkillsFlexLearner`) will
        be processed.

        >>> from evalconvolearn import EvalConvoLearn
        >>> from evalconvolearn.models.evaluation import EvaluationConfig, LearnerEvalConfig
        >>> from my_learner import MyBlackBoxLearner
        >>> sdk = EvalConvoLearn()
        >>> eval_config = EvaluationConfig(
        ...     learner_configs=[
        ...         LearnerEvalConfig(
        ...             learner_class=MyBlackBoxLearner,
        ...             label="my_bb_learner",
        ...             mastered_skills=["skill_1"],
        ...         ),
        ...     ],
        ...     benchmarks=["BaseLinePlacementTestBenchmark"],
        ... )
        >>> results = sdk.run_base_learner_evaluation(eval_config, skill_space, items)
        >>> print(results.all_passed)
        """
        service = BaseLearnerEvaluationService(
            eval_config=eval_config,
            skill_space=skill_space,
            practice_item_pool=practice_item_pool,
            sdk_config=self.config,
            skill_misconceptions=skill_misconceptions,
        )
        raw = service.run()
        return EvaluationResults._from_raw(raw)

    def aggregate_results(
        self,
        results: list[EvaluationResults],
        eval_configs: list[EvaluationConfig] | None = None,
        evalset_label: str | None = None,
        output_dir: Path | str | None = None,
    ) -> EvalSetResults:
        """Aggregate multiple :class:`EvaluationResults` into an :class:`EvalSetResults`.

        Groups individual benchmark run summaries by benchmark x learner type,
        merges structured metrics across runs, and optionally saves a summary JSON.

        Parameters
        ----------
        results:
            List of :class:`EvaluationResults` returned by :meth:`run_evaluation`
            or :meth:`run_base_learner_evaluation`.
        eval_configs:
            Optional list of the :class:`EvaluationConfig` objects used to produce
            *results*.  When provided, learner class names are derived from the
            configs directly instead of falling back to the config label string.
        evalset_label:
            Human-readable name for this evaluation set.
        output_dir:
            If provided, the summary JSON is written here automatically.

        Returns
        -------
        EvalSetResults
            Aggregated results including per-benchmark x per-learner-type metrics
            and a reference to every individual result.

        Examples
        --------
            >>> results = []
            >>> for ec in my_fl_configs:
            ...     results.append(sdk.run_evaluation(ec, skill_space, fl_items))
            >>> for ec in my_bl_configs:
            ...     results.append(sdk.run_base_learner_evaluation(ec, skill_space, bl_items))
            >>> evalset = sdk.aggregate_results(results, eval_configs=my_fl_configs + my_bl_configs)
            >>> evalset.print_summary()
            >>> evalset.save("outputs/my_run")

        """
        return build_evalset_results(
            results=results,
            eval_configs=eval_configs,
            evalset_label=evalset_label,
            output_dir=output_dir,
        )
