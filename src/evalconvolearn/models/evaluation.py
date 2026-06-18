"""Evaluation configuration models for FlexLearner benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, field_validator, model_validator

from ..core.base_learner import BaseLearner
from ..core.flexlearner import FlexLearner

# ---------------------------------------------------------------------------
# Benchmark identifiers — kept in sync with benchmarks.__all__
# ---------------------------------------------------------------------------

BenchmarkName = Literal[
    "PlacementTestBenchmark",
    "LearningFromConversationBenchmark",
    "MultiConversationsPracticeBenchmark",
    # Base-learner benchmarks
    "BaseLinePlacementTestBenchmark",
    "BaseLineLearningFromConversationBenchmark",
    "BaselineMultiConversationsBenchmark",
    "DatasetFittedConversationalBenchmark",
    "DatasetFittedPlacementBenchmark",
    "all",
]

ALL_BENCHMARKS: list[BenchmarkName] = list(get_args(BenchmarkName))

# Convenience alias for the per-label benchmark map accepted by EvaluationConfig
BenchmarkMap = dict[str, list[BenchmarkName] | Literal["all"]]


def _validate_benchmark_list(names: list[str]) -> list[BenchmarkName]:
    """Validate that every entry in *names* is a known BenchmarkName."""
    valid = set(ALL_BENCHMARKS)
    unknown = [n for n in names if n not in valid]
    if unknown:
        raise ValueError(
            f"Unknown benchmark(s): {unknown}. Available benchmarks: {ALL_BENCHMARKS}.",
        )
    return names  # type: ignore[return-value]


def _normalize_benchmark_value(
    v: list[str] | str | None,
) -> list[BenchmarkName] | Literal["all"] | None:
    """Shared normalization for a single benchmark field value."""
    if v is None:
        return None
    if v == "all":
        return "all"
    if isinstance(v, list):
        return _validate_benchmark_list(v)
    raise ValueError(
        f"benchmarks must be a list of benchmark names, 'all', or None; got {v!r}.",
    )


# ---------------------------------------------------------------------------
# LearnerEvalConfig
# ---------------------------------------------------------------------------


class LearnerEvalConfig(BaseModel):
    """Configuration for a single learner archetype used in an evaluation run.

    Captures:
    - ``learner_class``: a concrete subclass of BaseLearner.
    - ``init_knowledge_kwargs``: keyword arguments forwarded to
      ``learner.initialize_learner_knowledge()``. Mapping between skills and custom knowledge configuration model.
    - ``label``: optional human-readable name used in output filenames / logs.
    - ``mastered_skills``: list of skill IDs the learner has already mastered at evaluation start.
    - ``benchmarks``: benchmarks to run for this learner config specifically.
        A learner pool with learners of this config will be created and shared across all benchmarks.
      Takes precedence over any benchmark specification in the parent
      :class:`EvaluationConfig`.  Pass ``None`` (default) to defer to the
      parent config.

    Example:
    -------
        learner_config = LearnerEvalConfig(
            learner_class=MyLearner,
            label="MyLearnerConfigTest",
            mastered_skills=["skill_1", "skill_2"],
            init_knowledge_kwargs={"initial_knowledge_items_mapping": {"skill_1": "You master addition", "skill_2": "You master subtraction", "skill_3": "You have basic knowledge of multiplication"}},
            benchmarks=["PlacementTestBenchmark"],
        )

    """

    learner_class: type[BaseLearner] = Field(
        ...,
        description=("Concrete BaseLearner subclass to instantiate for this evaluation. Must not be abstract."),
    )
    label: str = Field(
        description=(
            "Human-readable identifier for this learner config, used in benchmark to run evals on"
            "different learner archetypes. Should be unique across all LearnerEvalConfig entries in an EvaluationConfig"
            "Like: 'beginner' or 'advanced'"
        ),
    )
    mastered_skills: list[str] = Field(
        default_factory=list,
        description="Skill IDs the learner has already mastered at evaluation start. Empty if using learner_level",
    )
    learner_level: str | None = Field(
        default=None,
        description=(
            "Optional learner proficiency, 'beginner', 'intermediate', 'advanced' etc."
            "Should be used instead of mastered_skills to indicate which benchmark-specific"
            "skills to test for this learner."
        ),
    )
    init_knowledge_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extra keyword arguments forwarded to "
            "``learner.initialize_learner_knowledge(**init_knowledge_kwargs)`` "
            "after the learner is constructed.  Use this to pass learner-specific "
            "initialization data."
        ),
    )
    benchmarks: list[BenchmarkName] | Literal["all"] | None = Field(
        default=None,
        description=(
            "Benchmarks to run for this specific learner config. "
            "Pass a list of benchmark names or ``'all'`` to run all benchmarks. "
            "Pass ``None`` (default) to defer benchmark selection to the parent "
            ":class:`EvaluationConfig`. "
            f"Available: {ALL_BENCHMARKS}."
        ),
    )

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("learner_class")
    @classmethod
    def must_be_concrete_subclass(cls, v: type) -> type:
        """Ensure ``learner_class`` is a non-abstract BaseLearner subclass."""
        if not (isinstance(v, type) and issubclass(v, BaseLearner)):
            raise ValueError(
                f"learner_class must be a subclass of BaseLearner, got {v!r}.",
            )
        # Pydantic/ABC marks abstract classes via __abstractmethods__
        if getattr(v, "__abstractmethods__", None):
            raise ValueError(
                f"{v.__name__} still has abstract methods "
                f"({', '.join(v.__abstractmethods__)}) and cannot be instantiated.",
            )
        return v

    @field_validator("benchmarks", mode="before")
    @classmethod
    def normalize_benchmarks(
        cls,
        v: list[str] | str | None,
    ) -> list[BenchmarkName] | Literal["all"] | None:
        """Normalise and validate the ``benchmarks`` field."""
        return _normalize_benchmark_value(v)

    # ensure that not both mastered_skills and learner_level are set
    @model_validator(mode="after")
    def check_mastered_skills_vs_level(self) -> LearnerEvalConfig:
        if self.mastered_skills and self.learner_level:
            raise ValueError(
                "Choose either mastered_skills or learner_level."
                "mastered_skills is more flexible and allows explicit specification of which skills are mastered, "
                "while learner_level is a higher-level abstraction that is mapped to predefined benchmark-specific skill levels.",
            )
        return self

    @property
    def is_base_learner(self) -> bool:
        """True when ``learner_class`` extends BaseLearner (not FlexLearner)."""
        return issubclass(self.learner_class, BaseLearner) and not issubclass(
            self.learner_class,
            FlexLearner,
        )


# ---------------------------------------------------------------------------
# EvaluationConfig
# ---------------------------------------------------------------------------


class EvaluationConfig(BaseModel):
    """Top-level configuration for a FlexLearner evaluation run.

    Groups one or more :class:`LearnerEvalConfig` entries with benchmark
    selection, number of runs, and optional output directory overrides.

    Benchmark resolution precedence (highest → lowest):

    1. ``LearnerEvalConfig.benchmarks`` — explicit per-learner override.
    2. ``EvaluationConfig.benchmarks`` as a **dict** — maps a learner config
       label to its benchmark list, e.g.
       ``{"MyLearner": ["PlacementTestBenchmark"], "OtherLearner": "all"}``.
    3. ``EvaluationConfig.benchmarks`` as a **list** or ``"all"`` — applied to
       every learner config that has not been given an explicit override.
    4. Fall-back: all available benchmarks (``ALL_BENCHMARKS``).

    To run all available benchmarks for every learner, leave ``benchmarks``
    as ``None`` or ``"all"``.
    """

    learner_configs: list[LearnerEvalConfig] = Field(
        ...,
        min_length=1,
        description=(
            "One or more learner configurations to evaluate. Each entry describes a distinct learner archetype."
        ),
    )
    output_dir: Path | None = Field(
        default=None,
        description=("Optional root directory for all benchmark output artifacts."),
    )
    runs_per_scenario: int = Field(
        default=4,
        ge=1,
        description=("Number of independent runs per evaluation scenario.  "),
    )
    benchmarks: list[BenchmarkName] | Literal["all"] | BenchmarkMap | None = Field(
        default="all",
        description=(
            "Which benchmarks to run. Accepted forms:\n"
            "- ``'all'`` or ``None`` (default): run all benchmarks for every learner.\n"
            "- ``list[BenchmarkName]``: run the given benchmarks for every learner that does not have its own override.\n"
            "- ``dict[str, list[BenchmarkName] | 'all']``: map each "
            "LearnerEvalConfig label to its benchmark list; labels not "
            "present in the dict fall back to ``ALL_BENCHMARKS``.\n"
            f"Available benchmark names: {ALL_BENCHMARKS}."
        ),
    )
    benchmarks_custom_args: dict[BenchmarkName, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Optional mapping of benchmark names to dicts of extra keyword arguments to pass to the benchmarks"
        ),
    )
    skill_levels: dict[BenchmarkName, dict[str, set[str]]] = Field(
        default_factory=dict,
        description=(
            "Optional dictionary mapping benchmark names to learner levels and their corresponding sets of skill IDs."
        ),
    )
    label: str | None = Field(
        default=None,
        description=("Optional human-readable label for the overall evaluation run, used in logs and summary reports."),
    )
    num_threads: int = Field(
        default=4,
        ge=1,
        description="Maximum number of benchmark evaluations to run concurrently.",
    )

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("benchmarks", mode="before")
    @classmethod
    def normalize_benchmarks(
        cls,
        v: list[str] | str | dict[str, list[str] | str] | None,
    ) -> list[BenchmarkName] | Literal["all"] | BenchmarkMap:
        """Normalize and validate the top-level ``benchmarks`` field."""
        if v is None or v == "all":
            return "all"
        if isinstance(v, dict):
            validated: BenchmarkMap = {}
            for label, benchmarks in v.items():
                if not isinstance(label, str):
                    raise ValueError(
                        f"Keys in the benchmarks dict must be strings (learner config labels); got {label!r}.",
                    )
                result = _normalize_benchmark_value(benchmarks)
                if result is None:
                    raise ValueError(
                        f"Dict value for label {label!r} must be a list of benchmark names or 'all', not None.",
                    )
                validated[label] = result
            return validated
        if isinstance(v, list):
            return _validate_benchmark_list(v)
        raise ValueError(
            f"benchmarks must be a list, 'all', None, or a dict mapping labels to benchmark lists; got {v!r}.",
        )

    @model_validator(mode="after")
    def validate_benchmark_coverage(self) -> EvaluationConfig:
        """Ensure every learner config has a resolvable, valid benchmark list.

        Checks:
        - All learner config labels are distinct.
        - When ``benchmarks`` is a ``dict``, all keys must match a known
          learner config label.
        - Every learner config must ultimately resolve to at least one valid
          benchmark (either from its own field, the dict, the global list, or
          the ALL_BENCHMARKS fall-back).
        """
        labels = [cfg.label for cfg in self.learner_configs]
        seen: set[str] = set()
        duplicates = {label for label in labels if label in seen or seen.add(label)}  # type: ignore[func-returns-value]
        if duplicates:
            raise ValueError(
                f"All learner_configs must have distinct labels, but found duplicate label(s): {sorted(duplicates)}.",
            )

        known_labels: set[str] = set(labels)
        # --- dict-form: validate that all keys are known learner labels ------
        if isinstance(self.benchmarks, dict):
            unknown_labels = set(self.benchmarks.keys()) - known_labels
            if unknown_labels:
                raise ValueError(
                    f"The benchmarks dict references label(s) that do not match any "
                    f"LearnerEvalConfig label: {sorted(unknown_labels)}. "
                    f"Known labels: {sorted(known_labels)}.",
                )

        # --- per-learner resolution check ------------------------------------
        for cfg in self.learner_configs:
            resolved = self._resolve_benchmarks_for(cfg)
            if not resolved:
                raise ValueError(
                    f"LearnerEvalConfig {cfg.label!r} resolved to an empty benchmark "
                    f"list. Ensure at least one benchmark is specified.",
                )

        # all learner config learner_levels are valid and consistent with skill_levels (if provided)
        if self.skill_levels:
            for cfg in self.learner_configs:
                if cfg.learner_level is not None:
                    if cfg.learner_level not in self.skill_levels.get(
                        "PlacementTestBenchmark",
                        {},
                    ):
                        raise ValueError(
                            f"LearnerEvalConfig {cfg.label!r} has learner_level {cfg.learner_level!r} "
                            f"which is not defined in skill_levels for PlacementTestBenchmark. "
                            f"Defined levels: {list(self.skill_levels.get('PlacementTestBenchmark', {}).keys())}.",
                        )

        return self

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def _resolve_benchmarks_for(
        self,
        learner_config: LearnerEvalConfig,
    ) -> list[BenchmarkName]:
        """Resolve the effective benchmark list for a single learner config.

        Precedence (highest → lowest):

        1. ``learner_config.benchmarks`` (explicit per-learner override).
        2. ``self.benchmarks[learner_config.label]`` when ``self.benchmarks``
           is a dict.
        3. ``self.benchmarks`` when it is a list or ``"all"``.
        4. Fall-back: ``ALL_BENCHMARKS``.
        """
        # 1. Per-learner explicit override
        if learner_config.benchmarks is not None:
            if learner_config.benchmarks == "all":
                return list(ALL_BENCHMARKS)
            return list(learner_config.benchmarks)

        # 2. Dict-based per-label override in EvaluationConfig
        if isinstance(self.benchmarks, dict):
            entry = self.benchmarks.get(learner_config.label) if learner_config.label is not None else None
            if entry is not None:
                if entry == "all":
                    return list(ALL_BENCHMARKS)
                return list(entry)
            # Label not in dict → fall back to ALL_BENCHMARKS
            return list(ALL_BENCHMARKS)

        # 3. Global list or "all" in EvaluationConfig
        if self.benchmarks == "all":
            return list(ALL_BENCHMARKS)
        if isinstance(self.benchmarks, list):
            return list(self.benchmarks)

        # 4. Ultimate fall-back
        return list(ALL_BENCHMARKS)

    @property
    def resolved_benchmarks(self) -> dict[str, list[BenchmarkName]]:
        """Return a mapping of each learner config label to its resolved benchmark list."""
        # TODO - when resolving benchmarks, verify that if learner_level is used, then the corresponding skill_levels are provided in the config for the relevant benchmarks (e.g. PlacementTestBenchmark)

        return {
            cfg.label or cfg.learner_class.__name__: self._resolve_benchmarks_for(cfg) for cfg in self.learner_configs
        }
