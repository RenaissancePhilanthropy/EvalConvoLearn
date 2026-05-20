"""Evaluation service for BaseLearner (black-box) benchmarks.

This is intentionally separate from `EvaluationService` to avoid
coupling with `StudentPool` and `FlexLearner`.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.config import EvalConvoLearnConfig
from ..models.evaluation import EvaluationConfig
from ..models.practice_item import PracticeItemPool
from ..models.skill import SkillSpace

logger = logging.getLogger(__name__)

# Mapping from benchmark name to class
_BASELINE_BENCHMARKS: dict[str, type] = {}


def _load_benchmark_classes() -> dict[str, type]:
    """Lazy-import to avoid circular dependencies."""
    if not _BASELINE_BENCHMARKS:
        from ..benchmarks.base_learners import (
            BaseLineLearningFromConversationBenchmark,
            BaselineMultiConversationsBenchmark,
            BaseLinePlacementTestBenchmark,
        )
        from ..benchmarks.realistic_benchmarks_from_conversation_data import (
            DatasetFittedConversationalBenchmark,
        )

        _BASELINE_BENCHMARKS.update(
            {
                "BaseLinePlacementTestBenchmark": BaseLinePlacementTestBenchmark,
                "BaseLineLearningFromConversationBenchmark": BaseLineLearningFromConversationBenchmark,
                "BaselineMultiConversationsBenchmark": BaselineMultiConversationsBenchmark,
                "DatasetFittedConversationalBenchmark": DatasetFittedConversationalBenchmark,
            },
        )
    return _BASELINE_BENCHMARKS


class BaseLearnerEvaluationService:
    """Orchestrate benchmark runs for :class:`BaseLearner` subclasses.

    Parameters
    ----------
    eval_config:
        Top-level evaluation configuration.  Only ``learner_configs``
        whose ``is_base_learner`` flag is True will be processed.
    skill_space / practice_item_pool:
        Shared across all benchmarks.
    sdk_config:
        FlexLearner SDK configuration.
    skill_misconceptions:
        Forwarded to confusion-alignment benchmarks.

    """

    def __init__(
        self,
        eval_config: EvaluationConfig,
        skill_space: SkillSpace,
        practice_item_pool: PracticeItemPool,
        sdk_config: EvalConvoLearnConfig | None = None,
        skill_misconceptions: dict[str, str] | None = None,
    ):
        self.eval_config = eval_config
        self.skill_space = skill_space
        self.practice_item_pool = practice_item_pool
        self.sdk_config = sdk_config or EvalConvoLearnConfig()
        self.skill_misconceptions = skill_misconceptions or {}

        # get label from eval_config and parse it to create an eval label id
        eval_config_label = getattr(eval_config, "label", None)
        if eval_config_label:
            self.eval_label_id = eval_config_label.lower().replace(" ", "_")
        else:
            self.eval_label_id = f"bl_eval_{uuid.uuid4().hex[:8]}"

    def run(self) -> dict[str, Any]:
        """Execute all requested benchmarks and return a results dict."""
        started_at = datetime.now().isoformat()
        run_id = f"{self.eval_label_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        root_output_dir = Path(
            self.eval_config.output_dir or self.sdk_config.evaluations_dir,
        )
        output_dir = root_output_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        benchmark_classes = _load_benchmark_classes()

        # Determine which benchmarks to run
        requested: list[str]
        if self.eval_config.benchmarks == "all" or self.eval_config.benchmarks is None:
            requested = list(benchmark_classes.keys())
        else:
            requested = [
                b for b in self.eval_config.benchmarks if b in benchmark_classes
            ]

        results: dict[str, Any] = {
            "run_id": run_id,
            "run_dir": str(output_dir),
            "label": self.eval_config.label,
            "started_at": started_at,
            "benchmarks": {},
        }

        for lconfig in self.eval_config.learner_configs:
            if not lconfig.is_base_learner:
                logger.info(
                    "Skipping non-BaseLearner config: %s",
                    lconfig.label,
                )
                continue

            # Per-learner-config benchmarks can override the top-level list
            lc_benchmarks = (
                requested
                if lconfig.benchmarks is None
                else (
                    list(benchmark_classes.keys())
                    if lconfig.benchmarks == "all"
                    else [b for b in lconfig.benchmarks if b in benchmark_classes]
                )
            )

            # Resolve skill levels from eval_config or learner_config
            skill_levels = self.eval_config.skill_levels or {}
            if lconfig.mastered_skills and not skill_levels:
                skill_levels = {"default": set(lconfig.mastered_skills)}

            for bname in lc_benchmarks:
                logger.info(
                    "Running %s for learner config '%s'",
                    bname,
                    lconfig.label,
                )
                BenchmarkCls = benchmark_classes[bname]

                # Build kwargs common to all benchmarks
                benchmark_extra_args = (
                    self.eval_config.benchmarks_custom_args.get(bname, {})  # type: ignore[arg-type]
                    if self.eval_config.benchmarks_custom_args
                    else {}
                )
                kwargs: dict[str, Any] = {
                    "skill_space": self.skill_space,
                    "practice_item_pool": self.practice_item_pool,
                    "learner_config": lconfig,
                    "skill_levels": skill_levels,
                    "output_dir": output_dir,
                    "benchmark_extra_args": benchmark_extra_args,
                    "practice_conversations_file": output_dir
                    / f"{lconfig.label}_conversations.jsonl",
                }

                try:
                    benchmark = BenchmarkCls(**kwargs)
                    output_file = benchmark.run_all_evaluations()

                    # Compute structured metrics if the benchmark supports it
                    structured_metrics = None
                    if hasattr(BenchmarkCls, "compute_structured_metrics"):
                        try:
                            structured_metrics = (
                                BenchmarkCls.compute_structured_metrics(output_file)
                            )
                        except Exception as exc:
                            logger.debug(
                                "Could not compute structured metrics: %s",
                                exc,
                            )

                    if bname not in results["benchmarks"]:
                        results["benchmarks"][bname] = {}
                    results["benchmarks"][bname][lconfig.label] = {
                        "status": "success",
                        "output": {
                            "benchmark_class": bname,
                            "output_file": str(output_file),
                            "structured_metrics": structured_metrics,
                        },
                    }
                except Exception as e:
                    logger.exception("Benchmark %s failed: %s", bname, e)
                    if bname not in results["benchmarks"]:
                        results["benchmarks"][bname] = {}
                    results["benchmarks"][bname][lconfig.label] = {
                        "status": "error",
                        "error": str(e),
                        "output": {},
                    }

        results["finished_at"] = datetime.now().isoformat()
        return results
