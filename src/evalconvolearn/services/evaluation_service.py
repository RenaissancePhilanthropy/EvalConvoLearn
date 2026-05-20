from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ..benchmarks.flexlearners.learning_from_conversation_benchmark import (
    LearningFromConversationBenchmark,
)
from ..benchmarks.flexlearners.multi_conversations_practice_benchmark import (
    MultiConversationsPracticeBenchmark,
)
from ..benchmarks.flexlearners.placement_test_benchmark import PlacementTestBenchmark
from ..core.config import EvalConvoLearnConfig
from ..models.binary_skills_flexlearner import StudentPool
from ..models.evaluation import BenchmarkName, EvaluationConfig, LearnerEvalConfig
from ..models.practice_item import PracticeItemPool
from ..models.skill import SkillSpace

logger = logging.getLogger(__name__)


class EvaluationService:
    """Orchestrates benchmark evaluations driven by EvaluationConfig.

    1. Accept an EvaluationConfig eval_config.
    2. Build a StudentPool for every LearnerEvalConfig entry
    3. Create a timestamped run directory under config.evaluations_dir.
    4. Dispatch to each selected benchmark, passing runs_per_scenario and per-benchmark output directories from config.
    5. Write a consolidated evaluation_summary.json alongside each benchmark's native output artifacts.
    """

    def __init__(
        self,
        eval_config: EvaluationConfig,
        skill_space: SkillSpace,
        practice_item_pool: PracticeItemPool,
        sdk_config: EvalConvoLearnConfig | None = None,
        skill_misconceptions: dict[str, str] | None = None,
    ) -> None:
        self.eval_config = eval_config
        self.skill_space = skill_space
        self.practice_item_pool = practice_item_pool
        self.sdk_config = sdk_config or EvalConvoLearnConfig()
        self.skill_misconceptions = skill_misconceptions or {}

        self._run_id = str(uuid.uuid4())[:8]
        self._run_dir: Path | None = None
        self._validate_skills()

    def _validate_skills(self) -> None:
        skill_ids = {skill.id for skill in self.skill_space.skills}
        for learner_config in self.eval_config.learner_configs:
            for skill_id in learner_config.mastered_skills:
                if skill_id not in skill_ids:
                    raise ValueError(
                        f"Invalid mastered skill ID '{skill_id}' in learner config '{learner_config.label}'. "
                        f"Mastered skills must be present in the skill space.",
                    )
        if self.eval_config.skill_levels:
            for benchmark_name, levels in self.eval_config.skill_levels.items():
                for level_name, skill_ids_in_level in levels.items():
                    for skill_id in skill_ids_in_level:
                        if skill_id not in skill_ids:
                            raise ValueError(
                                f"Invalid skill ID '{skill_id}' in skill_levels for benchmark '{benchmark_name}', level '{level_name}'. "
                                f"All skill IDs must be present in the skill space.",
                            )

    def run(self) -> dict[str, Any]:
        """Execute the full evaluation and return a summary dict.

        The summary is also persisted to <run_dir>/evaluation_summary.json.
        """
        self._run_dir = self._create_run_dir()
        logger.info(
            "EvaluationService run started — id=%s dir=%s",
            self._run_id,
            self._run_dir,
        )
        pools = self._build_pools()
        summary: dict[str, Any] = {
            "run_id": self._run_id,
            "run_dir": str(self._run_dir),
            "started_at": datetime.now().isoformat(),
            "label": self.eval_config.label,
            "benchmarks_requested": self.eval_config.resolved_benchmarks,
            "learner_configs": [
                {
                    "label": lc.label,
                    "learner_class": lc.learner_class,
                }
                for lc in self.eval_config.learner_configs
            ],
            "benchmarks": {},
        }

        for learner_config in self.eval_config.learner_configs:
            lc_label = learner_config.label
            benchmark_list = self.eval_config.resolved_benchmarks[lc_label]
            lc_pool = pools[lc_label]
            for benchmark_name in benchmark_list:
                logger.info("Running benchmark: %s", benchmark_name)
                benchmark_label_output_dir = self._benchmark_label_dir(
                    benchmark_name,
                    lc_label,
                )
                benchmark_label_output_dir.mkdir(parents=True, exist_ok=True)
                try:
                    result = self._dispatch(
                        benchmark_name,
                        lc_pool,
                        learner_config,
                        benchmark_label_output_dir,
                    )
                    if benchmark_name not in summary["benchmarks"]:
                        summary["benchmarks"][benchmark_name] = {}
                    summary["benchmarks"][benchmark_name][lc_label] = {
                        "status": "success",
                        "output": result,
                    }
                except Exception as exc:
                    logger.exception(
                        "Benchmark %s - learner config label %s failed: %s",
                        benchmark_name,
                        lc_label,
                        exc,
                    )
                    if benchmark_name not in summary["benchmarks"]:
                        summary["benchmarks"][benchmark_name] = {}
                    summary["benchmarks"][benchmark_name][lc_label] = {
                        "status": "error",
                        "error": str(exc),
                    }

        summary["finished_at"] = datetime.now().isoformat()
        self._write_summary(summary)
        return summary

    def _create_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label_slug = (
            f"_{self.eval_config.label.replace(' ', '_')}"
            if self.eval_config.label
            else ""
        )
        # eval_config.output_dir takes precedence to support multi-run setups
        root_evals = self.eval_config.output_dir or self.sdk_config.evaluations_dir
        run_dir = Path(root_evals) / f"{timestamp}_{self._run_id}{label_slug}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _benchmark_label_dir(
        self,
        benchmark_name: BenchmarkName,
        lc_label: str,
    ) -> Path:
        assert self._run_dir is not None
        return self._run_dir / benchmark_name / lc_label

    def _build_pools(self) -> dict[str, StudentPool]:
        """Build one StudentPool per LearnerEvalConfig under the run directory."""
        assert self._run_dir is not None
        pools: dict[str, StudentPool] = {}

        for lc in self.eval_config.learner_configs:
            label: str = lc.label or lc.learner_class.__name__
            pool_dir = self._run_dir / "pools" / label
            pool_dir.mkdir(parents=True, exist_ok=True)
            conversations_file = pool_dir / "conversations.jsonl"
            conversations_file.touch(exist_ok=True)

            pool = StudentPool(
                id=f"eval_pool_{label}_{self._run_id}",
                learner_class=lc.learner_class,
                base_directory=str(pool_dir),
                directory_file=str(pool_dir),
                practice_conversations_file=str(conversations_file),
            )
            pools[label] = pool
            logger.debug("Built pool '%s' with learner", pool.id)

        return pools

    def _dispatch(
        self,
        benchmark_name: BenchmarkName,
        pool: StudentPool,
        learner_config: LearnerEvalConfig,
        output_dir: Path,
    ) -> dict[str, Any]:
        handlers = {
            "PlacementTestBenchmark": self._run_placement_test,
            "LearningFromConversationBenchmark": self._run_learning_from_conversation,
            "MultiConversationsPracticeBenchmark": self._run_multi_conversations_practice,
        }
        handler = handlers[benchmark_name]
        return handler(pool, learner_config, output_dir)

    def _run_placement_test(
        self,
        pool: StudentPool,
        learner_config: LearnerEvalConfig,
        output_dir: Path,
    ) -> dict[str, Any]:
        benchmark = PlacementTestBenchmark(
            skill_space=self.skill_space,
            practice_item_pool=self.practice_item_pool,
            runs_per_level=self.eval_config.runs_per_scenario,
            output_dir=output_dir,
            learner_pool=pool,
            learner_config=learner_config,
            skill_levels=(
                self.eval_config.skill_levels.get("PlacementTestBenchmark")
                if self.eval_config.skill_levels
                else None
            ),
            benchmark_extra_args=(
                self.eval_config.benchmarks_custom_args.get(
                    "PlacementTestBenchmark",
                    {},
                )
                if self.eval_config.benchmarks_custom_args
                else None
            ),
        )
        output_file = benchmark.run_all_evaluations()
        structured_metrics = PlacementTestBenchmark.compute_structured_metrics(
            output_file,
        )
        return {
            "benchmark_class": "PlacementTestBenchmark",
            "output_file": str(output_file),
            "runs_per_level": self.eval_config.runs_per_scenario,
            "structured_metrics": structured_metrics,
        }

    def _run_learning_from_conversation(
        self,
        pool: StudentPool,
        learner_config: LearnerEvalConfig,
        output_dir: Path,
    ) -> dict[str, Any]:
        benchmark = LearningFromConversationBenchmark(
            skill_space=self.skill_space,
            practice_item_pool=self.practice_item_pool,
            runs_per_level=self.eval_config.runs_per_scenario,
            output_dir=output_dir,
            learner_pool=pool,
            learner_config=learner_config,
            skill_levels=(
                self.eval_config.skill_levels.get("LearningFromConversationBenchmark")
                if self.eval_config.skill_levels
                else None
            ),
            benchmark_extra_args=(
                self.eval_config.benchmarks_custom_args.get(
                    "LearningFromConversationBenchmark",
                    {},
                )
                if self.eval_config.benchmarks_custom_args
                else None
            ),
        )
        output_file = benchmark.run_all_evaluations()
        structured_metrics = (
            LearningFromConversationBenchmark.compute_structured_metrics(output_file)
        )
        return {
            "benchmark_class": "LearningFromConversationBenchmark",
            "output_file": str(output_file),
            "runs_per_level": self.eval_config.runs_per_scenario,
            "structured_metrics": structured_metrics,
        }

    def _run_multi_conversations_practice(
        self,
        pool: StudentPool,
        learner_config: LearnerEvalConfig,
        output_dir: Path,
    ) -> dict[str, Any]:
        extra_args = (
            self.eval_config.benchmarks_custom_args.get(
                "MultiConversationsPracticeBenchmark",
                {},
            )
            if self.eval_config.benchmarks_custom_args
            else {}
        )

        # "oversampled_item_pool" may be injected via benchmarks_custom_args;
        # the benchmark falls back to practice_item_pool when absent.
        oversampled_item_pool = extra_args.pop("oversampled_item_pool", None)

        benchmark = MultiConversationsPracticeBenchmark(
            skill_space=self.skill_space,
            practice_item_pool=self.practice_item_pool,
            oversampled_item_pool=oversampled_item_pool,
            output_dir=output_dir,
            learner_pool=pool,
            learner_config=learner_config,
            skill_levels=(
                self.eval_config.skill_levels.get("MultiConversationsPracticeBenchmark")
                if self.eval_config.skill_levels
                else None
            ),
            benchmark_extra_args=extra_args or None,
        )
        output_file = benchmark.run_all_evaluations()
        structured_metrics = (
            MultiConversationsPracticeBenchmark.compute_structured_metrics(output_file)
        )
        return {
            "benchmark_class": "MultiConversationsPracticeBenchmark",
            "output_file": str(output_file),
            "structured_metrics": structured_metrics,
        }

    def _write_summary(self, summary: dict[str, Any]) -> None:
        assert self._run_dir is not None
        summary_path = self._run_dir / "evaluation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        logger.info("Evaluation summary written to %s", summary_path)
