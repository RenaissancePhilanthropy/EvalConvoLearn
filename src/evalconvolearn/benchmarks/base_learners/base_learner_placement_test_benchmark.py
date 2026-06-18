"""Placement benchmark for `BaseLearner` implementations."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from evalconvolearn.benchmarks.base_learners.base_learner_benchmarks import (
    _create_learner_for_scenario,
    _items_for_skill_scenario,
)
from evalconvolearn.core.base_learner import BaseLearner, LearnerInitializationError
from evalconvolearn.models.evaluation import LearnerEvalConfig
from evalconvolearn.models.practice_item import PracticeItem, PracticeItemPool
from evalconvolearn.models.skill import SkillSpace
from evalconvolearn.utils.llm_evaluator import evaluate_response_correctness

logger = logging.getLogger(__name__)


class BaseLinePlacementTestBenchmark:
    """Can the learner solve problems for skills it has, and fail for skills it lacks?"""

    def __init__(
        self,
        skill_space: SkillSpace,
        practice_item_pool: PracticeItemPool,
        learner_config: LearnerEvalConfig,
        skill_levels: dict[str, set[str]],
        runs: int = 1,
        max_items: int = 10,
        output_dir: Path | None = None,
        benchmark_extra_args: dict | None = None,
        practice_conversations_file: Path | str | None = None,
    ) -> None:
        self.skill_space = skill_space
        self.practice_item_pool = practice_item_pool
        self.learner_config = learner_config
        self.skill_levels = skill_levels
        self.runs = runs
        self.max_items = max_items
        self.output_dir = output_dir or Path("data/benchmark_evaluations/base_learner")
        self.practice_conversations_file = practice_conversations_file
        self.test_run_id = f"bl_placement_{uuid.uuid4().hex[:8]}"
        self.benchmark_extra_args = benchmark_extra_args or {}
        self.max_assessment_turns: int = 1

        for key, value in self.benchmark_extra_args.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def run_all_evaluations(self) -> Path:
        output_dir = self.output_dir / self.test_run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        all_results: list[dict] = []

        for level_name, mastered_ids in self.skill_levels.items():
            mastered_set = set(mastered_ids)
            mastered_items = _items_for_skill_scenario(
                self.practice_item_pool,
                mastered_set,
                want_mastered=True,
                max_items=self.max_items,
                skill_space=self.skill_space,
                retrieve_all_learner_skill_prerequisites=True,
                select_items_near_mastery_boundary_first=True,
                item_prerequisites_should_be_mastered=False,
            )
            unmastered_items = _items_for_skill_scenario(
                self.practice_item_pool,
                mastered_set,
                want_mastered=False,
                max_items=self.max_items,
                skill_space=self.skill_space,
                retrieve_all_learner_skill_prerequisites=True,
                select_items_near_mastery_boundary_first=True,
                item_prerequisites_should_be_mastered=False,
            )

            for run_id in range(self.runs):
                logger.info(
                    "[BaseLinePlacementTestBenchmark] run=%s level=%s mastered_items=%d unmastered_items=%d",
                    run_id,
                    level_name,
                    len(mastered_items),
                    len(unmastered_items),
                )

                for item_id, item in enumerate(mastered_items):
                    try:
                        learner = _create_learner_for_scenario(
                            self.learner_config,
                            self.skill_space,
                            list(mastered_ids),
                            learner_id=f"placement_{level_name}_{run_id}_mastered_items_{item_id}",
                            practice_conversations_file=self.practice_conversations_file,
                            practice_item_pool=self.practice_item_pool,
                        )
                    except LearnerInitializationError as exc:
                        logger.warning(
                            "[BaseLinePlacementTestBenchmark] Skipping learner run "
                            "level=%s run_id=%d mastered item_id=%d — initialization failed: %s",
                            level_name,
                            run_id,
                            item_id,
                            exc,
                        )
                        all_results.append(
                            {
                                "benchmark": "BaseLinePlacementTestBenchmark",
                                "level": level_name,
                                "run_id": run_id,
                                "scenario": "mastered",
                                "item_text": item.text[:100],
                                "item_skills": item.associated_skills,
                                "expect_correct": True,
                                "learner_response": None,
                                "is_correct": None,
                                "asked_followup": None,
                                "expectation_met": False,
                                "reasoning": "Learner initialization failed.",
                                "max_assessment_turns": self.max_assessment_turns,
                                "initialization_failed": True,
                                "timestamp": datetime.now().isoformat(),
                            },
                        )
                        continue
                    all_results.append(
                        self._evaluate_item(
                            learner,
                            item,
                            expect_correct=True,
                            level=level_name,
                            run_id=run_id,
                            scenario="mastered",
                        ),
                    )

                for item_id, item in enumerate(unmastered_items):
                    try:
                        learner = _create_learner_for_scenario(
                            self.learner_config,
                            self.skill_space,
                            list(mastered_ids),
                            learner_id=f"placement_{level_name}_{run_id}_unmastered_items_{item_id}",
                            practice_conversations_file=self.practice_conversations_file,
                            practice_item_pool=self.practice_item_pool,
                        )
                    except LearnerInitializationError as exc:
                        logger.warning(
                            "[BaseLinePlacementTestBenchmark] Skipping learner run "
                            "level=%s run_id=%d unmastered item_id=%d — initialization failed: %s",
                            level_name,
                            run_id,
                            item_id,
                            exc,
                        )
                        all_results.append(
                            {
                                "benchmark": "BaseLinePlacementTestBenchmark",
                                "level": level_name,
                                "run_id": run_id,
                                "scenario": "unmastered",
                                "item_text": item.text[:100],
                                "item_skills": item.associated_skills,
                                "expect_correct": False,
                                "learner_response": None,
                                "is_correct": None,
                                "asked_followup": None,
                                "expectation_met": False,
                                "reasoning": "Learner initialization failed.",
                                "max_assessment_turns": self.max_assessment_turns,
                                "initialization_failed": True,
                                "timestamp": datetime.now().isoformat(),
                            },
                        )
                        continue
                    all_results.append(
                        self._evaluate_item(
                            learner,
                            item,
                            expect_correct=False,
                            level=level_name,
                            run_id=run_id,
                            scenario="unmastered",
                        ),
                    )

        output_file = output_dir / "placement_test_results.jsonl"
        with open(output_file, "w", encoding="utf-8") as file_handle:
            for result in all_results:
                file_handle.write(json.dumps(result) + "\n")

        summary = self._compute_summary(all_results)
        summary_file = output_dir / "placement_test_summary.json"
        with open(summary_file, "w", encoding="utf-8") as file_handle:
            json.dump(summary, file_handle, indent=2)

        logger.info("Placement test results → %s", output_file)
        logger.info("Placement test summary → %s", summary_file)
        return output_file

    def _compute_summary(self, all_results: list[dict]) -> dict:
        df = pd.DataFrame(all_results)
        if df.empty:
            return {
                "overall": {},
                "by_scenario": {},
                "by_level": {},
                "by_level_and_scenario": {},
            }

        def pct_met(subset: pd.DataFrame) -> dict:
            total = len(subset)
            met = int(subset["expectation_met"].sum())
            return {
                "expectation_met_count": met,
                "total": total,
                "expectation_met_pct": round(met / total * 100, 2) if total else None,
            }

        overall = pct_met(df)
        by_scenario = {scenario: pct_met(group) for scenario, group in df.groupby("scenario")}
        by_level = {level: pct_met(group) for level, group in df.groupby("level")}

        by_level_and_scenario: dict[str, dict[str, dict]] = {}
        for (level, scenario), group in df.groupby(["level", "scenario"]):
            by_level_and_scenario.setdefault(level, {})[scenario] = pct_met(group)

        return {
            "test_run_id": self.test_run_id,
            "runs": self.runs,
            "timestamp": datetime.now().isoformat(),
            "overall": overall,
            "by_scenario": by_scenario,
            "by_level": by_level,
            "by_level_and_scenario": by_level_and_scenario,
        }

    @staticmethod
    def compute_structured_metrics(output_file: Path) -> dict:
        with open(output_file, encoding="utf-8") as file_handle:
            summary = json.load(file_handle)

        by_scenario = summary.get("by_scenario", {})
        breakdowns: dict[str, dict] = {}
        for scenario_name, values in by_scenario.items():
            pct = values.get("expectation_met_pct")
            breakdowns[scenario_name] = {
                "avg_alignment": (pct / 100.0) if pct is not None else 0.0,
                "n_items": values.get("total", 0),
            }

        overall = summary.get("overall", {})
        overall_pct = overall.get("expectation_met_pct")
        return {
            "metric_type": "alignment",
            "overall_avg_alignment": ((overall_pct / 100.0) if overall_pct is not None else 0.0),
            "total_items": overall.get("total", 0),
            "breakdowns": {"by_scenario": breakdowns},
            "breakdown_keys": ["scenario"],
        }

    def _evaluate_item(
        self,
        learner: BaseLearner,
        item: PracticeItem,
        expect_correct: bool,
        level: str,
        run_id: int,
        scenario: str,
    ) -> dict:
        correct_answer = getattr(item, "answer", "")
        response = learner.assess_with_problem(
            problem_text=item.text,
            max_turns=self.max_assessment_turns,
            item_answer=correct_answer,
        )
        response = (
            response if isinstance(response, str) else [resp["content"] for resp in response if resp["role"] == "user"]
        )
        verdict = evaluate_response_correctness(
            problem_text=item.text,
            learner_response=response,
            correct_answer=correct_answer,
        )
        expectation_met = verdict.is_correct == expect_correct

        return {
            "benchmark": "BaseLinePlacementTestBenchmark",
            "level": level,
            "run_id": run_id,
            "scenario": scenario,
            "item_text": item.text[:100],
            "item_skills": item.associated_skills,
            "expect_correct": expect_correct,
            "learner_response": response[:300],
            "is_correct": verdict.is_correct,
            "asked_followup": verdict.asked_followup,
            "expectation_met": expectation_met,
            "reasoning": verdict.reasoning,
            "max_assessment_turns": self.max_assessment_turns,
            "timestamp": datetime.now().isoformat(),
        }
