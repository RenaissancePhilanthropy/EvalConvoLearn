"""Multi-conversation practice benchmark for `BaseLearner` implementations."""

from __future__ import annotations

import json
import logging
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from evalconvolearn.benchmarks.base_learners.base_learner_benchmarks import (
    DEFAULT_BL_CONSOLIDATION_RUNS,
    DEFAULT_BL_MAX_CLIMB_ITEMS_PER_SKILL,
    DEFAULT_BL_MAX_CONVERSATION_TURNS,
    _create_learner_for_scenario,
)
from evalconvolearn.core.base_learner import BaseLearner, LearnerInitializationError
from evalconvolearn.models.base_learner_conversation import (
    BaseConversationResult,
    run_base_learner_conversation,
)
from evalconvolearn.models.evaluation import LearnerEvalConfig
from evalconvolearn.models.practice_item import PracticeItem, PracticeItemPool
from evalconvolearn.models.skill import SkillSpace
from evalconvolearn.models.tutor import Tutor

logger = logging.getLogger(__name__)


class BaselineMultiConversationsBenchmark:
    """Progressive multi-conversation practice benchmark for `BaseLearner`."""

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
        self.test_run_id = f"bl_multi_conv_{uuid.uuid4().hex[:8]}"

        self.benchmark_extra_args = benchmark_extra_args or {}
        self.consolidation_runs = self.benchmark_extra_args.get(
            "consolidation_runs",
            DEFAULT_BL_CONSOLIDATION_RUNS,
        )
        self.max_conversation_turns = self.benchmark_extra_args.get(
            "max_conversation_turns",
            DEFAULT_BL_MAX_CONVERSATION_TURNS,
        )
        self.max_climb_items_per_skill = self.benchmark_extra_args.get(
            "max_climb_items_per_skill",
            DEFAULT_BL_MAX_CLIMB_ITEMS_PER_SKILL,
        )

        for key, value in self.benchmark_extra_args.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.tutor = Tutor(
            id=str(uuid.uuid4()),
            tutor_type="llm",
            tutor_characteristics={"helpfulness": True},
            practice_item_pool=self.practice_item_pool,
            response_interaction_mode="return_only",
        )
        self.tutor.initialize_strategy()

    @staticmethod
    def compute_structured_metrics(output_file: Path) -> dict:
        with output_file.open(encoding="utf-8") as file_handle:
            summary = json.load(file_handle)

        aggregate_metrics = summary.get("aggregate_metrics", {})
        structured = {
            "metric_type": "multi_conv_practice",
            "overall_avg_turns_per_skill": aggregate_metrics.get(
                "overall_avg_turns_per_skill",
                0.0,
            ),
            "overall_consolidation_solution_rate": aggregate_metrics.get(
                "overall_consolidation_solution_rate",
                0.0,
            ),
            "targets_mastered": aggregate_metrics.get("targets_mastered", 0),
            "total_targets": aggregate_metrics.get("total_targets", 0),
            "total_skills_learned": aggregate_metrics.get("total_skills_learned", 0),
            "breakdown_keys": [],
        }
        return structured

    def _get_items_for_skill(
        self,
        skill_id: str,
        exclude_texts: set[str] | None = None,
        max_items: int | None = None,
    ) -> list[PracticeItem]:
        items = self.practice_item_pool.get_items_having_skill(skill_id)
        if exclude_texts:
            items = [item for item in items if item.text not in exclude_texts]
        if max_items is not None:
            items = items[:max_items]
        return items

    def _run_single_conversation(
        self,
        learner: BaseLearner,
        practice_item: PracticeItem,
        session_id: str,
    ) -> BaseConversationResult:
        return run_base_learner_conversation(
            learner=learner,
            practice_item=practice_item,
            tutor=self.tutor,
            max_turns=self.max_conversation_turns,
            session_id=session_id,
            item_skills=list(practice_item.associated_skills),
            save_conversation=True,
            correct_answer=getattr(practice_item, "answer", None),
        )

    def _run_evaluation_for_target_skill(
        self,
        tier: str,
        target_skill_id: str,
        learner: BaseLearner,
        run_id: int,
    ) -> dict[str, Any]:
        logger.info(
            "[BaselineMultiConv] run=%s target_skill=%s tier=%s — starting climb",
            run_id,
            target_skill_id,
            tier,
        )

        skill_order = self.skill_space.get_bfs_skill_order(target_skill_id)
        skill_order_ids = [skill.id for skill in skill_order]
        used_item_texts: set[str] = set(learner.get_problems_seen())
        climb_records: list[dict] = []
        total_climb_turns = 0
        total_skills_learned_in_climb = 0
        target_mastered = False
        known_skills: set[str] = set()

        for skill in skill_order:
            if skill.id in known_skills:
                continue

            items = self._get_items_for_skill(skill.id)
            if not items:
                logger.warning(
                    "[BaselineMultiConv] no items for skill=%s, skipping",
                    skill.id,
                )
                continue

            # create the list of items by randomly choosing items in items WITH replacement:
            # potentially working on similar items multiple times if not enough items variety
            items = [random.choice(items) for _ in range(self.max_climb_items_per_skill)]

            skill_mastered = False
            for attempt_idx, item in enumerate(items):
                used_item_texts.add(item.text)
                session_id = f"climb_{skill.id}_{attempt_idx}_{uuid.uuid4().hex[:8]}"
                conv_result = self._run_single_conversation(
                    learner=learner,
                    practice_item=item,
                    session_id=session_id,
                )

                turns = conv_result.num_turns
                total_climb_turns += turns

                climb_record = {
                    "phase": "climb",
                    "tier": tier,
                    "target_skill": target_skill_id,
                    "current_skill": skill.id,
                    "attempt": attempt_idx + 1,
                    "item_text": item.text,
                    "item_skills": item.associated_skills,
                    "turns": turns,
                    "solution_found": conv_result.solution_found,
                    "conversation_ended_reason": conv_result.conversation_ended_reason,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                }
                climb_records.append(climb_record)

                try:
                    has = learner.has_skill(
                        skill.id,
                        practice_item_pool=self.practice_item_pool,
                        n_problems=2,
                        correctness_threshold=0.5,
                    )
                except Exception as error:
                    logger.warning(
                        "[BaselineMultiConv] has_skill probe failed for %s: %s",
                        skill.id,
                        error,
                    )
                    has = False

                if has:
                    known_skills.add(skill.id)
                    total_skills_learned_in_climb += 1
                    skill_mastered = True
                    logger.info("[BaselineMultiConv] skill=%s MASTERED", skill.id)
                    break

            if not skill_mastered:
                logger.info(
                    "[BaselineMultiConv] skill=%s not mastered after %d attempts",
                    skill.id,
                    self.max_climb_items_per_skill,
                )

            if target_skill_id in known_skills:
                target_mastered = True
                logger.info(
                    "[BaselineMultiConv] target=%s MASTERED after climbing",
                    target_skill_id,
                )
                break

        consolidation_records: list[dict] = []
        consolidation_solutions_found = 0
        if target_mastered and self.consolidation_runs > 0:
            consol_items = self._get_items_for_skill(target_skill_id)
            for consol_idx in range(min(self.consolidation_runs, len(consol_items))):
                item = consol_items[consol_idx]
                used_item_texts.add(item.text)
                session_id = f"consolidation_{target_skill_id}_{consol_idx}_{uuid.uuid4().hex[:8]}"
                conv_result = self._run_single_conversation(
                    learner=learner,
                    practice_item=item,
                    session_id=session_id,
                )
                if conv_result.solution_found:
                    consolidation_solutions_found += 1
                consolidation_records.append(
                    {
                        "phase": "consolidation",
                        "tier": tier,
                        "target_skill": target_skill_id,
                        "consolidation_run": consol_idx + 1,
                        "item_text": item.text[:200],
                        "item_skills": item.associated_skills,
                        "turns": conv_result.num_turns,
                        "solution_found": conv_result.solution_found,
                        "conversation_ended_reason": conv_result.conversation_ended_reason,
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

        total_consolidation_runs = len(consolidation_records)
        consolidation_solution_rate = (
            consolidation_solutions_found / total_consolidation_runs if total_consolidation_runs else 0.0
        )
        avg_turns_per_skill = (
            total_climb_turns / total_skills_learned_in_climb if total_skills_learned_in_climb > 0 else 0.0
        )

        return {
            "run_id": run_id,
            "tier": tier,
            "target_skill": target_skill_id,
            "skill_climb_order": skill_order_ids,
            "target_mastered": target_mastered,
            "total_climb_conversations": len(climb_records),
            "total_climb_turns": total_climb_turns,
            "total_skills_learned_in_climb": total_skills_learned_in_climb,
            "avg_turns_per_skill": avg_turns_per_skill,
            "consolidation_runs_completed": total_consolidation_runs,
            "consolidation_solutions_found": consolidation_solutions_found,
            "consolidation_solution_rate": consolidation_solution_rate,
            "final_known_skills": sorted(known_skills),
            "climb_records": climb_records,
            "consolidation_records": consolidation_records,
            "timestamp": datetime.now().isoformat(),
        }

    def run_all_evaluations(self) -> Path:
        """Run the benchmark for every configured level, target, and repeated run."""
        output_dir = self.output_dir / self.test_run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        root_skill_ids = [skill.id for skill in self.skill_space.get_root_skills()]
        all_results: list[dict[str, Any]] = []

        for level_name, mastered_ids in self.skill_levels.items():
            target_skill_ids = [
                target_skill_id
                for target_skill_id in list(mastered_ids)[: self.max_items]
                if target_skill_id not in root_skill_ids
            ]

            logger.info(
                "[BaselineMultiConv] level=%s runs=%d targets=%d",
                level_name,
                self.runs,
                len(target_skill_ids),
            )

            for run_id in range(self.runs):
                for target_skill_id in target_skill_ids:
                    learner_id = f"bl_multi_conv_{level_name}_{run_id}_{target_skill_id}_{self.test_run_id}"
                    try:
                        learner = _create_learner_for_scenario(
                            self.learner_config,
                            self.skill_space,
                            mastered_skill_ids=root_skill_ids,
                            learner_id=learner_id,
                            practice_conversations_file=(
                                self.practice_conversations_file or output_dir / f"{learner_id}_conversations.jsonl"
                            ),
                            practice_item_pool=self.practice_item_pool,
                            tutor=self.tutor,
                        )
                    except LearnerInitializationError as exc:
                        logger.warning(
                            "[BaselineMultiConv] Skipping learner run "
                            "level=%s run_id=%d target_skill=%s — initialization failed: %s",
                            level_name,
                            run_id,
                            target_skill_id,
                            exc,
                        )
                        all_results.append(
                            {
                                "run_id": run_id,
                                "tier": level_name,
                                "target_skill": target_skill_id,
                                "skill_climb_order": [],
                                "target_mastered": False,
                                "total_climb_conversations": 0,
                                "total_climb_turns": 0,
                                "total_skills_learned_in_climb": 0,
                                "avg_turns_per_skill": 0.0,
                                "consolidation_runs_completed": 0,
                                "consolidation_solutions_found": 0,
                                "consolidation_solution_rate": 0.0,
                                "final_known_skills": [],
                                "climb_records": [],
                                "consolidation_records": [],
                                "initialization_failed": True,
                                "timestamp": datetime.now().isoformat(),
                            },
                        )
                        continue
                    all_results.append(
                        self._run_evaluation_for_target_skill(
                            tier=level_name,
                            target_skill_id=target_skill_id,
                            learner=learner,
                            run_id=run_id,
                        ),
                    )

        total_skills_learned = sum(result["total_skills_learned_in_climb"] for result in all_results)
        total_turns = sum(result["total_climb_turns"] for result in all_results)
        overall_avg_turns_per_skill = total_turns / total_skills_learned if total_skills_learned > 0 else 0.0
        total_consol_runs = sum(result["consolidation_runs_completed"] for result in all_results)
        total_consol_solutions = sum(result["consolidation_solutions_found"] for result in all_results)
        aggregate_metrics = {
            "overall_avg_turns_per_skill": overall_avg_turns_per_skill,
            "overall_consolidation_solution_rate": (
                total_consol_solutions / total_consol_runs if total_consol_runs > 0 else 0.0
            ),
            "total_targets": len(all_results),
            "targets_mastered": sum(1 for result in all_results if result["target_mastered"]),
            "total_climb_conversations": sum(result["total_climb_conversations"] for result in all_results),
            "total_climb_turns": total_turns,
            "total_skills_learned": total_skills_learned,
            "total_consolidation_runs": total_consol_runs,
            "total_consolidation_solutions": total_consol_solutions,
        }

        summary = {
            "test_run_id": self.test_run_id,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "runs": self.runs,
                "skill_levels": {key: list(value) for key, value in self.skill_levels.items()},
                "consolidation_runs": self.consolidation_runs,
                "max_conversation_turns": self.max_conversation_turns,
                "max_climb_items_per_skill": self.max_climb_items_per_skill,
            },
            "aggregate_metrics": aggregate_metrics,
            "per_target_results": all_results,
        }

        output_file = output_dir / "multi_conv_practice_results.json"
        with output_file.open("w", encoding="utf-8") as file_handle:
            json.dump(summary, file_handle, indent=2, default=str)
        return output_file
