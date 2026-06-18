"""Learning-from-conversation benchmark for `BaseLearner` implementations."""

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
from evalconvolearn.core.base_learner import LearnerInitializationError
from evalconvolearn.models.evaluation import LearnerEvalConfig
from evalconvolearn.models.practice_item import PracticeItemPool
from evalconvolearn.models.skill import SkillSpace
from evalconvolearn.utils import load_tutor_responses_mapping

logger = logging.getLogger(__name__)


class BaseLineLearningFromConversationBenchmark:
    """Can the learner learn a skill from a helpful tutor response, and avoid learning from an unhelpful one?"""

    def __init__(
        self,
        skill_space: SkillSpace,
        practice_item_pool: PracticeItemPool,
        learner_config: LearnerEvalConfig,
        skill_levels: dict[str, set[str]],
        tutor_responses: dict[str, dict] | None = None,
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
        self.tutor_responses = tutor_responses or {}
        self.runs = runs
        self.max_items = max_items
        self.output_dir = output_dir or Path("data/benchmark_evaluations/base_learner")
        self.practice_conversations_file = practice_conversations_file
        self.test_run_id = f"bl_learning_{uuid.uuid4().hex[:8]}"
        self.benchmark_extra_args = benchmark_extra_args or {}
        self.max_assessment_turns: int = 1

        for key, value in self.benchmark_extra_args.items():
            if hasattr(self, key):
                setattr(self, key, value)

        if not self.tutor_responses and "mocked_tutor_responses_csv_path" in self.benchmark_extra_args:
            self.tutor_responses = load_tutor_responses_mapping(
                mocked_tutor_responses_csv_path=self.benchmark_extra_args["mocked_tutor_responses_csv_path"],
            )

    def run_all_evaluations(self) -> Path:
        output_dir = self.output_dir / self.test_run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        all_results: list[dict] = []

        for level_name, mastered_ids in self.skill_levels.items():
            mastered_set = set(mastered_ids)
            unmastered_items = _items_for_skill_scenario(
                pool=self.practice_item_pool,
                mastered_ids=mastered_set,
                want_mastered=False,
                max_items=self.max_items,
                skill_space=self.skill_space,
                retrieve_all_learner_skill_prerequisites=True,
                item_prerequisites_should_be_mastered=True,
            )

            logger.info(
                "[BaseLineLearningFromConversationBenchmark] level=%s unmastered_items=%d",
                level_name,
                len(unmastered_items),
            )

            for response_type in ("helpful", "unhelpful"):
                for run_id in range(self.runs):
                    for item_id, item in enumerate(unmastered_items):
                        try:
                            learner = _create_learner_for_scenario(
                                self.learner_config,
                                self.skill_space,
                                list(mastered_ids),
                                learner_id=f"learning_{level_name}_{response_type}_{run_id}_item_{item_id}",
                                practice_conversations_file=self.practice_conversations_file,
                                practice_item_pool=self.practice_item_pool,
                            )
                        except LearnerInitializationError as exc:
                            logger.warning(
                                "[BaseLineLearningFromConversationBenchmark] Skipping learner run "
                                "level=%s response_type=%s run_id=%d item_id=%d — initialization failed: %s",
                                level_name,
                                response_type,
                                run_id,
                                item_id,
                                exc,
                            )
                            all_results.append(
                                {
                                    "benchmark": "BaseLineLearningFromConversationBenchmark",
                                    "level": level_name,
                                    "response_type": response_type,
                                    "run_id": run_id,
                                    "item_text": item.text[:100],
                                    "target_skill": None,
                                    "had_skill_before": None,
                                    "expected_before": False,
                                    "pre_test_aligned": False,
                                    "had_skill_after": None,
                                    "actually_learned": False,
                                    "expect_learned": response_type == "helpful",
                                    "learning_aligned": False,
                                    "expectation_met": False,
                                    "initialization_failed": True,
                                    "timestamp": datetime.now().isoformat(),
                                },
                            )
                            continue

                        target_skill_id = next(
                            (skill_id for skill_id in item.associated_skills if skill_id not in mastered_set),
                            None,
                        )
                        if target_skill_id is None:
                            continue

                        has_before = learner.has_skill(
                            target_skill_id,
                            practice_item_pool=self.practice_item_pool,
                            n_problems=3,
                            correctness_threshold=0.6,
                            max_assessment_turns=self.max_assessment_turns,
                        )

                        opening_history = [
                            {
                                "role": "assistant",
                                "content": f"Let's work on the following problem together: {item.text}",
                            },
                            {
                                "role": "user",
                                "content": "I'm not sure how to solve this problem. Can you help me?",
                            },
                        ]

                        tutor_reply = self._get_tutor_response(item.text, response_type)
                        if not tutor_reply:
                            raise ValueError(
                                "No mocked tutor response found for problem "
                                f"'{item.text[:100]}...' and response type '{response_type}'.",
                            )

                        history = [
                            *opening_history,
                            {"role": "assistant", "content": tutor_reply},
                        ]
                        learner_follow_up = self._get_learner_follow_up(
                            item.text,
                            response_type,
                        )
                        history.append({"role": "user", "content": learner_follow_up})
                        learner.end_conversation(conversation_history=history)

                        has_after = learner.has_skill(
                            target_skill_id,
                            practice_item_pool=self.practice_item_pool,
                            n_problems=3,
                            correctness_threshold=0.6,
                            max_assessment_turns=self.max_assessment_turns,
                        )

                        expected_before = False
                        pre_test_aligned = has_before == expected_before
                        expect_learned = response_type == "helpful"
                        actually_learned = has_after is True and has_before is False
                        learning_aligned = actually_learned == expect_learned
                        expectation_met = pre_test_aligned and learning_aligned

                        all_results.append(
                            {
                                "benchmark": "BaseLineLearningFromConversationBenchmark",
                                "level": level_name,
                                "response_type": response_type,
                                "run_id": run_id,
                                "item_text": item.text[:100],
                                "target_skill": target_skill_id,
                                "had_skill_before": has_before,
                                "expected_before": expected_before,
                                "pre_test_aligned": pre_test_aligned,
                                "had_skill_after": has_after,
                                "actually_learned": actually_learned,
                                "expect_learned": expect_learned,
                                "learning_aligned": learning_aligned,
                                "expectation_met": expectation_met,
                                "timestamp": datetime.now().isoformat(),
                            },
                        )

        output_file = output_dir / "learning_from_conversation_results.jsonl"
        with open(output_file, "w", encoding="utf-8") as file_handle:
            for result in all_results:
                file_handle.write(json.dumps(result) + "\n")

        summary = self._compute_summary(all_results)
        summary_file = output_dir / "learning_from_conversation_summary.json"
        with open(summary_file, "w", encoding="utf-8") as file_handle:
            json.dump(summary, file_handle, indent=2)

        logger.info("Learning-from-conversation results → %s", output_file)
        logger.info("Learning-from-conversation summary → %s", summary_file)
        return output_file

    def _compute_summary(self, all_results: list[dict]) -> dict:
        df = pd.DataFrame(all_results)
        if df.empty:
            return {
                "overall": {},
                "by_response_type": {},
                "by_level": {},
                "by_level_and_response_type": {},
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
        by_response_type = {response_type: pct_met(group) for response_type, group in df.groupby("response_type")}
        by_level = {level: pct_met(group) for level, group in df.groupby("level")}

        by_level_and_response_type: dict[str, dict[str, dict]] = {}
        for (level, response_type), group in df.groupby(["level", "response_type"]):
            by_level_and_response_type.setdefault(level, {})[response_type] = pct_met(
                group,
            )

        return {
            "test_run_id": self.test_run_id,
            "runs": self.runs,
            "timestamp": datetime.now().isoformat(),
            "overall": overall,
            "by_response_type": by_response_type,
            "by_level": by_level,
            "by_level_and_response_type": by_level_and_response_type,
        }

    @staticmethod
    def compute_structured_metrics(output_file: Path) -> dict:
        summary_file = output_file.parent / "learning_from_conversation_summary.json"
        if summary_file.exists():
            with open(summary_file, encoding="utf-8") as file_handle:
                summary = json.load(file_handle)
        else:
            records = []
            with open(output_file, encoding="utf-8") as file_handle:
                for line in file_handle:
                    if line.strip():
                        records.append(json.loads(line))
            by_response_type: dict[str, list[bool]] = {}
            for record in records:
                by_response_type.setdefault(
                    record.get("response_type", "unknown"),
                    [],
                ).append(
                    bool(record.get("expectation_met", False)),
                )
            all_values = [bool(record.get("expectation_met", False)) for record in records]
            breakdowns = {
                key: {
                    "avg_alignment": sum(values) / len(values) if values else 0.0,
                    "n_items": len(values),
                }
                for key, values in by_response_type.items()
            }
            return {
                "metric_type": "alignment",
                "overall_avg_alignment": (sum(all_values) / len(all_values) if all_values else 0.0),
                "total_items": len(all_values),
                "breakdowns": {"by_response_type": breakdowns},
                "breakdown_keys": ["response_type"],
            }

        by_response_type = summary.get("by_response_type", {})
        breakdowns: dict[str, dict] = {}
        for response_type, values in by_response_type.items():
            pct = values.get("expectation_met_pct")
            breakdowns[response_type] = {
                "avg_alignment": (pct / 100.0) if pct is not None else 0.0,
                "n_items": values.get("total", 0),
            }

        overall = summary.get("overall", {})
        overall_pct = overall.get("expectation_met_pct")
        return {
            "metric_type": "alignment",
            "overall_avg_alignment": ((overall_pct / 100.0) if overall_pct is not None else 0.0),
            "total_items": overall.get("total", 0),
            "breakdowns": {"by_response_type": breakdowns},
            "breakdown_keys": ["response_type"],
        }

    def _get_tutor_response(self, problem_text: str, response_type: str) -> str | None:
        data = self.tutor_responses.get(problem_text.strip())
        if data:
            return data.get(f"{response_type}_response", "")
        return None

    def _get_learner_follow_up(self, problem_text: str, response_type: str) -> str:
        data = self.tutor_responses.get(problem_text.strip())
        if data:
            learner_response_key = f"learner_response_{response_type}"
            follow_up = data.get(learner_response_key, "").strip()
            if follow_up:
                return follow_up
        return "Thank you for the explanation!"
