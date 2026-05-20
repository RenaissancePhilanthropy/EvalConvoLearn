"""Placement test benchmark for FlexLearner implementations."""

import json
import logging
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from evalconvolearn.benchmarks.flexlearners.flexlearner_benchmark import (
    FlexLearnerBenchmark,
)
from evalconvolearn.models.binary_skills_flexlearner import (
    BinarySkillsFlexLearner,
    StudentPool,
)
from evalconvolearn.models.evaluation import LearnerEvalConfig
from evalconvolearn.models.placement_test import PlacementTest
from evalconvolearn.models.practice_item import PracticeItemPool
from evalconvolearn.models.skill import SkillSpace
from evalconvolearn.utils import (
    calculate_placement_test_alignment,
    get_placement_test_skill_levels,
    load_tagged_skill_ids,
)


class PlacementTestBenchmark(FlexLearnerBenchmark):
    """Benchmark evaluator for placement tests."""

    def __init__(
        self,
        skill_space: SkillSpace,
        practice_item_pool: PracticeItemPool,
        runs_per_level: int = 1,
        output_dir: Path | None = None,
        learner_pool: StudentPool | None = None,
        learner_config: LearnerEvalConfig | None = None,
        skill_levels: dict[str, set[str]] | None = None,
        benchmark_extra_args: dict | None = None,
    ):
        """Initialize placement test benchmark.

        Args:
        ----
            skill_space: SkillSpace object
            practice_item_pool: PracticeItemPool object
            runs_per_level: Number of runs to execute per learner level
            output_dir: Directory to save results (defaults to data/benchmark_evaluations/)
            learner_pool: Optional StudentPool to use for learner creation.
            learner_config: Optional LearnerEvalConfig describing the evaluated learner.
            skill_levels: Optional dictionary mapping learner levels to sets of skill IDs.

        """
        super().__init__(
            skill_space=skill_space,
            practice_item_pool=practice_item_pool,
            output_dir=output_dir,
            learner_pool=learner_pool,
            learner_config=learner_config,
            benchmark_extra_args=benchmark_extra_args,
        )
        self.logger = logging.getLogger(__name__)
        self.runs_per_level = runs_per_level

        # Resolve and validate skill levels
        self.skill_levels = self._resolve_skill_levels(
            skill_levels or get_placement_test_skill_levels(),
        )
        self._validate_skill_levels(self.skill_levels)

        # Build alignment matrix from the *resolved* skill levels so that a
        # custom "default" level (from mastered_skills in LearnerEvalConfig)
        # is correctly represented rather than defaulting to an empty dict.

        # Expand each level's skills to include all transitive prerequisites,
        # since a learner who has mastered a skill is expected to also know its prerequisites.
        all_skill_ids = load_tagged_skill_ids()
        self.alignment_matrix = {}
        for level, skills in self.skill_levels.items():
            expanded_skills = set(skills)
            for skill_id in skills:
                prerequisite_ids = cast(
                    "list[str]",
                    skill_space.get_all_prerequisites(skill_id, return_as_ids=True),
                )
                expanded_skills.update(prerequisite_ids)
            self.alignment_matrix[level] = {
                "correct": expanded_skills,
                "incorrect": all_skill_ids - expanded_skills,
            }

    @staticmethod
    def compute_structured_metrics(output_file: Path) -> dict:
        """Compute alignment metrics grouped by knowledge_check_mode and expected_correctness."""
        from collections import defaultdict
        from typing import Any

        records = []
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        def _avg(vals: list[float]) -> float | None:
            return sum(vals) / len(vals) if vals else None

        # Per-mode overall alignment rates (one float per run record)
        mode_rates: dict[str, list[float]] = defaultdict(list)
        # Per-mode x per-expected_correctness alignment rates (one float per run record)
        mode_exp_rates: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list),
        )
        # Per-expected_correctness alignment rates across all modes
        exp_rates: dict[str, list[float]] = defaultdict(list)

        for r in records:
            mode = r.get("knowledge_check_mode", "unknown")
            mode_rates[mode].append(r.get("alignment_accuracy", 0.0))

            items = r.get("items", [])
            for exp_bool, exp_key in [
                (True, "expected_correct_true"),
                (False, "expected_correct_false"),
            ]:
                group = [i for i in items if i.get("expected_correct") == exp_bool]
                if group:
                    aligned = sum(1 for i in group if i.get("is_aligned_to_matrix"))
                    rate = aligned / len(group)
                    mode_exp_rates[mode][exp_key].append(rate)
                    exp_rates[exp_key].append(rate)

        kc_breakdown: dict[str, Any] = {
            mode: {"avg_alignment": _avg(rates), "n_runs": len(rates)}
            for mode, rates in mode_rates.items()
        }

        exp_breakdown: dict[str, Any] = {
            exp_key: {"avg_alignment": _avg(rates), "n_runs": len(rates)}
            for exp_key, rates in exp_rates.items()
        }

        cross_breakdown: dict[str, Any] = {
            f"{mode}__{exp_key}": {"avg_alignment": _avg(rates), "n_runs": len(rates)}
            for mode, exp_dict in mode_exp_rates.items()
            for exp_key, rates in exp_dict.items()
        }

        all_vals = [r.get("alignment_accuracy", 0.0) for r in records]
        return {
            "breakdowns": {
                "knowledge_check_mode": kc_breakdown,
                "expected_correctness": exp_breakdown,
                "knowledge_check_x_expected_correctness": cross_breakdown,
            },
            "overall_avg_alignment": _avg(all_vals),
            "total_items": len(all_vals),
            "metric_type": "alignment",
            "breakdown_keys": ["knowledge_check_mode", "expected_correctness"],
        }

    def run_evaluation_for_level(
        self,
        learner_level: str,
        run_id: int,
        tmp_dir: Path,
        check_if_has_knowledge_before_answering: bool = True,
    ) -> dict:
        """Run placement test evaluation for a single learner level and run.

        Args:
        ----
            learner_level: Learner proficiency level (beginner, intermediate, expert)
            run_id: Run identifier
            tmp_dir: Temporary directory for practice files
            check_if_has_knowledge_before_answering: Whether to check if the learner
                has sufficient knowledge before generating an answer. When True, uses
                skill-binary check for Learner instances and LLM-based knowledge
                sufficiency check for other FlexLearner implementations.

        Returns:
        -------
            dict with evaluation results

        """
        mode_suffix = "wkc" if check_if_has_knowledge_before_answering else "nokc"
        if self.learner_pool is not None and self.learner_config is not None:
            learner_id = f"{self.learner_pool.id}_learner_PlacementTestBenchmark_{learner_level}_{run_id}_{mode_suffix}"
            init_kwargs = self.learner_config.init_knowledge_kwargs or {}
            learner = self.learner_pool.create_learner(
                learner_id=learner_id,
                mastered_skills=list(set(self.skill_levels.get(learner_level, []))),
                skill_space=self.skill_space,
                **init_kwargs,
            )
        else:
            practice_file = tmp_dir / f"{learner_level}_{run_id}_{mode_suffix}.jsonl"
            practice_file.touch()
            learner = BinarySkillsFlexLearner(
                id=f"placement_test_{learner_level}_learner_{mode_suffix}",
                mastered_skills=self.skill_levels[learner_level].copy(),
                skill_space=self.skill_space,
                practice_conversations_file=practice_file,
            )

        # Create placement test
        placement_test = PlacementTest(practice_item_pool=self.practice_item_pool)
        placement_test.clear_results()

        # Get all unique skills from practice item pool
        placement_test_skill_ids = sorted(
            {
                skill_id
                for item in self.practice_item_pool.items
                for skill_id in item.associated_skills
            },
        )

        # Administer test for each skill
        results = []
        for skill_id in placement_test_skill_ids:
            matching_items = [
                item
                for item in self.practice_item_pool.items
                if skill_id in item.associated_skills
            ]
            if not matching_items:
                raise ValueError(f"No practice items found for skill ID {skill_id}.")

            result = placement_test.administer_item(
                matching_items[0],
                learner=learner,
                use_llm_for_answer=True,
                use_multiple_choice=False,
                check_if_has_knowledge_before_answering=check_if_has_knowledge_before_answering,
            )
            results.append(result)

        if not results:
            raise ValueError("No placement test results returned.")

        # Calculate metrics
        summary = placement_test.get_test_summary()
        alignment_correct = 0
        alignment_evaluated = 0
        per_item_results = []

        matrix = self.alignment_matrix.get(learner_level, {})
        expected_correct_skills = set(matrix.get("correct", set()))
        expected_incorrect_skills = set(matrix.get("incorrect", set()))

        for result in placement_test.test_results:
            skill_id = (
                result.practice_item.associated_skills[0]
                if result.practice_item.associated_skills
                else None
            )

            expected_correct, is_aligned = calculate_placement_test_alignment(
                result,
                expected_correct_skills,
                expected_incorrect_skills,
            )

            if is_aligned is not None:
                alignment_evaluated += 1
                if is_aligned:
                    alignment_correct += 1

            # Calculate effective correctness
            effective_is_correct = result.is_correct
            if (
                result.answer_choices
                and result.learner_choice_index is not None
                and 0 <= result.learner_choice_index < len(result.answer_choices)
            ):
                selected_answer = result.answer_choices[result.learner_choice_index]
                effective_is_correct = selected_answer == result.correct_answer

            per_item_results.append(
                {
                    "skill": skill_id,
                    "expected_correct": expected_correct,
                    "is_correct": result.is_correct,
                    "effective_is_correct": effective_is_correct,
                    "learner_has_all_required_skills": result.learner_has_all_required_skills,
                    "is_aligned_to_matrix": is_aligned,
                    "required_skills": result.required_skills,
                    "learner_answer": result.learner_answer,
                    "correct_answer_letter": result.correct_answer_letter,
                    "prompt_content": result.prompt_content,
                },
            )

        alignment_accuracy = (
            alignment_correct / alignment_evaluated if alignment_evaluated else 0.0
        )

        self.logger.info(
            "[%s run %d] alignment_accuracy=%.2f, test_score=%.2f",
            learner_level,
            run_id,
            alignment_accuracy,
            summary["accuracy"],
        )

        return {
            "test_name": "placement_test_skill_alignment",
            "timestamp": datetime.now().isoformat(),
            "learner_level": learner_level,
            "run_id": run_id,
            "runs_per_level": self.runs_per_level,
            "knowledge_check_mode": (
                "with_knowledge_check"
                if check_if_has_knowledge_before_answering
                else "without_knowledge_check"
            ),
            "mastered_skills": learner.mastered_skills,
            "summary": summary,
            "alignment_accuracy": alignment_accuracy,
            "alignment_evaluated": alignment_evaluated,
            "items": per_item_results,
        }

    def run_all_evaluations(self) -> Path:
        """Run placement test evaluations for all learner levels and runs.

        Runs **two passes** for each level × run combination:

        1. ``with_knowledge_check`` — ``check_if_has_knowledge_before_answering=True``
           The learner's knowledge is assessed first (skill-binary for ``BinarySkillsFlexLearner``,
           LLM-based sufficiency for other ``FlexLearner`` subclasses) and the
           prompt is conditioned on the result.
        2. ``without_knowledge_check`` — ``check_if_has_knowledge_before_answering=False``
           The learner directly produces a response without any prior knowledge
           assessment.

        All runs use open-ended LLM response generation (no answer choices).

        Returns
        -------
            Path to output file

        """
        output_file = (
            self.output_dir
            / f"placement_test_skill_alignment_{self.runs_per_level}runs.jsonl"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        all_levels = list(self.skill_levels.keys())
        knowledge_check_modes = (
            self.knowledge_check_modes
            if hasattr(self, "knowledge_check_modes") and self.knowledge_check_modes
            else [
                (True, "with_knowledge_check"),
                (False, "without_knowledge_check"),
            ]
        )
        total_runs = len(all_levels) * self.runs_per_level * len(knowledge_check_modes)
        print(
            f"[PlacementTestBenchmark] Starting "
            f"-- {len(all_levels)} level(s) x {self.runs_per_level} run(s) "
            f"x {len(knowledge_check_modes)} mode(s) = {total_runs} total run(s)",
        )

        with TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)

            with open(output_file, "w", encoding="utf-8") as f:
                completed = 0
                for check_knowledge, mode_label in knowledge_check_modes:
                    for learner_level in all_levels:
                        for run_id in range(self.runs_per_level):
                            completed += 1
                            print(
                                f"[PlacementTestBenchmark] Run {completed}/{total_runs}"
                                f" -- level={learner_level}, run_id={run_id},"
                                f" mode={mode_label}",
                            )
                            self.logger.info(
                                "Running placement test for %s (run %d/%d, mode=%s)",
                                learner_level,
                                run_id + 1,
                                self.runs_per_level,
                                mode_label,
                            )
                            result = self.run_evaluation_for_level(
                                learner_level,
                                run_id,
                                tmp_dir,
                                check_if_has_knowledge_before_answering=check_knowledge,
                            )
                            f.write(json.dumps(result) + "\n")

        self.logger.info("Results saved to: %s", output_file)
        print(f"[PlacementTestBenchmark] Done — results saved to {output_file}")
        return output_file
