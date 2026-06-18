"""Learning-from-conversation benchmark for FlexLearner implementations."""

import json
import logging
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from evalconvolearn.benchmarks.flexlearners.flexlearner_benchmark import (
    FlexLearnerBenchmark,
)
from evalconvolearn.core.flexlearner import FlexLearner
from evalconvolearn.models.binary_skills_flexlearner import (
    BinarySkillsFlexLearner,
    StudentPool,
)
from evalconvolearn.models.evaluation import LearnerEvalConfig
from evalconvolearn.models.practice_item import PracticeItem, PracticeItemPool
from evalconvolearn.models.skill import SkillSpace
from evalconvolearn.utils import (
    generate_learning_alignment_matrix,
    get_placement_test_skill_levels,
    load_tutor_responses_mapping,
)


class LearningFromConversationBenchmark(FlexLearnerBenchmark):
    """Benchmark evaluator for learning from conversation."""

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
        max_items: int = 10,
    ) -> None:
        """Initialize learning from conversation benchmark.

        Args:
        ----
            skill_space: SkillSpace object
            practice_item_pool: PracticeItemPool object
            runs_per_level: Number of runs to execute per learner level
            output_dir: Directory to save results (defaults to data/benchmark_evaluations/)
            learner_pool: Optional StudentPool to use for learner creation.
            learner_config: Optional LearnerEvalConfig describing the learner.
            skill_levels: Optional dictionary mapping learner levels to sets of skill IDs.
            benchmark_extra_args: Optional dict of benchmark-specific extra arguments (e.g. paths to mocked
            max_items: Maximum number of items to evaluate per learner level. Items are selected
                using the same near-mastery-boundary logic as the base-learner benchmarks —
                only items whose direct skill prerequisites are mastered (i.e. items the
                learner is ready to learn next) are included.

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
        self.max_items = self.benchmark_extra_args.get("max_items", max_items)

        # Resolve and validate skill levels
        self.skill_levels = self._resolve_skill_levels(
            skill_levels or get_placement_test_skill_levels(),
        )
        self._validate_skill_levels(self.skill_levels)

        self.alignment_matrix = generate_learning_alignment_matrix(
            skill_space,
            practice_item_pool,
            placement_test_skill_levels=self.skill_levels,
        )
        # Whether to use pre/post placement tests instead of skill-set
        # alignment checks.  When True, a pre-test checks the learner can /
        # cannot answer the practice item *before* the conversation, then a
        # post-test checks again *after*.  Alignment is determined by
        # comparing pre/post correctness with the expected learning outcome.
        self.evaluate_learning_with_pre_post_tests: bool = bool(
            self.benchmark_extra_args.get(
                "evaluate_learning_with_pre_post_tests",
                False,
            ),
        )

        # check_if_should_learn modes: list of (bool, label) pairs.
        # When the learner is the default skill-binary Learner, the guardrail
        # must always be active (True only).  For custom SimulatedLearner
        # subclasses the caller may request both passes to compare outcomes.
        # Falls back to [(True, "with_skill_check")] when not specified.
        learner_class = self.learner_config.learner_class if self.learner_config else None
        _is_skill_learner = learner_class is not None and learner_class is BinarySkillsFlexLearner
        _default_modes: list[tuple[bool, str]] = (
            [(True, "with_skill_check")]
            if _is_skill_learner
            else [(True, "with_skill_check"), (False, "without_skill_check")]
        )
        raw_modes = self.benchmark_extra_args.get(
            "check_if_should_learn_modes",
            _default_modes,
        )
        # For the base Learner, silently restrict to True-only to avoid
        # producing meaningless results (knowledge update is a no-op).
        if _is_skill_learner:
            self.check_if_should_learn_modes: list[tuple[bool, str]] = [
                (flag, label) for flag, label in raw_modes if flag
            ] or [(True, "with_skill_check")]
        else:
            self.check_if_should_learn_modes = list(raw_modes)

        self.tutor_responses = load_tutor_responses_mapping(
            mocked_tutor_responses_csv_path=self.benchmark_extra_args.get(
                "mocked_tutor_responses_csv_path",
                None,
            ),
        )

    # ------------------------------------------------------------------
    # Helper: reuse PlacementTest._can_learner_answer_correctly logic
    # ------------------------------------------------------------------

    @staticmethod
    def compute_structured_metrics(output_file: Path) -> dict:
        """Compute alignment metrics grouped by check_mode × response_type × evaluation_mode."""
        from collections import defaultdict
        from typing import Any

        records = []
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        by_check_mode: dict[str, list[float]] = defaultdict(list)
        by_response_type: dict[str, list[float]] = defaultdict(list)
        by_combo: dict[str, list[float]] = defaultdict(list)

        for r in records:
            mode = r.get("check_mode", "unknown")
            resp = r.get("response_type", "unknown")
            eval_mode = r.get("evaluation_mode", "skill_alignment")
            acc = r.get("alignment_accuracy", 0.0)
            by_check_mode[mode].append(acc)
            by_response_type[resp].append(acc)
            by_combo[f"{mode}__{resp}__{eval_mode}"].append(acc)

        def _avg(vals: list[float]) -> dict:
            return {
                "avg_alignment": sum(vals) / len(vals) if vals else 0.0,
                "n_items": len(vals),
            }

        metrics: dict[str, Any] = {
            "breakdowns": {
                "by_check_mode": {k: _avg(v) for k, v in by_check_mode.items()},
                "by_response_type": {k: _avg(v) for k, v in by_response_type.items()},
                "by_check_mode_x_response_type_x_eval_mode": {k: _avg(v) for k, v in by_combo.items()},
            },
        }
        all_vals = [r.get("alignment_accuracy", 0.0) for r in records]
        metrics["overall_avg_alignment"] = sum(all_vals) / len(all_vals) if all_vals else 0.0
        metrics["total_items"] = len(all_vals)
        metrics["metric_type"] = "alignment"
        metrics["breakdown_keys"] = ["check_mode", "response_type", "evaluation_mode"]
        return metrics

    def _check_learner_can_answer(
        self,
        practice_item: PracticeItem,
        learner: FlexLearner,
    ) -> bool:
        """Check whether *learner* can answer *practice_item* correctly.

        Delegates to :pymethod:`PlacementTest._can_learner_answer_correctly`
        which handles both skill-binary (``BinarySkillsFlexLearner``) and LLM-based
        knowledge-sufficiency checks (custom ``FlexLearner`` subclasses).
        """
        from evalconvolearn.models.placement_test import PlacementTest

        pt = PlacementTest(practice_item_pool=self.practice_item_pool)
        return pt._can_learner_answer_correctly(practice_item, learner)

    def run_evaluation_for_level(
        self,
        learner_level: str,
        response_type: str,
        run_id: int,
        tmp_dir: Path,
        check_if_should_learn: bool = True,
        check_mode_label: str = "with_skill_check",
        items_to_evaluate: list[PracticeItem] | None = None,
    ) -> list[dict]:
        """Run learning from conversation evaluation for a learner level with a response type.

        Args:
        ----
            learner_level: Learner proficiency level (beginner, intermediate, expert)
            response_type: Type of tutor response ("helpful" or "unhelpful")
            run_id: Run identifier
            tmp_dir: Temporary directory for practice files
            check_if_should_learn: Whether to use the skill-guardrail LLM check
                before mastering skills.  When False the learner only updates its
                knowledge representation without mastering any skills.
            check_mode_label: Human-readable label for this mode (recorded in results).
            items_to_evaluate: Subset of practice items to evaluate. Defaults to all
                items in the pool when None.

        Returns:
        -------
            list of result dicts, one per practice item

        """
        mastered_skills = self.skill_levels[learner_level].copy()

        # Get alignment expectations for this learner level
        level_matrix = self.alignment_matrix.get(learner_level, {})

        # Check if we have the tutor responses loaded
        if not self.tutor_responses:
            self.logger.error("No tutor responses found in CSV. Cannot run evaluation.")
            return []

        self.logger.debug(
            f"Using {response_type} responses for {len(self.practice_item_pool.items)} items",
        )

        results = []
        total_alignment_accuracy = 0
        total_items_evaluated = 0

        items = items_to_evaluate if items_to_evaluate is not None else self.practice_item_pool.items

        # Test each practice item
        for idx, practice_item in enumerate(items, start=1):
            problem_text = practice_item.text
            item_skills = practice_item.associated_skills

            # Get the tutor response for this item
            problem_text_key = problem_text.strip()
            tutor_data = self.tutor_responses.get(problem_text_key)
            if not tutor_data:
                self.logger.warning(
                    f"Skipping item {idx}: No tutor response found for problem: {problem_text}\n -- with skills {item_skills}",
                )
                continue

            # Extract the appropriate response type (helpful or unhelpful)
            response_key = f"{response_type}_response"
            tutor_response_content = tutor_data.get(response_key, "")
            if not tutor_response_content:
                self.logger.warning(
                    f"Skipping item {idx}: No {response_type} response found",
                )
                continue

            tutor_response_data = {
                "content": tutor_response_content,
                "skill_focus": [tutor_data.get("skill_id", "")],
            }
            if self.learner_pool is not None and self.learner_config is not None:
                learner_id = f"{self.learner_pool.id}_learner_LearningFromConversationBenchmark_{learner_level}_{response_type}_{check_mode_label}_item_{idx}_run_{run_id}"
                init_kwargs = self.learner_config.init_knowledge_kwargs or {}
                learner = self.learner_pool.create_learner(
                    learner_id=learner_id,
                    mastered_skills=list(set(self.skill_levels.get(learner_level, []))),
                    skill_space=self.skill_space,
                    **init_kwargs,
                )
            else:
                # Create temporary practice file for this learner
                practice_file = tmp_dir / f"{learner_level}_{response_type}_item_{idx}_run_{run_id}.jsonl"
                practice_file.touch()

                # Create learner with appropriate skill level
                learner = BinarySkillsFlexLearner(
                    id=f"learner_{learner_level}_item_{idx}",
                    mastered_skills=mastered_skills.copy(),
                    skill_space=self.skill_space,
                    practice_conversations_file=practice_file,
                )

            learnable_skills = learner.get_learnable_skills()
            learnable_skills_ids = [sk.id for sk in learnable_skills]

            # Build expected learning based on matrix and response type
            expected_learning = []
            for skill_id in item_skills:
                skill_expectations = level_matrix.get(skill_id, {})
                # For helpful responses, check if learner should learn
                # For unhelpful responses, learner should NOT learn
                if response_type == "helpful":
                    should_learn = skill_expectations.get("helpful_response", False)
                else:
                    should_learn = skill_expectations.get("unhelpful_response", False)

                if should_learn:
                    expected_learning.append(skill_id)

            # ----------------------------------------------------------
            # PRE-TEST (only in pre/post test mode)
            # ----------------------------------------------------------
            pre_test_can_answer: bool | None = None
            if self.evaluate_learning_with_pre_post_tests:
                try:
                    pre_test_can_answer = self._check_learner_can_answer(
                        practice_item,
                        learner,
                    )
                except Exception as e:
                    self.logger.error("[Pre-test] Error for item %d: %s", idx, e)
                    pre_test_can_answer = None

            # Build mock conversation
            learner_response_key = f"learner_response_{response_type}"
            learner_follow_up = tutor_data.get(learner_response_key, "").strip() or "Thank you for the explanation!"

            # mock conversation has adapted learner responses:
            mock_conversation = [
                {
                    "role": "system",
                    "content": f"Practice Item: {practice_item.text}",
                },
                {
                    "role": "user",
                    "content": "I'm not sure how to solve this problem. Can you help me?",
                },
                {
                    "role": "assistant",
                    "content": tutor_response_data["content"],
                },
                {
                    "role": "user",
                    "content": learner_follow_up,
                },
            ]

            try:
                item_skills_objs = [self.skill_space[sk_id] for sk_id in item_skills if sk_id in self.skill_space]

                learned_skills = learner.learns_from_conversation(
                    dialogue_history=mock_conversation,
                    item_skills=item_skills_objs,
                    correct_answer="",
                    check_if_should_learn=check_if_should_learn,
                )

                learned_skills_ids = [sk.id for sk in learned_skills] if learned_skills else []

            except Exception as e:
                self.logger.error("Error processing item %d: %s", idx, e)
                learned_skills_ids = []

            # ----------------------------------------------------------
            # POST-TEST (only in pre/post test mode)
            # ----------------------------------------------------------
            post_test_can_answer: bool | None = None
            if self.evaluate_learning_with_pre_post_tests:
                try:
                    post_test_can_answer = self._check_learner_can_answer(
                        practice_item,
                        learner,
                    )
                except Exception as e:
                    self.logger.error("[Post-test] Error for item %d: %s", idx, e)
                    post_test_can_answer = None

            # ----------------------------------------------------------
            # Alignment accuracy
            # ----------------------------------------------------------
            pre_test_aligned: bool = False
            actually_learned: bool = False
            expected_learned: bool = False
            learning_aligned: bool = False
            if self.evaluate_learning_with_pre_post_tests:
                # Pre/post-test mode:
                # 1. pre_test must align with the learner's initial mastered
                #    skills (pre_test_can_answer should be False for skills
                #    that the learner has NOT mastered, True for mastered).
                # 2. Learning outcome is inferred from pre/post change:
                #    - learned = (post_test True AND pre_test False)
                #    - not learned = all other combinations
                # 3. The inferred learning must match the expected learning
                #    from the alignment matrix / response type.
                #
                # Determine expected pre-test result: the learner should be
                # able to answer iff it has mastered ALL required skills
                # (including prerequisites) for the item.
                from evalconvolearn.models.placement_test import PlacementTest

                _pt = PlacementTest(practice_item_pool=self.practice_item_pool)
                required_skill_ids = {s.id for s in _pt._get_all_required_skills_for_item(practice_item)}
                expected_pre_test = required_skill_ids.issubset(set(mastered_skills))

                pre_test_aligned = (
                    (pre_test_can_answer == expected_pre_test) if pre_test_can_answer is not None else False
                )

                # Infer actual learning from pre/post tests
                actually_learned = post_test_can_answer is True and pre_test_can_answer is False
                # Expected learning: should have learned something?
                expected_learned = len(expected_learning) > 0

                learning_aligned = actually_learned == expected_learned

                item_alignment_accuracy = 1.0 if (pre_test_aligned and learning_aligned) else 0.0

                self.logger.debug(
                    "[Pre/Post] item %d — pre=%s post=%s pre_aligned=%s actually_learned=%s expected_learned=%s alignment=%.1f",
                    idx,
                    pre_test_can_answer,
                    post_test_can_answer,
                    pre_test_aligned,
                    actually_learned,
                    expected_learned,
                    item_alignment_accuracy,
                )
            else:
                # Original skill-set alignment mode
                expected_set = set(expected_learning)
                learned_set = set(learned_skills_ids)
                item_alignment_accuracy = 1.0 if expected_set == learned_set else 0.0

            total_alignment_accuracy += item_alignment_accuracy
            total_items_evaluated += 1

            # Record per-item result
            result = {
                "learner_level": learner_level,
                "response_type": response_type,
                "check_mode": check_mode_label,
                "check_if_should_learn": check_if_should_learn,
                "item_index": idx,
                "problem": problem_text[:200],
                "item_skills": item_skills,
                "mastered_skills": mastered_skills,
                "learnable_skills": learnable_skills_ids,
                "expected_skills": expected_learning,
                "learned_skills": learned_skills_ids,
                "alignment_accuracy": item_alignment_accuracy,
                "evaluation_mode": (
                    "pre_post_test" if self.evaluate_learning_with_pre_post_tests else "skill_alignment"
                ),
                "timestamp": datetime.now().isoformat(),
            }
            # Add pre/post test details when in that mode
            if self.evaluate_learning_with_pre_post_tests:
                result["pre_test_can_answer"] = pre_test_can_answer
                result["post_test_can_answer"] = post_test_can_answer
                result["pre_test_aligned"] = pre_test_aligned
                result["actually_learned"] = actually_learned
                result["expected_learned"] = expected_learned
                result["learning_aligned"] = learning_aligned
            results.append(result)

        avg_alignment = total_alignment_accuracy / total_items_evaluated if total_items_evaluated > 0 else 0.0
        self.logger.info(
            "[%s / %s] items=%d avg_alignment=%.2f",
            learner_level,
            response_type,
            total_items_evaluated,
            avg_alignment,
        )
        return results

    def run_all_evaluations(self) -> Path:
        """Run learning from conversation evaluations for all learner levels and response types.

        Returns
        -------
            Path to output file

        """
        output_file = self.output_dir / f"learning_from_conversation_{self.runs_per_level}runs.jsonl"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        all_levels = list(self.skill_levels.keys())
        response_types = ["helpful", "unhelpful"]
        check_modes = self.check_if_should_learn_modes  # list of (bool, label)
        total_runs = len(all_levels) * len(response_types) * len(check_modes) * self.runs_per_level
        print(
            f"[LearningFromConversationBenchmark] Starting"
            f" -- {len(all_levels)} level(s) x {len(response_types)} response type(s)"
            f" x {len(check_modes)} skill-check mode(s)"
            f" x {self.runs_per_level} run(s) = {total_runs} total run(s)",
        )
        for flag, label in check_modes:
            print(f"    skill-check mode: {label!r} (check_if_should_learn={flag})")

        with TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)

            with open(output_file, "w", encoding="utf-8") as f:
                completed = 0
                for learner_level in all_levels:
                    # Select items whose prerequisites are mastered for this level
                    level_items = self.practice_item_pool.get_items_for_skill_scenario(
                        mastered_ids=self.skill_levels[learner_level],
                        want_mastered=False,
                        max_items=self.max_items,
                        retrieve_all_learner_skill_prerequisites=True,
                        item_prerequisites_should_be_mastered=True,
                    )
                    print(
                        f"[LearningFromConversationBenchmark] level={learner_level}"
                        f" — {len(level_items)} item(s) selected (max_items={self.max_items})",
                    )
                    if not level_items:
                        self.logger.warning(
                            "No items found for level %s — skipping",
                            learner_level,
                        )
                        continue

                    # Run through all items with helpful responses, then all items with unhelpful responses
                    for response_type in response_types:
                        for check_flag, check_label in check_modes:
                            # Cannot have self.evaluate_learning_with_pre_post_tests=False and check_if_should_learn=False
                            # because for a FlexLearner with custom knowledge, check if should learn will not
                            # update the list of mastered_skills, which is used to evaluate learning when pre/post test is deactivated.
                            if not self.evaluate_learning_with_pre_post_tests and not check_flag:
                                continue

                            for run_id in range(self.runs_per_level):
                                completed += 1
                                print(
                                    f"[LearningFromConversationBenchmark] Run {completed}/{total_runs}"
                                    f" -- level={learner_level}, response_type={response_type},"
                                    f" check_mode={check_label}, run_id={run_id}",
                                )
                                self.logger.info(
                                    "Running %s - %s - %s (run %d/%d)",
                                    learner_level,
                                    response_type,
                                    check_label,
                                    run_id + 1,
                                    self.runs_per_level,
                                )
                                results = self.run_evaluation_for_level(
                                    learner_level,
                                    response_type,
                                    run_id,
                                    tmp_dir,
                                    check_if_should_learn=check_flag,
                                    check_mode_label=check_label,
                                    items_to_evaluate=level_items,
                                )

                                # Write results
                                for result in results:
                                    run_record = {
                                        "test_name": "learning_from_conversation",
                                        "run_id": run_id,
                                        "runs_per_level": self.runs_per_level,
                                        **result,
                                    }
                                    f.write(json.dumps(run_record) + "\n")

        self.logger.info("Results saved to: %s", output_file)
        print(
            f"[LearningFromConversationBenchmark] Done — results saved to {output_file}",
        )
        return output_file
