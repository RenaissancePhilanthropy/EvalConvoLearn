"""Conversation-metrics benchmark fitted to real tutoring-conversation datasets."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from evalconvolearn.benchmarks.base_learners.base_learner_benchmarks import (
    _create_learner_for_scenario,
)
from evalconvolearn.benchmarks.realistic_benchmarks_from_conversation_data.dataset_fitted_benchmark_utils import (
    DEFAULT_CONVERSATIONS_JSONL,
    counts_to_distribution,
    distribution_distance,
    extract_learner_turns,
    normalize_dialogue_history,
    select_top_skills_by_count,
)
from evalconvolearn.core.base_learner import LearnerInitializationError
from evalconvolearn.models.base_learner_conversation import (
    run_base_learner_conversation,
)
from evalconvolearn.models.evaluation import LearnerEvalConfig
from evalconvolearn.models.practice_item import PracticeItem, PracticeItemPool
from evalconvolearn.models.skill import SkillSpace
from evalconvolearn.models.tutor import Tutor
from evalconvolearn.utils.llm_evaluator import (
    ConversationBehaviorLabelsVerdict,
    ConversationErrorLabels,
    ConversationTalkMoveLabels,
    classify_conversation_behaviors,
)

logger = logging.getLogger(__name__)

MASTERY_GROUP_MASTERED = "mastered"
MASTERY_GROUP_UNMASTERED = "unmastered"
MASTERY_GROUPS = [MASTERY_GROUP_MASTERED, MASTERY_GROUP_UNMASTERED]

# Scenario keys: (target_skill_mastered_before, prerequisites_mastered)
SCENARIO_MASTERED_PREREQS_MET = "skill_mastered__prereqs_met"
SCENARIO_MASTERED_PREREQS_NOT_MET = "skill_mastered__prereqs_not_met"
SCENARIO_UNMASTERED_PREREQS_MET = "skill_unmastered__prereqs_met"
SCENARIO_UNMASTERED_PREREQS_NOT_MET = "skill_unmastered__prereqs_not_met"
SCENARIOS = [
    SCENARIO_MASTERED_PREREQS_MET,
    SCENARIO_MASTERED_PREREQS_NOT_MET,
    SCENARIO_UNMASTERED_PREREQS_MET,
    SCENARIO_UNMASTERED_PREREQS_NOT_MET,
]

ERROR_BUCKETS = ["zero", "one", "multiple"]
BINARY_BUCKETS = ["no", "yes"]

# Weight of learning-behavior vs conversational metrics in the final composite score
EVAL_CONVO_LEARN_ALPHA = 0.6
WORD_PATTERN = re.compile(r"\b[\w'-]+\b")

ERROR_TYPE_FIELDS = {
    "NC": "numerical_calculation",
    "CU": "conceptual_understanding",
    "PC": "problem_comprehension",
    "SD": "strategic_decision",
    "SO": "step_omission",
}

TALK_MOVE_FIELDS = {
    "asking_for_more_information": "asking_for_more_information",
    "making_a_claim": "making_a_claim",
    "providing_evidence_or_reasoning": "providing_evidence_or_reasoning",
}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _jsd_log2(p_vec: list[float], q_vec: list[float]) -> float:
    """Jensen-Shannon divergence between two non-negative vectors, log base 2, bounded [0, 1].

    Each vector is treated as an unnormalized probability distribution over the same
    finite label set (positions match).  Returns 0.0 whenever either vector is empty
    or all-zero.
    """
    if not p_vec or not q_vec or len(p_vec) != len(q_vec):
        return 0.0
    p_sum = sum(p_vec)
    q_sum = sum(q_vec)
    if p_sum <= 0 or q_sum <= 0:
        # At least one distribution is degenerate – treat as maximally distant.
        return 1.0 if p_sum != q_sum else 0.0
    p = [x / p_sum for x in p_vec]
    q = [x / q_sum for x in q_vec]
    m = [(p[i] + q[i]) / 2.0 for i in range(len(p))]

    def _kl(a: list[float], b: list[float]) -> float:
        total = 0.0
        for ai, bi in zip(a, b, strict=False):
            if ai > 0.0 and bi > 0.0:
                total += ai * math.log2(ai / bi)
        return total

    jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return max(0.0, min(1.0, jsd))


def _wasserstein1_normalized(
    real_vals: list[float],
    sim_vals: list[float],
    val_range: float | None = None,
) -> float:
    """Wasserstein-1 (Earth Mover's Distance) between two empirical distributions,
    normalized to [0, 1].

    Each input is a list of scalar observations.
    ``val_range`` fixes the normalization denominator; when *None* the joint
    range (real + sim) is used.  Pass the range observed in the real dataset
    alone to avoid normalization drift from the simulated distribution.
    """
    if not real_vals or not sim_vals:
        return 0.0

    if val_range is None:
        all_vals = real_vals + sim_vals
        computed_range = max(all_vals) - min(all_vals)
        val_range = computed_range if computed_range > 0.0 else 0.0
    if val_range <= 0.0:
        return 0.0

    n = len(real_vals)
    m = len(sim_vals)
    real_sorted = sorted(real_vals)
    sim_sorted = sorted(sim_vals)

    if n == m:
        # Closed-form W1 for equal-length empirical distributions
        w1 = sum(abs(real_sorted[i] - sim_sorted[i]) for i in range(n)) / n
    else:
        # Numerical integration of |CDF_real(x) - CDF_sim(x)|
        all_points = sorted(set(real_vals + sim_vals))
        w1 = 0.0
        for i in range(len(all_points) - 1):
            x0, x1 = all_points[i], all_points[i + 1]
            cdf_real = sum(1.0 for v in real_sorted if v <= x0) / n
            cdf_sim = sum(1.0 for v in sim_sorted if v <= x0) / m
            w1 += abs(cdf_real - cdf_sim) * (x1 - x0)

    return max(0.0, min(1.0, w1 / val_range))


def _count_words(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def _group_key(skill_id: str, mastery_group: str) -> str:
    return f"{skill_id}::{mastery_group}"


def _scenario_key(skill_id: str, scenario: str) -> str:
    return f"{skill_id}::{scenario}"


def _compute_scenario(
    target_skill_mastered_before: bool,
    prerequisites_mastered: bool,
) -> str:
    if target_skill_mastered_before and prerequisites_mastered:
        return SCENARIO_MASTERED_PREREQS_MET
    if target_skill_mastered_before and not prerequisites_mastered:
        return SCENARIO_MASTERED_PREREQS_NOT_MET
    if not target_skill_mastered_before and prerequisites_mastered:
        return SCENARIO_UNMASTERED_PREREQS_MET
    return SCENARIO_UNMASTERED_PREREQS_NOT_MET


def _fallback_conversation_behavior_verdict(
    reason: str,
) -> ConversationBehaviorLabelsVerdict:
    return ConversationBehaviorLabelsVerdict(
        reasoning=reason,
        errors=ConversationErrorLabels(),
        talk_moves=ConversationTalkMoveLabels(),
    )


def conversation_metrics_cache_key(
    *,
    dialogue_history: list[dict[str, str]],
    problem_text: str,
    correct_answer: str,
    cache_namespace: str,
    classification_model: str,
) -> str:
    canonical = json.dumps(
        {
            "cache_namespace": cache_namespace,
            "classification_model": classification_model,
            "problem_text": problem_text,
            "correct_answer": correct_answer,
            "dialogue_history": dialogue_history,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def classify_conversation_with_cache(
    *,
    dialogue_history: Any,
    problem_text: str,
    correct_answer: str,
    cache_namespace: str,
    classification_model: str,
    cache: dict[str, dict[str, Any]] | None = None,
) -> ConversationBehaviorLabelsVerdict:
    normalized_dialogue = normalize_dialogue_history(dialogue_history)
    effective_cache = cache if cache is not None else {}
    cache_key = conversation_metrics_cache_key(
        dialogue_history=normalized_dialogue,
        problem_text=problem_text,
        correct_answer=correct_answer,
        cache_namespace=cache_namespace,
        classification_model=classification_model,
    )
    cached = effective_cache.get(cache_key)
    if isinstance(cached, dict):
        try:
            return ConversationBehaviorLabelsVerdict.model_validate(cached)
        except Exception:
            logger.debug(
                "Ignoring unreadable cached conversation metrics for key=%s",
                cache_key,
            )

    try:
        verdict = classify_conversation_behaviors(
            dialogue_history=normalized_dialogue,
            problem_text=problem_text,
            correct_answer=correct_answer,
            model=classification_model,
        )
    except Exception as exc:
        logger.warning("Conversation classification failed: %s", exc)
        verdict = _fallback_conversation_behavior_verdict(
            reason=f"classification_failed: {exc}",
        )

    effective_cache[cache_key] = verdict.model_dump()
    return verdict


def compute_conversation_metrics(
    *,
    dialogue_history: Any,
    problem_text: str,
    correct_answer: str,
    cache_namespace: str,
    classification_model: str,
    cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_dialogue = normalize_dialogue_history(dialogue_history)
    learner_turns = extract_learner_turns(normalized_dialogue)
    interrogative_turns = [turn for turn in learner_turns if "?" in turn]
    questions_per_interrogative_turn = (
        sum(turn.count("?") for turn in interrogative_turns) / len(interrogative_turns)
        if interrogative_turns
        else 0.0
    )
    avg_words_per_learner_turn = _mean(
        [float(_count_words(turn)) for turn in learner_turns],
    )
    avg_learner_turn_string_length = _mean([float(len(turn)) for turn in learner_turns])

    classification = classify_conversation_with_cache(
        dialogue_history=normalized_dialogue,
        problem_text=problem_text,
        correct_answer=correct_answer,
        cache_namespace=cache_namespace,
        classification_model=classification_model,
        cache=cache,
    )

    error_flags = {
        code: bool(getattr(classification.errors, field_name, False))
        for code, field_name in ERROR_TYPE_FIELDS.items()
    }
    talk_move_flags = {
        move_name: bool(getattr(classification.talk_moves, field_name, False))
        for move_name, field_name in TALK_MOVE_FIELDS.items()
    }
    error_count = sum(1 for value in error_flags.values() if value)
    error_bucket = (
        "multiple" if error_count > 1 else "one" if error_count == 1 else "zero"
    )

    return {
        "n_learner_turns": len(learner_turns),
        "n_interrogative_turns": len(interrogative_turns),
        "questions_per_interrogative_turn": questions_per_interrogative_turn,
        "avg_words_per_learner_turn": avg_words_per_learner_turn,
        "avg_learner_turn_string_length": avg_learner_turn_string_length,
        "errors": error_flags,
        "error_count_bucket": error_bucket,
        "talk_moves": talk_move_flags,
        "has_any_talk_move": any(talk_move_flags.values()),
        "classification_reasoning": classification.reasoning,
        "classification_model": classification_model,
    }


class DatasetFittedConversationalBenchmark:
    """Compare simulated conversation metrics to the real student dataset."""

    def __init__(
        self,
        skill_space: SkillSpace,
        practice_item_pool: PracticeItemPool,
        learner_config: LearnerEvalConfig,
        skill_levels: dict[str, set[str]],
        runs: int = 1,
        output_dir: Path | None = None,
        benchmark_extra_args: dict | None = None,
        practice_conversations_file: Path | str | None = None,
    ) -> None:
        self.skill_space = skill_space
        self.practice_item_pool = practice_item_pool
        self.learner_config = learner_config
        self.skill_levels = skill_levels
        self.runs = (
            benchmark_extra_args.get("runs", runs) if benchmark_extra_args else runs
        )
        self.output_dir = output_dir or Path(
            "data/benchmark_evaluations/dataset_fitted_conversational",
        )
        self.practice_conversations_file = practice_conversations_file
        self.test_run_id = f"dataset_fitted_conversational_{uuid.uuid4().hex[:8]}"

        self.benchmark_extra_args = benchmark_extra_args or {}
        self.conversations_jsonl_path = Path(
            self.benchmark_extra_args.get(
                "conversations_jsonl_path",
                DEFAULT_CONVERSATIONS_JSONL,
            ),
        )
        self.max_skills = int(self.benchmark_extra_args.get("max_skills", 5))
        self.max_records_per_skill_mastery = int(
            self.benchmark_extra_args.get("max_records_per_skill_mastery", 5),
        )
        self.max_conversation_turns = int(
            self.benchmark_extra_args.get("max_conversation_turns", 6),
        )
        self.include_multi_skill_items = bool(
            self.benchmark_extra_args.get("include_multi_skill_items", False),
        )
        self.selected_skills_override = list(
            self.benchmark_extra_args.get("selected_skills", []),
        )
        self.max_dataset_conversations = self.benchmark_extra_args.get(
            "max_dataset_conversations",
        )
        self.require_both_mastery_groups = bool(
            self.benchmark_extra_args.get("require_both_mastery_groups", True),
        )
        self.random_seed = int(self.benchmark_extra_args.get("random_seed", 0))
        self.classification_model = str(
            self.benchmark_extra_args.get("classification_model", "gpt-4.1-mini"),
        )
        self.num_example_conversations_for_tutor_response_generation = int(
            self.benchmark_extra_args.get(
                "num_example_conversations_for_tutor_response_generation",
                5,
            ),
        )
        metrics_cache_path = self.benchmark_extra_args.get(
            "conversation_metrics_cache_path",
        )
        self.use_capped_dialogues = self.benchmark_extra_args.get(
            "use_capped_dialogues",
            True,
        )

        # other fields initialized from benchmark_extra_args
        for key, value in self.benchmark_extra_args.items():
            if not hasattr(self, key):
                setattr(self, key, value)

        self.conversation_metrics_cache_path = (
            Path(metrics_cache_path) if metrics_cache_path else None
        )
        self._conversation_metrics_cache = self._load_metrics_cache()
        self._rng = random.Random(self.random_seed)
        # Index of tutor_id -> list of raw conversation dicts (for few-shot examples)
        self._tutor_conversations_index: dict[str, list[dict[str, Any]]] = (
            self._build_tutor_conversations_index()
        )

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

        by_group = summary.get("by_skill_mastery", {})
        breakdowns = {
            group_name: {
                "average_distance": metrics.get("distance_metrics", {}).get(
                    "overall_group_average_distance",
                    0.0,
                ),
                "n_items": metrics.get("simulated", {}).get("n_records", 0),
            }
            for group_name, metrics in by_group.items()
        }
        return {
            "metric_type": "dataset_fitted_conversational",
            "overall_average_distance": summary.get(
                "overall_distance_by_metric",
                {},
            ).get(
                "overall_average_distance",
                0.0,
            ),
            "total_items": summary.get("counts", {}).get("simulated_records", 0),
            "breakdowns": {"by_skill_mastery": breakdowns},
            "breakdown_keys": ["skill", "mastery"],
            # Composite score and intermediate components
            "eval_convo_learn_score": summary.get("eval_convo_learn_score"),
            "lb_score": summary.get("aggregate_scores", {})
            .get(
                "eval_convo_learn",
                {},
            )
            .get("lb_score"),
            "conv_score": summary.get("aggregate_scores", {})
            .get(
                "eval_convo_learn",
                {},
            )
            .get("conv_score"),
            "aggregate_scores": summary.get("aggregate_scores"),
        }

    def run_all_evaluations(self) -> Path:
        output_dir = self.output_dir / self.test_run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        real_records, load_stats = self._load_real_dataset_records()
        if not real_records:
            message = (
                "No eligible conversational benchmark records found in "
                f"{self.conversations_jsonl_path}."
            )
            raise ValueError(message)

        selected_skills = self._select_skills(real_records)
        if not selected_skills:
            message = "No skills could be selected for the conversational benchmark."
            raise ValueError(message)

        sampled_records, sampling_stats = self._sample_records_for_selected_skills(
            real_records,
            selected_skills,
        )
        if not sampled_records:
            message = "No sampled records were available after skill selection."
            raise ValueError(message)

        selected_skill_set = set(selected_skills)
        selected_real_records = [
            record
            for record in real_records
            if record["target_skill_id"] in selected_skill_set
        ]
        real_metric_records = [
            self._attach_conversation_metrics(record, source="real")
            for record in selected_real_records
        ]

        details_path = output_dir / "dataset_fitted_conversational_results.jsonl"
        selected_records_path = (
            output_dir / "dataset_fitted_conversational_selected_real_records.jsonl"
        )

        with selected_records_path.open("w", encoding="utf-8") as file_handle:
            for record in sampled_records:
                file_handle.write(json.dumps(record) + "\n")

        sampled_results: list[dict[str, Any]] = []
        with details_path.open("w", encoding="utf-8") as file_handle:
            for record in sampled_records:
                for run_idx in range(self.runs):
                    result = self._evaluate_record(record=record, run_idx=run_idx)
                    sampled_results.append(result)
                    file_handle.write(json.dumps(result) + "\n")

        summary = self._build_summary(
            selected_skills=selected_skills,
            real_records=real_metric_records,
            sampled_records=sampled_records,
            sampled_results=sampled_results,
            load_stats=load_stats,
            sampling_stats=sampling_stats,
            details_path=details_path,
            selected_records_path=selected_records_path,
        )
        summary_path = output_dir / "dataset_fitted_conversational_summary.json"
        with summary_path.open("w", encoding="utf-8") as file_handle:
            json.dump(summary, file_handle, indent=2)

        self._persist_metrics_cache()

        logger.info("Dataset-fitted conversational results → %s", details_path)
        logger.info("Dataset-fitted conversational summary → %s", summary_path)
        return summary_path

    def _build_tutor_conversations_index(self) -> dict[str, list[dict[str, Any]]]:
        """Build a mapping from tutor_id to list of conversation dicts for few-shot examples."""
        index: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if not self.conversations_jsonl_path.exists():
            return dict(index)
        try:
            with self.conversations_jsonl_path.open(encoding="utf-8") as file_handle:
                for raw_line in file_handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        conv = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    tutor_id = str(conv.get("tutor_id", "")).strip()
                    if tutor_id:
                        index[tutor_id].append(conv)
        except OSError as exc:
            logger.warning("Could not build tutor conversations index: %s", exc)
        logger.info(
            "Built tutor conversations index: %d tutors, %d total conversations",
            len(index),
            sum(len(v) for v in index.values()),
        )
        return dict(index)

    def _sample_tutor_few_shot_conversations(
        self,
        tutor_id: str | None,
    ) -> list[dict[str, Any]]:
        """Sample few-shot conversation examples for a given tutor_id."""
        if not tutor_id or tutor_id not in self._tutor_conversations_index:
            return []
        pool = self._tutor_conversations_index[tutor_id]
        n = min(self.num_example_conversations_for_tutor_response_generation, len(pool))
        if n <= 0:
            return []
        sampled = self._rng.sample(pool, n)
        return [
            {
                "practice_item_text": conv.get("practice_item_text", ""),
                "dialogue_history": (
                    conv.get(
                        "capped_dialogue_history",
                        conv.get("dialogue_history", ""),
                    )
                    if self.use_capped_dialogues
                    else conv.get("dialogue_history", "")
                ),
                "mastered_skills_from_conversation": conv.get(
                    "mastered_skills_from_conversation",
                    [],
                ),
            }
            for conv in sampled
        ]

    def _load_metrics_cache(self) -> dict[str, dict[str, Any]]:
        if (
            not self.conversation_metrics_cache_path
            or not self.conversation_metrics_cache_path.exists()
        ):
            return {}
        try:
            with self.conversation_metrics_cache_path.open(
                encoding="utf-8",
            ) as file_handle:
                payload = json.load(file_handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to load conversation metrics cache %s: %s",
                self.conversation_metrics_cache_path,
                exc,
            )
            return {}

        return payload if isinstance(payload, dict) else {}

    def _persist_metrics_cache(self) -> None:
        if not self.conversation_metrics_cache_path:
            return
        self.conversation_metrics_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.conversation_metrics_cache_path.open(
            "w",
            encoding="utf-8",
        ) as file_handle:
            json.dump(
                self._conversation_metrics_cache,
                file_handle,
                indent=2,
                sort_keys=True,
            )

    def _load_real_dataset_records(self) -> tuple[list[dict[str, Any]], dict[str, int]]:
        normalized_records: list[dict[str, Any]] = []
        stats = {
            "raw_conversations": 0,
            "eligible_records": 0,
            "skipped_missing_skill": 0,
            "skipped_multi_skill": 0,
            "skipped_invalid_json": 0,
            "skipped_missing_problem": 0,
        }

        with self.conversations_jsonl_path.open(encoding="utf-8") as file_handle:
            for line_idx, raw_line in enumerate(file_handle, start=1):
                if self.max_dataset_conversations and stats["raw_conversations"] >= int(
                    self.max_dataset_conversations,
                ):
                    break

                line = raw_line.strip()
                if not line:
                    continue

                try:
                    conversation = json.loads(line)
                except json.JSONDecodeError:
                    stats["skipped_invalid_json"] += 1
                    logger.warning(
                        "Skipping invalid JSON in %s line %d",
                        self.conversations_jsonl_path,
                        line_idx,
                    )
                    continue

                stats["raw_conversations"] += 1
                item_skills = [
                    skill_id
                    for skill_id in conversation.get("item_skills", [])
                    if skill_id in self.skill_space
                ]
                if not item_skills:
                    stats["skipped_missing_skill"] += 1
                    continue
                if not self.include_multi_skill_items and len(item_skills) != 1:
                    stats["skipped_multi_skill"] += 1
                    continue
                if not conversation.get("practice_item_text"):
                    stats["skipped_missing_problem"] += 1
                    continue

                for skill_id in item_skills:
                    normalized = self._normalize_conversation_record(
                        conversation=conversation,
                        target_skill_id=skill_id,
                    )
                    if normalized is None:
                        continue
                    normalized_records.append(normalized)
                    stats["eligible_records"] += 1

        return normalized_records, stats

    def _normalize_conversation_record(
        self,
        conversation: dict[str, Any],
        target_skill_id: str,
    ) -> dict[str, Any] | None:
        session_id = str(conversation.get("session_id", ""))
        mastered_before = set(
            conversation.get("mastered_skills_before_conversation", []),
        )
        item_skills = list(conversation.get("item_skills", []))
        practice_item_text = str(conversation.get("practice_item_text", ""))
        correct_answer = str(conversation.get("correct_answer", ""))
        dialogue_history = normalize_dialogue_history(
            conversation.get("dialogue_history", []),
        )

        direct_prerequisites = set(self.skill_space[target_skill_id].prerequisites)
        record_prerequisites = set(conversation.get("item_skill_prerequisites", []))
        if (
            len(item_skills) == 1
            and item_skills[0] == target_skill_id
            and record_prerequisites
        ):
            effective_prerequisites = record_prerequisites
        else:
            effective_prerequisites = direct_prerequisites
        all_prerequisites = set(
            self.skill_space.get_all_prerequisites(target_skill_id, return_as_ids=True),
        ).union(effective_prerequisites)

        target_skill_mastered_before = target_skill_id in mastered_before
        mastery_group = (
            MASTERY_GROUP_MASTERED
            if target_skill_mastered_before
            else MASTERY_GROUP_UNMASTERED
        )

        prerequisites_mastered = all_prerequisites.issubset(mastered_before)
        scenario = _compute_scenario(
            target_skill_mastered_before,
            prerequisites_mastered,
        )
        # Preserve pre-computed conversation_metrics if already present in the source record
        precomputed_metrics = conversation.get("conversation_metrics")
        # Determine solution_found for real conversations:
        # - If the skill was unmastered before, learning occurred iff the skill appears in
        #   mastered_skills_from_conversation (i.e. the learner mastered it during the session).
        # - If the skill was already mastered, fall back to the explicit solution_found field.
        if not target_skill_mastered_before:
            mastered_from_conv = set(
                conversation.get("mastered_skills_from_conversation", []),
            )
            solution_found: bool | None = target_skill_id in mastered_from_conv
        else:
            raw_solution_found = conversation.get("skill_learned") or conversation.get(
                "solution_found",
            )
            solution_found = (
                bool(raw_solution_found) if raw_solution_found is not None else None
            )

        return {
            "session_id": session_id,
            "learner_id": conversation.get("learner_id"),
            "tutor_id": str(conversation.get("tutor_id", "") or ""),
            "target_skill_id": target_skill_id,
            "item_skills": item_skills,
            "practice_item_text": practice_item_text,
            "correct_answer": correct_answer,
            "dialogue_history": dialogue_history,
            "learner_turns": extract_learner_turns(dialogue_history),
            "mastered_skills_before_conversation": sorted(mastered_before),
            "direct_prerequisites": sorted(effective_prerequisites),
            "all_prerequisites": sorted(all_prerequisites),
            "target_skill_mastered_before": target_skill_mastered_before,
            "prerequisites_mastered": prerequisites_mastered,
            "mastery_group": mastery_group,
            "scenario": scenario,
            "solution_found": solution_found,
            "precomputed_conversation_metrics": precomputed_metrics,
        }

    def _select_skills(self, real_records: list[dict[str, Any]]) -> list[str]:
        required_labels = (
            set(MASTERY_GROUPS) if self.require_both_mastery_groups else None
        )
        return select_top_skills_by_count(
            real_records,
            self.max_skills,
            self.selected_skills_override,
            skill_key="target_skill_id",
            label_key="mastery_group",
            required_labels=required_labels,
        )

    def _sample_records_for_selected_skills(
        self,
        real_records: list[dict[str, Any]],
        selected_skills: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        selected_skill_set = set(selected_skills)
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        full_counts_by_skill: dict[str, Counter[str]] = defaultdict(Counter)

        for record in real_records:
            skill_id = record["target_skill_id"]
            if skill_id not in selected_skill_set:
                continue
            mastery_group = record["mastery_group"]
            grouped[(skill_id, mastery_group)].append(record)
            full_counts_by_skill[skill_id][mastery_group] += 1

        sampled_records: list[dict[str, Any]] = []
        sampling_stats: dict[str, Any] = {"by_skill": {}, "full_counts": {}}
        for skill_id in selected_skills:
            sampling_stats["by_skill"][skill_id] = {}
            sampling_stats["full_counts"][skill_id] = dict(
                full_counts_by_skill[skill_id],
            )
            for mastery_group in MASTERY_GROUPS:
                full_group = list(grouped.get((skill_id, mastery_group), []))
                if not full_group:
                    continue
                if len(full_group) <= self.max_records_per_skill_mastery:
                    sampled_group = full_group
                else:
                    sampled_group = self._rng.sample(
                        full_group,
                        self.max_records_per_skill_mastery,
                    )
                sampling_weight = len(full_group) / len(sampled_group)
                sampling_stats["by_skill"][skill_id][mastery_group] = {
                    "full_records": len(full_group),
                    "sampled_records": len(sampled_group),
                    "sampling_weight": sampling_weight,
                }
                sampled_records.extend(
                    [
                        {**record, "sampling_weight": sampling_weight}
                        for record in sampled_group
                    ],
                )

        sampled_records.sort(
            key=lambda record: (
                record["target_skill_id"],
                record["mastery_group"],
                record["session_id"],
            ),
        )
        return sampled_records, sampling_stats

    def _evaluate_record(self, record: dict[str, Any], run_idx: int) -> dict[str, Any]:
        learner_id = (
            f"dataset_fit_conv_{record['target_skill_id']}_{record['mastery_group']}_"
            f"{record['session_id']}_{run_idx}_{uuid.uuid4().hex[:6]}"
        )
        initialization_mastered_skills = self._initial_mastered_skills_for_record(
            record,
        )
        try:
            learner = _create_learner_for_scenario(
                self.learner_config,
                self.skill_space,
                initialization_mastered_skills,
                learner_id=learner_id,
                practice_conversations_file=self.practice_conversations_file,
                practice_item_pool=self.practice_item_pool,
                tutor=self.tutor,
            )
        except LearnerInitializationError as exc:
            return {
                "benchmark": "DatasetFittedConversationalBenchmark",
                "session_id": record["session_id"],
                "target_skill_id": record["target_skill_id"],
                "mastery_group": record["mastery_group"],
                "target_skill_mastered_before": record["target_skill_mastered_before"],
                "group_key": _group_key(
                    record["target_skill_id"],
                    record["mastery_group"],
                ),
                "run_id": run_idx,
                "initialization_failed": True,
                "error": str(exc),
                "sampling_weight": record["sampling_weight"] / max(self.runs, 1),
                "timestamp": datetime.now().isoformat(),
            }

        practice_item = self._resolve_practice_item(record)
        # Sample few-shot conversations from the same tutor for grounded response generation
        few_shot_convs = self._sample_tutor_few_shot_conversations(
            record.get("tutor_id") or None,
        )
        tutor_generation_metadata: dict[str, Any] = {}
        logger.info(
            "Using %d few-shot conversations for tutor response grounding.",
            len(few_shot_convs),
        )
        if few_shot_convs:
            tutor_generation_metadata["few_shot_conversations"] = few_shot_convs
        conversation = run_base_learner_conversation(
            learner=learner,
            practice_item=practice_item,
            tutor=self.tutor,
            max_turns=self.max_conversation_turns,
            session_id=f"dataset_fit_conv_{uuid.uuid4().hex[:10]}",
            item_skills=[record["target_skill_id"]],
            save_conversation=True,
            correct_answer=record.get("correct_answer", ""),
            tutor_generation_metadata=(
                tutor_generation_metadata if tutor_generation_metadata else None
            ),
        )
        dialogue_history = [
            {
                "role": "assistant",
                "content": f"Let's work on the following problem together: {practice_item.text}",
            },
            *conversation.to_history(),
        ]

        scenario = record.get(
            "scenario",
            _compute_scenario(
                record.get("target_skill_mastered_before", False),
                record.get("prerequisites_mastered", False),
            ),
        )
        result = {
            "benchmark": "DatasetFittedConversationalBenchmark",
            "session_id": record["session_id"],
            "learner_id": learner_id,
            "tutor_id": record.get("tutor_id", ""),
            "target_skill_id": record["target_skill_id"],
            "mastery_group": record["mastery_group"],
            "prerequisites_mastered": record.get("prerequisites_mastered", False),
            "scenario": scenario,
            "target_skill_mastered_before": record["target_skill_mastered_before"],
            "group_key": _group_key(record["target_skill_id"], record["mastery_group"]),
            "scenario_key": _scenario_key(record["target_skill_id"], scenario),
            "run_id": run_idx,
            "practice_item_text": practice_item.text[:200],
            "correct_answer": record.get("correct_answer", ""),
            "initialized_mastered_skills": initialization_mastered_skills,
            "direct_prerequisites": record.get("direct_prerequisites", []),
            "all_prerequisites": record.get("all_prerequisites", []),
            "few_shot_tutor_conversations_used": len(few_shot_convs),
            "dialogue_history": dialogue_history,
            "learner_turns": extract_learner_turns(dialogue_history),
            "solution_found": conversation.solution_found,
            "num_turns": conversation.num_turns,
            "conversation_ended_reason": conversation.conversation_ended_reason,
            "sampling_weight": record["sampling_weight"] / max(self.runs, 1),
            "initialization_failed": False,
            "timestamp": datetime.now().isoformat(),
        }
        return self._attach_conversation_metrics(result, source="simulated")

    def _initial_mastered_skills_for_record(self, record: dict[str, Any]) -> list[str]:
        mastered_ids = set(record.get("all_prerequisites", []))
        if record.get("target_skill_mastered_before"):
            mastered_ids.add(record["target_skill_id"])
        return sorted(mastered_ids)

    def _resolve_practice_item(self, record: dict[str, Any]) -> PracticeItem:
        try:
            practice_item = self.practice_item_pool.get_item_by_text(
                record["practice_item_text"],
            )
        except ValueError:
            logger.warning(
                "Could not find practice item in pool matching text '%s', using fallback for session_id=%s",
                record["practice_item_text"],
                record.get("session_id", ""),
            )
            practice_item = PracticeItem(
                text=record["practice_item_text"],
                associated_skills=[record["target_skill_id"]],
                answer=record.get("correct_answer", ""),
            )
        return PracticeItem(
            text=practice_item.text,
            associated_skills=list(practice_item.associated_skills),
            answer=practice_item.answer or record.get("correct_answer", ""),
            incorrect_answers=list(practice_item.incorrect_answers),
        )

    def _attach_conversation_metrics(
        self,
        record: dict[str, Any],
        *,
        source: str,
    ) -> dict[str, Any]:
        # For real records, reuse pre-computed metrics from the source JSONL when available
        if source == "real":
            precomputed = record.get("precomputed_conversation_metrics")
            if isinstance(precomputed, dict) and precomputed:
                logger.debug(
                    "Reusing precomputed conversation metrics for session=%s",
                    record.get("session_id", ""),
                )
                return {**record, "conversation_metrics": precomputed}
        conversation_metrics = self._compute_conversation_metrics(
            dialogue_history=record.get("dialogue_history", []),
            problem_text=str(record.get("practice_item_text", "")),
            correct_answer=str(record.get("correct_answer", "")),
            cache_namespace=source,
        )
        return {**record, "conversation_metrics": conversation_metrics}

    def _compute_conversation_metrics(
        self,
        *,
        dialogue_history: Any,
        problem_text: str,
        correct_answer: str,
        cache_namespace: str,
    ) -> dict[str, Any]:
        return compute_conversation_metrics(
            dialogue_history=dialogue_history,
            problem_text=problem_text,
            correct_answer=correct_answer,
            cache_namespace=cache_namespace,
            classification_model=self.classification_model,
            cache=self._conversation_metrics_cache,
        )

    def _aggregate_group_metrics(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        total_weight = 0.0
        questions_weighted_sum = 0.0
        words_weighted_sum = 0.0
        string_length_weighted_sum = 0.0
        error_bucket_counts = dict.fromkeys(ERROR_BUCKETS, 0.0)
        any_talk_move_counts = dict.fromkeys(BINARY_BUCKETS, 0.0)
        error_yes_weights = dict.fromkeys(ERROR_TYPE_FIELDS, 0.0)
        talk_move_yes_weights = dict.fromkeys(TALK_MOVE_FIELDS, 0.0)

        valid_records = 0
        for record in records:
            metrics = record.get("conversation_metrics")
            if not isinstance(metrics, dict):
                continue
            weight = float(record.get("sampling_weight", 1.0))
            total_weight += weight
            valid_records += 1
            questions_weighted_sum += (
                float(metrics.get("questions_per_interrogative_turn", 0.0)) * weight
            )
            words_weighted_sum += (
                float(metrics.get("avg_words_per_learner_turn", 0.0)) * weight
            )
            string_length_weighted_sum += (
                float(
                    metrics.get("avg_learner_turn_string_length", 0.0),
                )
                * weight
            )

            error_bucket = str(metrics.get("error_count_bucket", "zero"))
            error_bucket_counts[error_bucket] = (
                error_bucket_counts.get(error_bucket, 0.0) + weight
            )

            has_any_talk_move = bool(metrics.get("has_any_talk_move", False))
            any_talk_move_counts["yes" if has_any_talk_move else "no"] += weight

            for code in ERROR_TYPE_FIELDS:
                if bool(metrics.get("errors", {}).get(code, False)):
                    error_yes_weights[code] = error_yes_weights.get(code, 0.0) + weight
            for move_name in TALK_MOVE_FIELDS:
                if bool(metrics.get("talk_moves", {}).get(move_name, False)):
                    talk_move_yes_weights[move_name] = (
                        talk_move_yes_weights.get(move_name, 0.0) + weight
                    )

        def _binary_distribution(true_weight: float) -> dict[str, float]:
            return counts_to_distribution(
                {"no": max(total_weight - true_weight, 0.0), "yes": true_weight},
                BINARY_BUCKETS,
            )

        return {
            "n_records": valid_records,
            "total_weight": total_weight,
            "questions_per_interrogative_turn_mean": (
                questions_weighted_sum / total_weight if total_weight else 0.0
            ),
            "avg_words_per_learner_turn_mean": (
                words_weighted_sum / total_weight if total_weight else 0.0
            ),
            "avg_learner_turn_string_length_mean": (
                string_length_weighted_sum / total_weight if total_weight else 0.0
            ),
            "error_count_bucket_distribution": counts_to_distribution(
                error_bucket_counts,
                ERROR_BUCKETS,
            ),
            "has_any_talk_move_distribution": counts_to_distribution(
                any_talk_move_counts,
                BINARY_BUCKETS,
            ),
            "errors": {
                code: {
                    "distribution": _binary_distribution(
                        float(error_yes_weights.get(code, 0.0)),
                    ),
                    "yes_rate": (
                        float(error_yes_weights.get(code, 0.0)) / total_weight
                        if total_weight
                        else 0.0
                    ),
                }
                for code in ERROR_TYPE_FIELDS
            },
            "talk_moves": {
                move_name: {
                    "distribution": _binary_distribution(
                        float(talk_move_yes_weights.get(move_name, 0.0)),
                    ),
                    "yes_rate": (
                        float(talk_move_yes_weights.get(move_name, 0.0)) / total_weight
                        if total_weight
                        else 0.0
                    ),
                }
                for move_name in TALK_MOVE_FIELDS
            },
        }

    def _distance_metrics_for_group(
        self,
        real_metrics: dict[str, Any],
        simulated_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        error_count_distance = distribution_distance(
            real_metrics.get("error_count_bucket_distribution", {}),
            simulated_metrics.get("error_count_bucket_distribution", {}),
            ERROR_BUCKETS,
        )
        any_talk_move_distance = distribution_distance(
            real_metrics.get("has_any_talk_move_distribution", {}),
            simulated_metrics.get("has_any_talk_move_distribution", {}),
            BINARY_BUCKETS,
        )
        error_distances = {
            code: distribution_distance(
                real_metrics.get("errors", {}).get(code, {}).get("distribution", {}),
                simulated_metrics.get("errors", {})
                .get(code, {})
                .get("distribution", {}),
                BINARY_BUCKETS,
            )
            for code in ERROR_TYPE_FIELDS
        }
        talk_move_distances = {
            move_name: distribution_distance(
                real_metrics.get("talk_moves", {})
                .get(move_name, {})
                .get("distribution", {}),
                simulated_metrics.get("talk_moves", {})
                .get(move_name, {})
                .get("distribution", {}),
                BINARY_BUCKETS,
            )
            for move_name in TALK_MOVE_FIELDS
        }

        flat_distances = [
            abs(
                float(real_metrics.get("questions_per_interrogative_turn_mean", 0.0))
                - float(
                    simulated_metrics.get("questions_per_interrogative_turn_mean", 0.0),
                ),
            ),
            abs(
                float(real_metrics.get("avg_words_per_learner_turn_mean", 0.0))
                - float(simulated_metrics.get("avg_words_per_learner_turn_mean", 0.0)),
            ),
            abs(
                float(real_metrics.get("avg_learner_turn_string_length_mean", 0.0))
                - float(
                    simulated_metrics.get("avg_learner_turn_string_length_mean", 0.0),
                ),
            ),
            error_count_distance["l1_distance"],
            any_talk_move_distance["l1_distance"],
            *[distance["l1_distance"] for distance in error_distances.values()],
            *[distance["l1_distance"] for distance in talk_move_distances.values()],
        ]

        return {
            "questions_per_interrogative_turn_abs_diff": flat_distances[0],
            "avg_words_per_learner_turn_abs_diff": flat_distances[1],
            "avg_learner_turn_string_length_abs_diff": flat_distances[2],
            "error_count_bucket_distribution": error_count_distance,
            "has_any_talk_move_distribution": any_talk_move_distance,
            "errors": error_distances,
            "talk_moves": talk_move_distances,
            "overall_group_average_distance": _mean(flat_distances),
        }

    # ------------------------------------------------------------------
    # Aggregate composite scoring
    # ------------------------------------------------------------------

    def _compute_aggregate_scores(
        self,
        selected_skills: list[str],
        real_records: list[dict[str, Any]],
        successful_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute the composite EvalConvoLearn score and all intermediate steps.

        Returns a dict with:
        - ``scenario_weights``: occurrence frequency of each scenario in real data
        - ``learning_behavior``: MAE on solution rate per skill×scenario, macro-averaged
          across skills per scenario, then weighted across scenarios
        - ``conversational``: per-scenario distances.  Continuous metrics (W1) are
          normalized by the global real-data range.  Categorical metrics (JSD) pool
          conversations within a scenario across skills; each conversation is first
          converted to a normalized distribution, then distributions are averaged.
        - ``eval_convo_learn``: final composite score
        """
        from collections import Counter as _Counter

        # ---- Scenario weights from ALL real records ----
        scenario_counts: dict[str, int] = _Counter(
            record.get("scenario", "")
            for record in real_records
            if record.get("scenario")
        )
        total_real = sum(scenario_counts.values()) or 1
        scenario_weights = {
            scenario: scenario_counts.get(scenario, 0) / total_real
            for scenario in SCENARIOS
        }

        # ---- Global real-data ranges for continuous metric normalization ----
        # Collect all real observations across the entire dataset (all scenarios/skills)
        # so the normalization denominator is stable and independent of the simulator.
        import math as _math

        all_real_q = [
            float(
                r["conversation_metrics"].get("questions_per_interrogative_turn", 0.0),
            )
            for r in real_records
            if isinstance(r.get("conversation_metrics"), dict)
        ]
        all_real_w = [
            float(r["conversation_metrics"].get("avg_words_per_learner_turn", 0.0))
            for r in real_records
            if isinstance(r.get("conversation_metrics"), dict)
        ]
        real_q_range = (
            (max(all_real_q) - min(all_real_q)) if len(all_real_q) >= 2 else 1.0
        )
        real_w_range = (
            (max(all_real_w) - min(all_real_w)) if len(all_real_w) >= 2 else 1.0
        )
        if real_q_range <= 0.0:
            real_q_range = 1.0
        if real_w_range <= 0.0:
            real_w_range = 1.0
        # Pre-compute global log-scale range from real data for reference / diagnostics.
        # NOTE: the per-scenario w_w1_log uses the *joint* (real+sim) log range as
        # normalizer (val_range=None) to prevent saturation when the simulated
        # distribution falls far outside the real range.
        all_real_w_log = [_math.log1p(v) for v in all_real_w]
        real_w_log_range_global = (
            (max(all_real_w_log) - min(all_real_w_log))
            if len(all_real_w_log) >= 2
            else 1.0
        ) or 1.0

        # ---- Learning behavior (LB) score ----
        # For each skill×scenario compute |real_solution_rate - sim_solution_rate|.
        # Macro-average the absolute errors across skills within each scenario,
        # then do a scenario-frequency-weighted average across scenarios.
        # This gives an interpretable "the simulator is off by X% on solution rate".
        lb_per_scenario: dict[str, Any] = {}
        for scenario in SCENARIOS:
            real_skill_rates: dict[str, float] = {}
            sim_skill_rates: dict[str, float] = {}
            per_skill_ae: dict[str, float] = {}
            for skill_id in selected_skills:
                real_grp = [
                    r
                    for r in real_records
                    if r.get("target_skill_id") == skill_id
                    and r.get("scenario") == scenario
                    and r.get("solution_found") is not None
                ]
                sim_grp = [
                    r
                    for r in successful_results
                    if r.get("target_skill_id") == skill_id
                    and r.get("scenario") == scenario
                ]
                if real_grp:
                    real_skill_rates[skill_id] = _mean(
                        [1.0 if r.get("solution_found") else 0.0 for r in real_grp],
                    )
                if sim_grp:
                    sim_skill_rates[skill_id] = _mean(
                        [1.0 if r.get("solution_found") else 0.0 for r in sim_grp],
                    )
                if skill_id in real_skill_rates and skill_id in sim_skill_rates:
                    per_skill_ae[skill_id] = abs(
                        real_skill_rates[skill_id] - sim_skill_rates[skill_id],
                    )

            has_real_solution_data = bool(real_skill_rates)
            macro_mae: float | None = (
                _mean(list(per_skill_ae.values())) if per_skill_ae else None
            )

            lb_per_scenario[scenario] = {
                "weight": scenario_weights[scenario],
                "has_real_solution_data": has_real_solution_data,
                "real_skill_solution_rates": real_skill_rates,
                "sim_skill_solution_rates": sim_skill_rates,
                "per_skill_absolute_error": per_skill_ae,
                "macro_mae": macro_mae,
            }

        # Weighted average of per-scenario MAE across scenarios that have data
        lb_wsum = 0.0
        lb_wtotal = 0.0
        for scenario, data in lb_per_scenario.items():
            if data["macro_mae"] is not None:
                w = data["weight"]
                lb_wsum += data["macro_mae"] * w
                lb_wtotal += w
        lb_score = lb_wsum / lb_wtotal if lb_wtotal > 0 else 0.0

        # ---- Conversational score ----
        error_type_keys = sorted(ERROR_TYPE_FIELDS.keys())
        talk_move_keys = sorted(TALK_MOVE_FIELDS.keys())
        n_error_keys = len(error_type_keys)
        n_talk_keys = len(talk_move_keys)

        def _conv_to_dist(flags: list[float], n_keys: int) -> list[float]:
            """Normalize a per-category presence vector to a probability distribution.

            Each conversation is represented as a binary vector (1 = category observed).
            Divide by the total count to obtain a proper distribution; fall back to a
            uniform distribution when nothing is present (all zeros).
            """
            total = sum(flags)
            if total > 0.0:
                return [v / total for v in flags]
            return [1.0 / n_keys] * n_keys

        conv_per_scenario: dict[str, Any] = {}
        for scenario in SCENARIOS:
            # Pool ALL conversations in this scenario across all skills
            real_scen = [
                r
                for r in real_records
                if r.get("scenario") == scenario
                and isinstance(r.get("conversation_metrics"), dict)
            ]
            sim_scen = [
                r
                for r in successful_results
                if r.get("scenario") == scenario
                and isinstance(r.get("conversation_metrics"), dict)
            ]

            # ---- Continuous metrics: W1 normalized by global real-data range ----
            real_q_vals = [
                float(
                    r["conversation_metrics"].get(
                        "questions_per_interrogative_turn",
                        0.0,
                    ),
                )
                for r in real_scen
            ]
            sim_q_vals = [
                float(
                    r["conversation_metrics"].get(
                        "questions_per_interrogative_turn",
                        0.0,
                    ),
                )
                for r in sim_scen
            ]
            real_w_vals = [
                float(r["conversation_metrics"].get("avg_words_per_learner_turn", 0.0))
                for r in real_scen
            ]
            sim_w_vals = [
                float(r["conversation_metrics"].get("avg_words_per_learner_turn", 0.0))
                for r in sim_scen
            ]
            q_w1 = _wasserstein1_normalized(
                real_q_vals,
                sim_q_vals,
                val_range=real_q_range,
            )
            w_w1 = _wasserstein1_normalized(
                real_w_vals,
                sim_w_vals,
                val_range=real_w_range,
            )

            # Log-scale word counts before computing W1 to reduce sensitivity to
            # extreme length differences (simulated learners tend to write much more
            # than real students, which saturates the raw W1 at 1.0).
            # Use val_range=None (joint real+sim range) instead of the real-only range:
            # when sim values fall far outside the real range, the real-only log range
            # is too narrow and the metric saturates at 1.0, losing all signal.
            real_w_log = [_math.log1p(v) for v in real_w_vals]
            sim_w_log = [_math.log1p(v) for v in sim_w_vals]
            w_w1_log = _wasserstein1_normalized(real_w_log, sim_w_log, val_range=None)

            # ---- Categorical metrics: per-conversation distributions → average → JSD ----
            # Each conversation becomes a normalized distribution before averaging,
            # so every conversation is weighted equally regardless of how many
            # categories are present.
            def _avg_dist(
                records: list[dict[str, Any]],
                metric_key: str,
                keys: list[str],
                n_keys: int,
            ) -> list[float]:
                distributions = []
                for r in records:
                    metrics = r["conversation_metrics"]
                    flags = [
                        1.0 if metrics.get(metric_key, {}).get(k) else 0.0 for k in keys
                    ]
                    distributions.append(_conv_to_dist(flags, n_keys))
                if not distributions:
                    return [1.0 / n_keys] * n_keys
                # Average across conversations (weights each conversation equally)
                return [_mean([d[i] for d in distributions]) for i in range(n_keys)]

            real_err_dist = _avg_dist(
                real_scen,
                "errors",
                error_type_keys,
                n_error_keys,
            )
            sim_err_dist = _avg_dist(sim_scen, "errors", error_type_keys, n_error_keys)
            error_jsd = _jsd_log2(real_err_dist, sim_err_dist)

            real_talk_dist = _avg_dist(
                real_scen,
                "talk_moves",
                talk_move_keys,
                n_talk_keys,
            )
            sim_talk_dist = _avg_dist(
                sim_scen,
                "talk_moves",
                talk_move_keys,
                n_talk_keys,
            )
            talk_jsd = _jsd_log2(real_talk_dist, sim_talk_dist)

            # Weights: continuous metrics share 40%, categorical JSD metrics share 60%.
            METRIC_WEIGHTS = {
                "q_w1": 0.15,
                "w_w1": 0.25,
                "error_jsd": 0.30,
                "talk_jsd": 0.30,
            }
            macro_avg = (
                METRIC_WEIGHTS["q_w1"] * q_w1
                + METRIC_WEIGHTS["w_w1"] * w_w1_log
                + METRIC_WEIGHTS["error_jsd"] * error_jsd
                + METRIC_WEIGHTS["talk_jsd"] * talk_jsd
            )

            # ---- Per-skill breakdown (diagnostic only, not used in aggregation) ----
            per_skill_breakdown: dict[str, Any] = {}
            for skill_id in selected_skills:
                real_sk = [r for r in real_scen if r.get("target_skill_id") == skill_id]
                sim_sk = [r for r in sim_scen if r.get("target_skill_id") == skill_id]
                real_sk_q = [
                    float(
                        r["conversation_metrics"].get(
                            "questions_per_interrogative_turn",
                            0.0,
                        ),
                    )
                    for r in real_sk
                ]
                sim_sk_q = [
                    float(
                        r["conversation_metrics"].get(
                            "questions_per_interrogative_turn",
                            0.0,
                        ),
                    )
                    for r in sim_sk
                ]
                real_sk_w = [
                    float(
                        r["conversation_metrics"].get(
                            "avg_words_per_learner_turn",
                            0.0,
                        ),
                    )
                    for r in real_sk
                ]
                sim_sk_w = [
                    float(
                        r["conversation_metrics"].get(
                            "avg_words_per_learner_turn",
                            0.0,
                        ),
                    )
                    for r in sim_sk
                ]
                real_sk_w_log = [_math.log1p(v) for v in real_sk_w]
                sim_sk_w_log = [_math.log1p(v) for v in sim_sk_w]
                per_skill_breakdown[skill_id] = {
                    "n_real": len(real_sk),
                    "n_sim": len(sim_sk),
                    "real_q_mean": _mean(real_sk_q) if real_sk_q else None,
                    "sim_q_mean": _mean(sim_sk_q) if sim_sk_q else None,
                    "real_words_mean": _mean(real_sk_w) if real_sk_w else None,
                    "sim_words_mean": _mean(sim_sk_w) if sim_sk_w else None,
                    "q_w1": _wasserstein1_normalized(
                        real_sk_q,
                        sim_sk_q,
                        val_range=real_q_range,
                    ),
                    "words_w1": _wasserstein1_normalized(
                        real_sk_w,
                        sim_sk_w,
                        val_range=real_w_range,
                    ),
                    "words_w1_log": _wasserstein1_normalized(
                        real_sk_w_log,
                        sim_sk_w_log,
                        val_range=None,
                    ),
                    "words_w1_log_global_range": real_w_log_range_global,
                    "real_error_dist": dict(
                        zip(
                            error_type_keys,
                            _avg_dist(real_sk, "errors", error_type_keys, n_error_keys),
                            strict=False,
                        ),
                    ),
                    "sim_error_dist": dict(
                        zip(
                            error_type_keys,
                            _avg_dist(sim_sk, "errors", error_type_keys, n_error_keys),
                            strict=False,
                        ),
                    ),
                    "real_talk_dist": dict(
                        zip(
                            talk_move_keys,
                            _avg_dist(
                                real_sk,
                                "talk_moves",
                                talk_move_keys,
                                n_talk_keys,
                            ),
                            strict=False,
                        ),
                    ),
                    "sim_talk_dist": dict(
                        zip(
                            talk_move_keys,
                            _avg_dist(
                                sim_sk,
                                "talk_moves",
                                talk_move_keys,
                                n_talk_keys,
                            ),
                            strict=False,
                        ),
                    ),
                }

            conv_per_scenario[scenario] = {
                "weight": scenario_weights[scenario],
                "questions_per_interrogative_turn_w1": q_w1,
                "avg_words_per_learner_turn_w1": w_w1,
                "avg_words_per_learner_turn_w1_log": w_w1_log,
                "error_types_jsd": error_jsd,
                "talk_moves_jsd": talk_jsd,
                "macro_average_distance": macro_avg,
                "per_skill_breakdown": per_skill_breakdown,
                "detail": {
                    "real_q_vals": real_q_vals,
                    "sim_q_vals": sim_q_vals,
                    "real_w_vals": real_w_vals,
                    "sim_w_vals": sim_w_vals,
                    "real_error_dist": dict(
                        zip(error_type_keys, real_err_dist, strict=False),
                    ),
                    "sim_error_dist": dict(
                        zip(error_type_keys, sim_err_dist, strict=False),
                    ),
                    "real_talk_dist": dict(
                        zip(talk_move_keys, real_talk_dist, strict=False),
                    ),
                    "sim_talk_dist": dict(
                        zip(talk_move_keys, sim_talk_dist, strict=False),
                    ),
                    "real_q_range_used": real_q_range,
                    "real_w_range_used": real_w_range,
                    "real_w_log_range_global": real_w_log_range_global,
                },
            }

        # Weighted average conversational score
        conv_wsum = 0.0
        conv_wtotal = 0.0
        for scenario, data in conv_per_scenario.items():
            w = data["weight"]
            conv_wsum += data["macro_average_distance"] * w
            conv_wtotal += w
        conv_score = conv_wsum / conv_wtotal if conv_wtotal > 0 else 0.0

        # ---- Final EvalConvoLearn composite score ----
        alpha = EVAL_CONVO_LEARN_ALPHA
        eval_convo_learn = alpha * lb_score + (1.0 - alpha) * conv_score

        return {
            "scenario_weights": scenario_weights,
            "learning_behavior": {
                "per_scenario": lb_per_scenario,
                "weighted_lb_score": lb_score,
            },
            "conversational": {
                "per_scenario": conv_per_scenario,
                "weighted_conv_score": conv_score,
            },
            "eval_convo_learn": {
                "alpha": alpha,
                "lb_score": lb_score,
                "conv_score": conv_score,
                "score": eval_convo_learn,
            },
        }

    def _build_summary(
        self,
        selected_skills: list[str],
        real_records: list[dict[str, Any]],
        sampled_records: list[dict[str, Any]],
        sampled_results: list[dict[str, Any]],
        load_stats: dict[str, int],
        sampling_stats: dict[str, Any],
        details_path: Path,
        selected_records_path: Path,
    ) -> dict[str, Any]:
        successful_results = [
            result
            for result in sampled_results
            if not result.get("initialization_failed", False)
        ]
        initialization_failures = len(sampled_results) - len(successful_results)

        # ---------- by skill x scenario grouping ----------
        by_skill_scenario: dict[str, Any] = {}
        group_distances: list[dict[str, Any]] = []
        for skill_id in selected_skills:
            for scenario in SCENARIOS:
                scen_key = _scenario_key(skill_id, scenario)
                real_scen_group = [
                    record
                    for record in real_records
                    if record["target_skill_id"] == skill_id
                    and record.get("scenario") == scenario
                ]
                simulated_scen_group = [
                    result
                    for result in successful_results
                    if result["target_skill_id"] == skill_id
                    and result.get("scenario") == scenario
                ]
                if not real_scen_group and not simulated_scen_group:
                    continue  # skip absent scenarios
                real_metrics = self._aggregate_group_metrics(real_scen_group)
                simulated_metrics = self._aggregate_group_metrics(simulated_scen_group)
                distance_metrics = self._distance_metrics_for_group(
                    real_metrics,
                    simulated_metrics,
                )
                # solution_found stats for simulated group
                solution_found_values = [
                    1.0 if r.get("solution_found") else 0.0
                    for r in simulated_scen_group
                ]
                avg_solution_found = _mean(solution_found_values)
                by_skill_scenario[scen_key] = {
                    "skill_id": skill_id,
                    "scenario": scenario,
                    "real": real_metrics,
                    "simulated": simulated_metrics,
                    "distance_metrics": distance_metrics,
                    "solution_found_rate": avg_solution_found,
                    "n_simulated": len(simulated_scen_group),
                }
                if real_metrics.get("n_records", 0) > 0:
                    group_distances.append(distance_metrics)

        # ---------- backward-compat by_skill_mastery grouping ----------
        by_skill_mastery: dict[str, Any] = {}
        for skill_id in selected_skills:
            for mastery_group in MASTERY_GROUPS:
                group_name = _group_key(skill_id, mastery_group)
                real_group = [
                    record
                    for record in real_records
                    if record["target_skill_id"] == skill_id
                    and record["mastery_group"] == mastery_group
                ]
                simulated_group = [
                    result
                    for result in successful_results
                    if result["target_skill_id"] == skill_id
                    and result["mastery_group"] == mastery_group
                ]
                real_metrics_mg = self._aggregate_group_metrics(real_group)
                simulated_metrics_mg = self._aggregate_group_metrics(simulated_group)
                distance_metrics_mg = self._distance_metrics_for_group(
                    real_metrics_mg,
                    simulated_metrics_mg,
                )
                solution_found_mg = [
                    1.0 if r.get("solution_found") else 0.0 for r in simulated_group
                ]
                by_skill_mastery[group_name] = {
                    "skill_id": skill_id,
                    "mastery_group": mastery_group,
                    "real": real_metrics_mg,
                    "simulated": simulated_metrics_mg,
                    "distance_metrics": distance_metrics_mg,
                    "solution_found_rate": _mean(solution_found_mg),
                    "n_simulated": len(simulated_group),
                }
        overall_distances = {
            "questions_per_interrogative_turn_abs_diff": _mean(
                [
                    distance["questions_per_interrogative_turn_abs_diff"]
                    for distance in group_distances
                ],
            ),
            "avg_words_per_learner_turn_abs_diff": _mean(
                [
                    distance["avg_words_per_learner_turn_abs_diff"]
                    for distance in group_distances
                ],
            ),
            "avg_learner_turn_string_length_abs_diff": _mean(
                [
                    distance["avg_learner_turn_string_length_abs_diff"]
                    for distance in group_distances
                ],
            ),
            "error_count_bucket_l1_distance": _mean(
                [
                    distance["error_count_bucket_distribution"]["l1_distance"]
                    for distance in group_distances
                ],
            ),
            "has_any_talk_move_l1_distance": _mean(
                [
                    distance["has_any_talk_move_distribution"]["l1_distance"]
                    for distance in group_distances
                ],
            ),
            "errors": {
                code: _mean(
                    [
                        distance["errors"][code]["l1_distance"]
                        for distance in group_distances
                    ],
                )
                for code in ERROR_TYPE_FIELDS
            },
            "talk_moves": {
                move_name: _mean(
                    [
                        distance["talk_moves"][move_name]["l1_distance"]
                        for distance in group_distances
                    ],
                )
                for move_name in TALK_MOVE_FIELDS
            },
            "overall_average_distance": _mean(
                [
                    distance["overall_group_average_distance"]
                    for distance in group_distances
                ],
            ),
        }

        # overall solution_found rate across all successful simulated results
        all_solution_found = [
            1.0 if r.get("solution_found") else 0.0 for r in successful_results
        ]
        overall_solution_found_rate = _mean(all_solution_found)

        # ---- Aggregate composite scores ----
        aggregate_scores = self._compute_aggregate_scores(
            selected_skills=selected_skills,
            real_records=real_records,
            successful_results=successful_results,
        )

        return {
            "benchmark": "DatasetFittedConversationalBenchmark",
            "test_run_id": self.test_run_id,
            "timestamp": datetime.now().isoformat(),
            "learner_label": self.learner_config.label,
            "dataset_conversations_path": str(self.conversations_jsonl_path),
            "details_file": str(details_path),
            "selected_real_records_file": str(selected_records_path),
            "selected_skills": selected_skills,
            "counts": {
                **load_stats,
                "selected_skill_real_records": len(real_records),
                "sampled_real_records": len(sampled_records),
                "simulated_records": len(sampled_results),
                "successful_simulated_records": len(successful_results),
                "initialization_failures": initialization_failures,
            },
            "sampling": {
                "runs_per_record": self.runs,
                "max_skills": self.max_skills,
                "max_records_per_skill_mastery": self.max_records_per_skill_mastery,
                "max_conversation_turns": self.max_conversation_turns,
                "include_multi_skill_items": self.include_multi_skill_items,
                "random_seed": self.random_seed,
                "classification_model": self.classification_model,
                "num_example_conversations_for_tutor_response_generation": self.num_example_conversations_for_tutor_response_generation,
            },
            "sampling_stats": sampling_stats,
            "solution_found": {
                "overall_rate": overall_solution_found_rate,
                "by_skill_scenario": {
                    key: {
                        "skill_id": val["skill_id"],
                        "scenario": val["scenario"],
                        "solution_found_rate": val["solution_found_rate"],
                        "n_simulated": val["n_simulated"],
                    }
                    for key, val in by_skill_scenario.items()
                },
                "by_skill_mastery": {
                    key: {
                        "skill_id": val["skill_id"],
                        "mastery_group": val["mastery_group"],
                        "solution_found_rate": val["solution_found_rate"],
                        "n_simulated": val["n_simulated"],
                    }
                    for key, val in by_skill_mastery.items()
                },
            },
            "by_skill_scenario": by_skill_scenario,
            "by_skill_mastery": by_skill_mastery,
            "overall_distance_by_metric": overall_distances,
            # ---- Composite scoring (intermediate steps + final score) ----
            "aggregate_scores": aggregate_scores,
            "eval_convo_learn_score": aggregate_scores["eval_convo_learn"]["score"],
        }
