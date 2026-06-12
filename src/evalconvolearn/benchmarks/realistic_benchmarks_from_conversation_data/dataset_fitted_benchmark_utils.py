"""Shared utilities for benchmarks fitted to tutoring-conversation datasets."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

# No default path — callers must supply `conversations_jsonl_path` explicitly via
# `benchmark_extra_args={"conversations_jsonl_path": "path/to/your/conversations.jsonl"}`.
DEFAULT_CONVERSATIONS_JSONL: Path | None = None

_DIALOGUE_TURN_PATTERN = re.compile(
    r"<<<\s*(?P<turn>\d+)\.\s*(?P<speaker>Learner|Tutor)\s*:\s*(?P<content>.*?)\s*>>>",
    re.DOTALL,
)


def normalize_dialogue_history(dialogue_history: Any) -> list[dict[str, str]]:
    """Normalize stored dialogue history to a standard message list."""
    if isinstance(dialogue_history, list):
        normalized: list[dict[str, str]] = []
        for message in dialogue_history:
            if not isinstance(message, dict):
                continue
            raw_role = str(message.get("role", "user")).strip().lower()
            role = {
                "learner": "user",
                "user": "user",
                "assistant": "assistant",
                "tutor": "assistant",
            }.get(raw_role, raw_role or "user")
            content = message.get("content", "")
            if hasattr(content, "message"):
                content = content.message
            normalized.append({"role": role, "content": str(content)})
        return normalized

    if not isinstance(dialogue_history, str):
        return []

    matches = list(_DIALOGUE_TURN_PATTERN.finditer(dialogue_history))
    if matches:
        return [
            {
                "role": "user" if match.group("speaker") == "Learner" else "assistant",
                "content": match.group("content").strip(),
            }
            for match in matches
        ]

    normalized = []
    for line in dialogue_history.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for prefix, role in (
            ("Learner:", "user"),
            ("Tutor:", "assistant"),
            ("User:", "user"),
            ("Assistant:", "assistant"),
        ):
            if stripped.startswith(prefix):
                normalized.append(
                    {"role": role, "content": stripped.split(prefix, 1)[1].strip()},
                )
                break
    return normalized


def extract_dialogue_turns(
    dialogue_history: Any,
    roles: str | Iterable[str],
) -> list[str]:
    """Extract normalized message contents for the requested roles."""
    role_set = {roles} if isinstance(roles, str) else {str(role) for role in roles}
    return [
        message["content"]
        for message in normalize_dialogue_history(dialogue_history)
        if message.get("role") in role_set
    ]


def extract_learner_turns(dialogue_history: Any) -> list[str]:
    """Extract learner turns from flexible dialogue-history formats."""
    return extract_dialogue_turns(dialogue_history, {"user"})


def counts_to_distribution(
    counts: dict[str, float],
    labels: Sequence[str],
) -> dict[str, float]:
    """Normalize counts over a fixed label set into a probability distribution."""
    total = float(sum(counts.get(label, 0.0) for label in labels))
    if total <= 0:
        return {label: 0.0 for label in labels}
    return {label: counts.get(label, 0.0) / total for label in labels}


def distribution_distance(
    real_distribution: dict[str, float],
    simulated_distribution: dict[str, float],
    labels: Sequence[str],
) -> dict[str, Any]:
    """Return the L1 distance and deltas between two discrete distributions."""
    deltas = {
        label: simulated_distribution.get(label, 0.0)
        - real_distribution.get(label, 0.0)
        for label in labels
    }
    l1_distance = sum(abs(value) for value in deltas.values())
    return {
        "l1_distance": l1_distance,
        "total_variation_distance": l1_distance / 2.0,
        "per_category_delta": deltas,
    }


def select_top_skills_by_count(
    records: list[dict[str, Any]],
    max_skills: int,
    selected_skills_override: list[str] | None = None,
    *,
    skill_key: str = "target_skill_id",
    label_key: str | None = None,
    required_labels: set[str] | None = None,
) -> list[str]:
    """Select the most frequent skills, optionally preferring label coverage."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        skill_id = str(record.get(skill_key, ""))
        if skill_id:
            grouped[skill_id].append(record)

    if selected_skills_override:
        return [
            skill_id for skill_id in selected_skills_override if skill_id in grouped
        ][:max_skills]

    ranked_skills = sorted(
        grouped,
        key=lambda skill_id: (-len(grouped[skill_id]), skill_id),
    )
    if not label_key or not required_labels:
        return ranked_skills[:max_skills]

    preferred = [
        skill_id
        for skill_id in ranked_skills
        if required_labels.issubset(
            {str(record.get(label_key, "")) for record in grouped[skill_id]},
        )
    ]
    if len(preferred) >= max_skills:
        return preferred[:max_skills]

    selected = preferred.copy()
    for skill_id in ranked_skills:
        if skill_id not in selected:
            selected.append(skill_id)
        if len(selected) >= max_skills:
            break
    return selected[:max_skills]


# ---------------------------------------------------------------------------
# Benchmark-structure constants (shared with compute_aggregate_scores)
# ---------------------------------------------------------------------------

SCENARIO_MASTERED_PREREQS_MET = "skill_mastered__prereqs_met"
SCENARIO_MASTERED_PREREQS_NOT_MET = "skill_mastered__prereqs_not_met"
SCENARIO_UNMASTERED_PREREQS_MET = "skill_unmastered__prereqs_met"
SCENARIO_UNMASTERED_PREREQS_NOT_MET = "skill_unmastered__prereqs_not_met"
SCENARIOS: list[str] = [
    SCENARIO_MASTERED_PREREQS_MET,
    SCENARIO_MASTERED_PREREQS_NOT_MET,
    SCENARIO_UNMASTERED_PREREQS_MET,
    SCENARIO_UNMASTERED_PREREQS_NOT_MET,
]

ERROR_TYPE_FIELDS: dict[str, str] = {
    "NC": "numerical_calculation",
    "CU": "conceptual_understanding",
    "PC": "problem_comprehension",
    "SD": "strategic_decision",
    "SO": "step_omission",
}

TALK_MOVE_FIELDS: dict[str, str] = {
    "asking_for_more_information": "asking_for_more_information",
    "making_a_claim": "making_a_claim",
    "providing_evidence_or_reasoning": "providing_evidence_or_reasoning",
}

EVAL_CONVO_LEARN_ALPHA: float = (
    0.5  # weight of learning-behavior vs conversational metrics in the final composite score
)

# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def _se(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    return _std(values) / math.sqrt(n)


def _jsd_log2(p_vec: list[float], q_vec: list[float]) -> float:
    """Jensen-Shannon divergence (log base 2, bounded [0, 1]) between two unnormalized vectors."""
    if not p_vec or not q_vec or len(p_vec) != len(q_vec):
        return 0.0
    p_sum, q_sum = sum(p_vec), sum(q_vec)
    if p_sum <= 0 or q_sum <= 0:
        return 1.0 if p_sum != q_sum else 0.0
    p = [x / p_sum for x in p_vec]
    q = [x / q_sum for x in q_vec]
    m = [(p[i] + q[i]) / 2.0 for i in range(len(p))]

    def _kl(a: list[float], b: list[float]) -> float:
        return sum(
            ai * math.log2(ai / bi)
            for ai, bi in zip(a, b, strict=False)
            if ai > 0.0 and bi > 0.0
        )

    return max(0.0, min(1.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m)))


def _wasserstein1_normalized(
    real_vals: list[float],
    sim_vals: list[float],
    val_range: float | None = None,
) -> float:
    """Wasserstein-1 (Earth Mover's Distance) between two empirical distributions, normalized to [0, 1]."""
    if not real_vals or not sim_vals:
        return 0.0
    if val_range is None:
        all_vals = real_vals + sim_vals
        computed_range = max(all_vals) - min(all_vals)
        val_range = computed_range if computed_range > 0.0 else 0.0
    if val_range <= 0.0:
        return 0.0
    n, m = len(real_vals), len(sim_vals)
    real_sorted, sim_sorted = sorted(real_vals), sorted(sim_vals)
    if n == m:
        w1 = sum(abs(real_sorted[i] - sim_sorted[i]) for i in range(n)) / n
    else:
        all_points = sorted(set(real_vals + sim_vals))
        w1 = 0.0
        for i in range(len(all_points) - 1):
            x0, x1 = all_points[i], all_points[i + 1]
            cdf_real = sum(1.0 for v in real_sorted if v <= x0) / n
            cdf_sim = sum(1.0 for v in sim_sorted if v <= x0) / m
            w1 += abs(cdf_real - cdf_sim) * (x1 - x0)
    return max(0.0, min(1.0, w1 / val_range))


# ---------------------------------------------------------------------------
# Aggregate score computation (module-level so it can be called per-run)
# ---------------------------------------------------------------------------

_SE_METRIC_KEYS: list[str] = [
    "q_w1",
    "w_w1_log",
    "error_jsd",
    "talk_jsd",
    "conv_score",
    "lb_score",
    "eval_convo_learn_score",
]


def compute_aggregate_scores(
    selected_skills: list[str],
    real_records: list[dict[str, Any]],
    successful_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute the composite EvalConvoLearn score and all intermediate steps.

    Returns a dict with keys: ``scenario_weights``, ``learning_behavior``,
    ``conversational``, ``eval_convo_learn``.
    """
    # ---- Scenario weights ----
    scenario_counts: dict[str, int] = Counter(
        r.get("scenario", "") for r in real_records if r.get("scenario")
    )
    total_real = sum(scenario_counts.values()) or 1
    scenario_weights = {s: scenario_counts.get(s, 0) / total_real for s in SCENARIOS}

    # ---- Global real-data ranges for normalization ----
    all_real_q = [
        float(r["conversation_metrics"].get("questions_per_interrogative_turn", 0.0))
        for r in real_records
        if isinstance(r.get("conversation_metrics"), dict)
    ]
    all_real_w = [
        float(r["conversation_metrics"].get("avg_words_per_learner_turn", 0.0))
        for r in real_records
        if isinstance(r.get("conversation_metrics"), dict)
    ]
    real_q_range = (max(all_real_q) - min(all_real_q)) if len(all_real_q) >= 2 else 1.0
    real_w_range = (max(all_real_w) - min(all_real_w)) if len(all_real_w) >= 2 else 1.0
    if real_q_range <= 0.0:
        real_q_range = 1.0
    if real_w_range <= 0.0:
        real_w_range = 1.0
    all_real_w_log = [math.log1p(v) for v in all_real_w]
    real_w_log_range_global = (
        (max(all_real_w_log) - min(all_real_w_log)) if len(all_real_w_log) >= 2 else 1.0
    ) or 1.0

    # ---- Learning behavior (LB) score ----
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
        macro_mae: float | None = (
            _mean(list(per_skill_ae.values())) if per_skill_ae else None
        )
        lb_per_scenario[scenario] = {
            "weight": scenario_weights[scenario],
            "has_real_solution_data": bool(real_skill_rates),
            "real_skill_solution_rates": real_skill_rates,
            "sim_skill_solution_rates": sim_skill_rates,
            "per_skill_absolute_error": per_skill_ae,
            "macro_mae": macro_mae,
        }

    lb_wsum = lb_wtotal = 0.0
    for data in lb_per_scenario.values():
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

    def _avg_dist(
        records: list[dict[str, Any]],
        metric_key: str,
        keys: list[str],
        n_keys: int,
        conditional_on_presence: bool = True,
    ) -> list[float]:
        distribs = []
        for r in records:
            flags = [
                1.0 if r["conversation_metrics"].get(metric_key, {}).get(k) else 0.0
                for k in keys
            ]
            if conditional_on_presence and sum(flags) == 0.0:
                continue
            if sum(flags) > 0:
                distribs.append([v / sum(flags) for v in flags])
            else:
                distribs.append([1.0 / n_keys] * n_keys)
        if not distribs:
            return [1.0 / n_keys] * n_keys
        return [_mean([d[i] for d in distribs]) for i in range(n_keys)]

    conv_per_scenario: dict[str, Any] = {}
    for scenario in SCENARIOS:
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

        real_q_vals = [
            float(
                r["conversation_metrics"].get("questions_per_interrogative_turn", 0.0),
            )
            for r in real_scen
        ]
        sim_q_vals = [
            float(
                r["conversation_metrics"].get("questions_per_interrogative_turn", 0.0),
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
        q_w1 = _wasserstein1_normalized(real_q_vals, sim_q_vals, val_range=None)
        w_w1 = _wasserstein1_normalized(real_w_vals, sim_w_vals, val_range=None)
        real_w_log = [math.log1p(v) for v in real_w_vals]
        sim_w_log = [math.log1p(v) for v in sim_w_vals]
        w_w1_log = _wasserstein1_normalized(real_w_log, sim_w_log, val_range=None)

        real_err_dist = _avg_dist(real_scen, "errors", error_type_keys, n_error_keys)
        sim_err_dist = _avg_dist(sim_scen, "errors", error_type_keys, n_error_keys)
        error_jsd = math.sqrt(_jsd_log2(real_err_dist, sim_err_dist))

        real_talk_dist = _avg_dist(real_scen, "talk_moves", talk_move_keys, n_talk_keys)
        sim_talk_dist = _avg_dist(sim_scen, "talk_moves", talk_move_keys, n_talk_keys)
        talk_jsd = math.sqrt(_jsd_log2(real_talk_dist, sim_talk_dist))

        # Weights: continuous 40 %, categorical JSD 60 %
        METRIC_WEIGHTS = {
            "q_w1": 0.25,
            "w_w1": 0.25,
            "error_jsd": 0.25,
            "talk_jsd": 0.25,
        }
        macro_avg = (
            METRIC_WEIGHTS["q_w1"] * q_w1
            + METRIC_WEIGHTS["w_w1"] * w_w1_log
            + METRIC_WEIGHTS["error_jsd"] * error_jsd
            + METRIC_WEIGHTS["talk_jsd"] * talk_jsd
        )

        # Per-skill breakdown (diagnostic only, not used in aggregation)
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
                float(r["conversation_metrics"].get("avg_words_per_learner_turn", 0.0))
                for r in real_sk
            ]
            sim_sk_w = [
                float(r["conversation_metrics"].get("avg_words_per_learner_turn", 0.0))
                for r in sim_sk
            ]
            real_sk_w_log = [math.log1p(v) for v in real_sk_w]
            sim_sk_w_log = [math.log1p(v) for v in sim_sk_w]
            per_skill_breakdown[skill_id] = {
                "n_real": len(real_sk),
                "n_sim": len(sim_sk),
                "real_q_mean": _mean(real_sk_q) if real_sk_q else None,
                "sim_q_mean": _mean(sim_sk_q) if sim_sk_q else None,
                "real_words_mean": _mean(real_sk_w) if real_sk_w else None,
                "sim_words_mean": _mean(sim_sk_w) if sim_sk_w else None,
                "q_w1": _wasserstein1_normalized(real_sk_q, sim_sk_q, val_range=None),
                "words_w1": _wasserstein1_normalized(
                    real_sk_w,
                    sim_sk_w,
                    val_range=None,
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
                        _avg_dist(real_sk, "talk_moves", talk_move_keys, n_talk_keys),
                        strict=False,
                    ),
                ),
                "sim_talk_dist": dict(
                    zip(
                        talk_move_keys,
                        _avg_dist(sim_sk, "talk_moves", talk_move_keys, n_talk_keys),
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
                "sim_talk_dist": dict(zip(talk_move_keys, sim_talk_dist, strict=False)),
                "real_q_range_used": real_q_range,
                "real_w_range_used": real_w_range,
                "real_w_log_range_global": real_w_log_range_global,
            },
        }

    conv_wsum = conv_wtotal = 0.0
    for data in conv_per_scenario.values():
        w = data["weight"]
        conv_wsum += data["macro_average_distance"] * w
        conv_wtotal += w
    conv_score = conv_wsum / conv_wtotal if conv_wtotal > 0 else 0.0

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


def compute_standard_errors(
    selected_skills: list[str],
    real_records: list[dict[str, Any]],
    successful_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute standard errors of final metrics across benchmark runs.

    Splits ``successful_results`` by ``run_id``, re-computes each aggregate
    metric per run, then reports mean ± SE across runs.  Requires >= 2 runs.
    """
    run_ids = sorted({int(r.get("run_id", 0)) for r in successful_results})
    per_run_scores: list[dict[str, Any]] = []

    for run_id in run_ids:
        run_results = [r for r in successful_results if r.get("run_id") == run_id]
        if not run_results:
            continue
        run_agg = compute_aggregate_scores(selected_skills, real_records, run_results)
        conv_per_scen = run_agg["conversational"]["per_scenario"]
        total_w = sum(d["weight"] for d in conv_per_scen.values()) or 1.0
        per_run_scores.append(
            {
                "run_id": run_id,
                "q_w1": sum(
                    d["questions_per_interrogative_turn_w1"] * d["weight"]
                    for d in conv_per_scen.values()
                )
                / total_w,
                "w_w1_log": sum(
                    d["avg_words_per_learner_turn_w1_log"] * d["weight"]
                    for d in conv_per_scen.values()
                )
                / total_w,
                "error_jsd": sum(
                    d["error_types_jsd"] * d["weight"] for d in conv_per_scen.values()
                )
                / total_w,
                "talk_jsd": sum(
                    d["talk_moves_jsd"] * d["weight"] for d in conv_per_scen.values()
                )
                / total_w,
                "conv_score": run_agg["conversational"]["weighted_conv_score"],
                "lb_score": run_agg["learning_behavior"]["weighted_lb_score"],
                "eval_convo_learn_score": run_agg["eval_convo_learn"]["score"],
            },
        )

    if len(per_run_scores) < 2:
        return {
            "note": f"SE requires >= 2 runs; {len(per_run_scores)} run(s) available.",
            "n_runs": len(per_run_scores),
            "per_run_scores": per_run_scores,
        }

    result: dict[str, Any] = {
        "n_runs": len(per_run_scores),
        "per_run_scores": per_run_scores,
    }
    for key in _SE_METRIC_KEYS:
        vals = [float(s[key]) for s in per_run_scores]
        result[key] = {
            "mean": _mean(vals),
            "std": _std(vals),
            "se": _se(vals),
            "values": vals,
        }
    return result


__all__ = [
    "DEFAULT_CONVERSATIONS_JSONL",
    "EVAL_CONVO_LEARN_ALPHA",
    "ERROR_TYPE_FIELDS",
    "SCENARIO_MASTERED_PREREQS_MET",
    "SCENARIO_MASTERED_PREREQS_NOT_MET",
    "SCENARIO_UNMASTERED_PREREQS_MET",
    "SCENARIO_UNMASTERED_PREREQS_NOT_MET",
    "SCENARIOS",
    "TALK_MOVE_FIELDS",
    "_SE_METRIC_KEYS",
    "_jsd_log2",
    "_mean",
    "_se",
    "_std",
    "_wasserstein1_normalized",
    "compute_aggregate_scores",
    "compute_standard_errors",
    "counts_to_distribution",
    "distribution_distance",
    "extract_dialogue_turns",
    "extract_learner_turns",
    "normalize_dialogue_history",
    "select_top_skills_by_count",
]
