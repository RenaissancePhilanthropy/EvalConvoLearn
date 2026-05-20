"""Shared utilities for benchmarks fitted to tutoring-conversation datasets."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

DEFAULT_CONVERSATIONS_JSONL = Path(
    "data/student_pools/large_simulated_dataset_20260421-163654/all_conversations.jsonl",
)

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


__all__ = [
    "DEFAULT_CONVERSATIONS_JSONL",
    "counts_to_distribution",
    "distribution_distance",
    "extract_dialogue_turns",
    "extract_learner_turns",
    "normalize_dialogue_history",
    "select_top_skills_by_count",
]
