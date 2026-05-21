"""Abstract base class and shared tutor implementations."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.tutor import TutorResponse

logger = logging.getLogger(__name__)


class BaseTutor(ABC):
    """Abstract base class for tutors.

    Any custom tutor must inherit from this class and implement
    the `generate_response` method.
    """

    @abstractmethod
    def generate_response(
        self,
        dialogue_history: list[dict],
        **kwargs,
    ) -> TutorResponse:
        """Generate a tutor response given the conversation history.

        Args:
        ----
            dialogue_history: List of message dicts with keys "role" and "content".
                Roles are "user" (learner) and "assistant" (tutor).
            should_check_conversation_end: If True, the tutor should determine whether
                the conversation should end and add to the TutorResponse metadata as 'should_conversation_end'.
            **kwargs: Additional arguments that may be needed for response generation,
                such as student_pool_id, learner_id, session_id, etc.

        """
        ...


def load_effective_conversations(jsonl_path: Path | str) -> list[dict]:
    """Load conversations from a JSONL file where the learner demonstrated learning.

    A record is kept when ALL of the following hold:

    - ``mastered_skills_from_conversation`` is non-empty.
    - Every skill in ``mastered_skills_from_conversation`` was NOT already in
      ``mastered_skills_before_conversation`` (truly new learning).
    - Every skill in ``mastered_skills_from_conversation`` is listed in
      ``item_skills`` (aligned with the practice item).
    - All ``item_skill_prerequisites`` are present in
      ``mastered_skills_before_conversation`` (the learner was ready).
    """
    records: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                conv = json.loads(line)
            except json.JSONDecodeError:
                continue

            mastered_before: set[str] = set(
                conv.get("mastered_skills_before_conversation", []),
            )
            mastered_from: list[str] = conv.get("mastered_skills_from_conversation", [])
            item_skills: set[str] = set(conv.get("item_skills", []))
            prerequisites: list[str] = conv.get("item_skill_prerequisites", [])

            if not mastered_from:
                continue
            if not all(s not in mastered_before for s in mastered_from):
                continue
            if not all(s in item_skills for s in mastered_from):
                continue
            if not all(p in mastered_before for p in prerequisites):
                continue

            records.append(conv)

    logger.info(
        "Loaded %d effective tutoring conversations from %s",
        len(records),
        jsonl_path,
    )
    return records


def format_conversation_as_few_shot(conv: dict) -> str:
    """Render a single effective conversation as a readable few-shot block."""
    item_text = conv.get("practice_item_text", "")
    dialogue = conv.get("dialogue_history", "")
    # dialogue may be a list of turn dicts or a pre-formatted string
    if isinstance(dialogue, list):
        dialogue = "\n".join(
            f"{m.get('role', 'unknown').capitalize()}: {m.get('content', '')}"
            for m in dialogue
            if isinstance(m, dict)
        )
    return f"### Practice item: {item_text}\n### Dialogue:\n{dialogue}"
