"""BaseLearner implementation: binary skill mastery.

Knowledge is represented as a set of mastered skill IDs drawn from the
loaded skill space.  On the first turn of each conversation, an LLM
identifies which skills the practice item targets.  The learner behaves
as a competent student for mastered skills and as a struggling student
for unmastered ones.  After each conversation the learner checks which
targeted skills were demonstrated and marks them as mastered.

Optional knowledge-state caching (via ``knowledge_cache_dir``) allows
expensive initialization runs to be skipped on subsequent calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from evalconvolearn import BaseLearner
from evalconvolearn.models.tutor import Tutor

logger = logging.getLogger(__name__)


class BinarySkillLearner(BaseLearner):
    """A base learner whose knowledge is a set of mastered skill IDs.

    Fields
    ------
    mastered_skill_ids
        Skill IDs the learner currently has mastered.
    model
        OpenAI model used for all internal LLM calls.
    knowledge_cache_dir
        If set, initialized knowledge states are cached here so that
        repeated ``initialize_from_skills`` calls with the same skill
        set skip the (re-)initialization work.
    """

    mastered_skill_ids: set[str] = set()
    model: str = "gpt-4.1-mini"
    knowledge_cache_dir: str | Path | None = None

    _current_item_skills: list[str] = []
    _client: OpenAI | None = None

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------ #
    #  Knowledge-state caching
    # ------------------------------------------------------------------ #

    @staticmethod
    def _skill_set_cache_key(mastered_skill_ids: list[str]) -> str:
        canonical = ",".join(sorted(mastered_skill_ids))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def _get_cache_path(self, mastered_skill_ids: list[str]) -> Path | None:
        if not self.knowledge_cache_dir:
            return None
        cache_key = self._skill_set_cache_key(mastered_skill_ids)
        return (
            Path(self.knowledge_cache_dir) / f"{type(self).__name__}_{cache_key}.json"
        )

    def save_knowledge_snapshot(self, mastered_skill_ids: list[str]) -> None:
        cache_path = self._get_cache_path(mastered_skill_ids)
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "learner_class": type(self).__name__,
            "mastered_skill_ids": sorted(mastered_skill_ids),
            "knowledge_state": self.export_knowledge_state(),
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.debug("[save_knowledge_snapshot] saved to %s", cache_path)

    def load_knowledge_snapshot(self, mastered_skill_ids: list[str]) -> bool:
        cache_path = self._get_cache_path(mastered_skill_ids)
        if cache_path is None or not cache_path.exists():
            return False
        try:
            with open(cache_path, encoding="utf-8") as f:
                payload = json.load(f)
            self.import_knowledge_state(payload["knowledge_state"])
            logger.debug("[load_knowledge_snapshot] restored from %s", cache_path)
            return True
        except Exception as exc:
            logger.error("[load_knowledge_snapshot] failed: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _get_client(self) -> OpenAI:
        if self._client is None:
            load_dotenv()
            self._client = OpenAI()
        return self._client

    def _skill_catalogue_str(self) -> str:
        return "\n".join(
            f"- {s.id}: {s.description[:120]}" for s in self.skill_space.skills
        )

    def _tag_item_with_skills(self, first_tutor_message: str) -> list[str]:
        """Use an LLM to identify which skill-space skills the item targets."""
        catalogue = self._skill_catalogue_str()
        resp = self._get_client().chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a curriculum alignment expert. Given a math practice item "
                        "and a list of skills, return ONLY the skill IDs (one per line) that "
                        "the item directly assesses. Return nothing else.\n\n"
                        f"Skill catalogue:\n{catalogue}"
                    ),
                },
                {"role": "user", "content": first_tutor_message},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        text = resp.choices[0].message.content or ""
        valid_ids = {s.id for s in self.skill_space.skills}
        tagged = [
            line.strip() for line in text.splitlines() if line.strip() in valid_ids
        ]
        logger.debug("[BinarySkillLearner] tagged skills: %s", tagged)
        return tagged

    # ------------------------------------------------------------------ #
    #  Knowledge state
    # ------------------------------------------------------------------ #

    def export_knowledge_state(self) -> dict[str, Any]:
        return {"mastered_skill_ids": sorted(self.mastered_skill_ids)}

    def import_knowledge_state(self, state: dict[str, Any]) -> None:
        self.mastered_skill_ids = set(state.get("mastered_skill_ids", []))

    # ------------------------------------------------------------------ #
    #  initialize_from_skills
    # ------------------------------------------------------------------ #

    def initialize_from_skills(
        self,
        mastered_skill_ids: list[str],
        **kwargs: Any,
    ) -> None:
        """Seed the learner by marking *mastered_skill_ids* as mastered.

        Accepted kwargs (forwarded via ``init_knowledge_kwargs``):
        - ``model``: override the LLM model for all learner calls.
        - ``knowledge_cache_dir``: directory for caching knowledge states.
        - ``use_knowledge_cache``: whether to restore from cache (default True).
        """
        model_override = kwargs.pop("model", None)
        if model_override is not None:
            self.model = model_override

        cache_dir = kwargs.pop("knowledge_cache_dir", None)
        if cache_dir is not None:
            self.knowledge_cache_dir = cache_dir

        use_cache = kwargs.pop("use_knowledge_cache", True)

        if use_cache and self.load_knowledge_snapshot(mastered_skill_ids):
            logger.info(
                "[initialize_from_skills] learner=%s loaded from cache",
                self.id,
            )
            return

        self.mastered_skill_ids = set(mastered_skill_ids)
        logger.info(
            "[BinarySkillLearner %s] initialized with %d mastered skills",
            self.id,
            len(self.mastered_skill_ids),
        )
        self.save_knowledge_snapshot(mastered_skill_ids)

    # ------------------------------------------------------------------ #
    #  Conversation interface
    # ------------------------------------------------------------------ #

    def start_or_continue_conversation(
        self,
        conversation_history: list[dict],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Respond to the tutor, adapting behavior based on skill mastery."""

        def _extract_content(content: Any) -> str:
            if isinstance(content, str):
                return content
            if hasattr(content, "message"):
                return content.message
            return str(content)

        tutor_messages = [
            m for m in conversation_history if m.get("role") == "assistant"
        ]
        if len(tutor_messages) <= 1:
            first_msg = (
                _extract_content(tutor_messages[0]["content"]) if tutor_messages else ""
            )
            self._current_item_skills = self._tag_item_with_skills(first_msg)

        unmastered = [
            s for s in self._current_item_skills if s not in self.mastered_skill_ids
        ]
        all_mastered = len(unmastered) == 0 and len(self._current_item_skills) > 0

        mastered_str = ", ".join(sorted(self.mastered_skill_ids)) or "none"
        unmastered_str = ", ".join(unmastered) or "none"

        if all_mastered:
            behavior = (
                "You already master all skills needed for this problem. "
                "Provide a complete, correct solution and show your work."
            )
        else:
            behavior = (
                "You have NOT mastered some skills needed for this problem. "
                f"Unmastered skills: {unmastered_str}. "
                "Act as a struggling student: make mistakes related to those "
                "unmastered skills, ask clarifying questions, and request help "
                "from the tutor. Do NOT solve the problem correctly until the "
                "tutor has explained the concepts."
            )

        flipped = [
            {
                "role": "user" if m["role"] == "assistant" else "assistant",
                "content": _extract_content(m["content"]),
            }
            for m in conversation_history
        ]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a math student interacting with a tutor.\n"
                    f"Your mastered skills: {mastered_str}\n"
                    f"Item's tagged skills: {', '.join(self._current_item_skills) or 'unknown'}\n\n"
                    f"{behavior}\n"
                    "Respond in at most 3 sentences as a student, NOT as a tutor."
                ),
            },
            *flipped,
        ]

        resp = self._get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )
        return {
            "response": resp.choices[0].message.content,
            "is_conversation_ended": False,
        }

    # ------------------------------------------------------------------ #
    #  End conversation — learn demonstrated skills
    # ------------------------------------------------------------------ #

    def end_conversation(
        self,
        conversation_history: list[dict],
        **kwargs: Any,
    ) -> None:
        """Check which targeted skills were demonstrated and mark them mastered."""
        if not conversation_history or not self._current_item_skills:
            self._current_item_skills = []
            return

        role_names = {"assistant": "Tutor", "user": "Learner"}
        dialogue = "\n".join(
            f"{role_names.get(m['role'], m['role'].capitalize())}: {m.get('content', '')}"
            for m in conversation_history
            if isinstance(m, dict)
        )

        try:
            resp = self._get_client().chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an assessment expert. Given a student-tutor conversation "
                            "and a list of skill IDs, determine which skills the student "
                            "successfully demonstrated understanding of by the end of the "
                            "conversation. Return ONLY the skill IDs (one per line) that the "
                            "student demonstrated. If none were demonstrated, return 'NONE'."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Skills to evaluate: {', '.join(self._current_item_skills)}\n\n"
                            f"Conversation:\n{dialogue}"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=200,
            )
            text = resp.choices[0].message.content or ""
            valid_ids = set(self._current_item_skills)
            demonstrated = [
                line.strip() for line in text.splitlines() if line.strip() in valid_ids
            ]
            if demonstrated:
                self.mastered_skill_ids.update(demonstrated)
                logger.info(
                    "[BinarySkillLearner %s] learned skills: %s (total mastered: %d)",
                    self.id,
                    demonstrated,
                    len(self.mastered_skill_ids),
                )
            else:
                logger.info(
                    "[BinarySkillLearner %s] no new skills demonstrated",
                    self.id,
                )
        except Exception as exc:
            logger.error("Error evaluating demonstrated skills: %s", exc)

        self._current_item_skills = []

    # ------------------------------------------------------------------ #
    #  Initialization tutor (set up by the benchmark)
    # ------------------------------------------------------------------ #

    def set_up_initialization_tutor(self) -> None:
        self._default_skill_initialization_tutor = Tutor(
            id="initialization_tutor",
            tutor_type="llm",
            tutor_characteristics={"helpfulness": True},
            practice_item_pool=None,
            response_interaction_mode="return_only",
        )
        self._default_skill_initialization_tutor.initialize_strategy()
