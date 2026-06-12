"""BaseLearner implementation: conversation-history knowledge representation.

Knowledge is stored as a list of plain-text summaries (``knowledge_items``)
extracted from past tutoring conversations rather than explicit skill IDs.
The learner uses these items as in-context knowledge when generating student
responses, and updates them after each conversation via an LLM-based
summarisation step.

Optional knowledge-state caching (via ``knowledge_cache_dir``) allows
the upskilling initialization loop to be skipped on subsequent calls
with the same skill set.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

from evalconvolearn import BaseLearner
from evalconvolearn.models.tutor import Tutor
from evalconvolearn.utils.llm_client import make_client

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)


class ConversationHistoryLearner(BaseLearner):
    """A base learner whose knowledge is a list of past conversation summaries.

    Fields
    ------
    knowledge_items
        Natural-language summaries of what the learner has learned so far.
    model
        OpenAI model used for all internal LLM calls.
    knowledge_cache_dir
        If set, initialized knowledge states are cached here so that
        repeated ``initialize_from_skills`` calls with the same skill
        set skip the upskilling conversations.
    """

    knowledge_items: list[str] = []
    model: str = "gpt-4.1-mini"
    knowledge_cache_dir: str | Path | None = None
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

    def export_knowledge_state(self) -> dict[str, Any]:
        return {"knowledge_items": list(self.knowledge_items)}

    def import_knowledge_state(self, state: dict[str, Any]) -> None:
        self.knowledge_items = list(state.get("knowledge_items", []))

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
    #  initialize_from_skills with caching
    # ------------------------------------------------------------------ #

    def initialize_from_skills(
        self,
        mastered_skill_ids: list[str],
        **kwargs: Any,
    ) -> None:
        """Initialize knowledge via upskilling conversations, using cache when available.

        Accepted kwargs (forwarded via ``init_knowledge_kwargs``):
        - ``model``: override the LLM model for all learner calls.
        - ``knowledge_cache_dir``: directory for caching knowledge states.
        - ``use_knowledge_cache``: whether to restore from cache (default True).
        - ``num_few_shot_examples``: number of few-shot examples for the init tutor.
        """
        model_override = kwargs.pop("model", None)
        if model_override is not None:
            self.model = model_override

        tutor_model = kwargs.pop("tutor_model", None)
        # at this point, the tutor was already initialized in the parent base_learner init through set_up_initialization_tutor
        # if a new tutor_model is provided, we need to reinitialize the strategy with the new model instead
        if tutor_model is not None:
            self.set_up_initialization_tutor(model=tutor_model)

        cache_dir = kwargs.pop("knowledge_cache_dir", None)
        if cache_dir is not None:
            self.knowledge_cache_dir = cache_dir

        use_cache = kwargs.pop("use_knowledge_cache", True)

        if use_cache and self.load_knowledge_snapshot(mastered_skill_ids):
            logger.info(
                "[initialize_from_skills] learner=%s loaded from cache — skipping upskilling",
                self.id,
            )
            return

        super().initialize_from_skills(mastered_skill_ids, **kwargs)
        self.save_knowledge_snapshot(mastered_skill_ids)

    # ------------------------------------------------------------------ #
    #  Conversation interface
    # ------------------------------------------------------------------ #

    def start_or_continue_conversation(
        self,
        conversation_history: list[dict],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Respond to the tutor using conversation history and current knowledge."""
        knowledge_str = (
            "\n".join(self.knowledge_items)
            if self.knowledge_items
            else "No prior knowledge."
        )

        def _extract_content(content: Any) -> str:
            if isinstance(content, str):
                return content
            if hasattr(content, "message"):
                return content.message
            return str(content)

        flipped_history = [
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
                    "You are a student with limited knowledge engaging with a tutor on a math practice problem.\n"
                    "Respond naturally based ONLY on the limited knowledge below, assuming very limited prior "
                    "understanding if the knowledge does not cover the math topic.\n"
                    "If the prior knowledge is insufficient, express uncertainty, ask for clarification, or "
                    "make a math mistake until you have practiced enough in the conversation.\n"
                    "If the knowledge is sufficient to understand, respond to the tutor and completely solve "
                    "the problem in your response.\n"
                    f"Your knowledge:\n<<<{knowledge_str}>>>\n"
                    "Respond in less than 3 sentences by acting as a student and NOT a tutor."
                ),
            },
            *flipped_history,
        ]

        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )
        return {
            "response": response.choices[0].message.content,
            "is_conversation_ended": False,
        }

    def end_conversation(self, conversation_history: list[dict], **kwargs: Any) -> None:
        """Extract a knowledge summary from the conversation and store it."""
        if not conversation_history:
            return

        role_names = {"assistant": "Tutor", "user": "Learner"}
        dialogue = "\n".join(
            f"{role_names.get(m['role'], m['role'].capitalize())}: {m.get('content', '')}"
            for m in conversation_history
            if isinstance(m, dict)
        )

        logger.info(
            "Updating knowledge from conversation (first 200 chars): %s…",
            dialogue[:200],
        )
        try:
            completion = self._get_client().chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You summarize what mathematical concepts a student learned in a conversation.\n"
                            "If the learner did not learn any new math concepts, respond with 'No new knowledge learned.'\n"
                            "Write one concise sentence (max 30 words) per key concept learned.\n"
                            "Format sentences as learning objectives, e.g. "
                            "'Understand how to apply the distributive property'.\n"
                            "List each concept on a separate line."
                        ),
                    },
                    {"role": "user", "content": f"Conversation:\n{dialogue}"},
                ],
                max_tokens=400,
            )
            summary = completion.choices[0].message.content.strip()
            for raw_line in summary.splitlines():
                clean_line = raw_line.strip(" -•*")
                if clean_line and "No new knowledge learned" not in clean_line:
                    logger.info("Adding knowledge item: %s", clean_line)
                    self.knowledge_items.append(clean_line)
                    break  # one summary item per conversation
        except Exception as exc:
            logger.exception("Error updating knowledge from conversation: %s", exc)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _get_client(self) -> OpenAI:
        if self._client is None:
            load_dotenv()
            self._client = make_client(self.model)
        return self._client

    def set_up_initialization_tutor(self, **kwargs) -> None:
        self._default_skill_initialization_tutor = Tutor(
            id="initialization_tutor",
            tutor_type="llm",
            tutor_characteristics={"helpfulness": True},
            practice_item_pool=None,
            response_interaction_mode="return_only",
        )
        self._default_skill_initialization_tutor.initialize_strategy(**kwargs)
