"""FlexLearner implementation: conversation-history knowledge representation.

Instead of binary skill mastery, the learner maintains a list of
``knowledge_items`` — short natural-language summaries extracted from past
tutoring conversations.  The hidden ``mastered_skills`` list is still
maintained as a prerequisite guardrail but is never shown in prompts.
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from evalconvolearn import EvalConvoLearn, FlexLearner
from evalconvolearn.models.practice_item import PracticeItem
from evalconvolearn.models.skill import Skill
from evalconvolearn.models.tutor import Tutor


class ConversationHistoryLearner(FlexLearner):
    """A FlexLearner whose knowledge is a list of past conversation summaries.

    ``knowledge_items`` are short natural-language sentences describing what
    the learner has learned.  They serve as in-context knowledge when
    generating student responses and are updated after each conversation via
    an LLM-based summarisation step.
    """

    knowledge_items: list[str] = []

    model_config = {"arbitrary_types_allowed": True}

    # ── FlexLearner abstract implementations ─────────────────────────── #

    def get_knowledge_description(self) -> str:
        if not self.knowledge_items:
            return "You have no prior knowledge yet."
        return "What you have learned so far:\n" + "\n".join(f"- {item}" for item in self.knowledge_items)

    def get_knowledge_for_problem(
        self,
        practice_item: str | PracticeItem,
        item_skills: list[Skill],
        knowledge_attrs: dict | None = None,
    ) -> str:
        return self.get_knowledge_description()

    def get_required_knowledge_to_answer_practice_item(
        self,
        practice_item: str | PracticeItem,
        practice_item_skills: list[Skill],
        knowledge_attrs: dict | None = None,
    ) -> str:
        knowledge = self.get_knowledge_description()
        associated_skills = (
            "\n".join(f"- {skill.id}: {skill.description}" for skill in practice_item_skills)
            if practice_item_skills
            else "None provided"
        )
        return (
            f"Your current knowledge (from past conversations):\n{knowledge}\n\n"
            f"You must have mastered the following skills (by assigned ID) to answer this question:\n"
            f"{associated_skills}"
        )

    def update_knowledge_from_conversation(self, dialogue_history: str) -> None:
        """Extract key takeaways from the conversation and add them to ``knowledge_items``."""
        prompt = (
            "You are summarising what a student learned from a tutoring conversation.\n\n"
            f"Conversation:\n{dialogue_history}\n\n"
            "For each concept learned, write ONE concise sentence (max 30 words) describing "
            "the key concept or procedure the student should remember.\n"
            'Return a JSON list of strings, e.g. ["...", "..."].'
        )
        try:
            load_dotenv()
            client = OpenAI()
            completion = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You produce JSON arrays of short knowledge summaries.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            raw = completion.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            new_items = json.loads(raw)
            if isinstance(new_items, list):
                self.knowledge_items.extend(new_items)
        except Exception as exc:
            print(f"Error updating knowledge from conversation: {exc}")

    def initialize_learner_knowledge(self, *args, **kwargs) -> None:
        """Initialize ``knowledge_items`` from mastered skills or explicit items.

        Accepted kwargs:
        - ``initial_knowledge_items``: list of knowledge strings to seed directly.
          If not provided, generates one item per mastered root skill.
        """
        initial_items = kwargs.get("initial_knowledge_items")
        if initial_items is not None:
            self.knowledge_items.extend(initial_items)
        elif self.mastered_skills and not self.knowledge_items:
            for sk_id in self.mastered_skills:
                skill = self.skill_space[sk_id]
                self.knowledge_items.append(f"Foundational concept: {skill.description}")

    # ── Core learner methods ─────────────────────────────────────────── #

    def save_practice_conversation(self, conversation_record: dict) -> None:
        required_keys = {"session_id", "practice_item_text", "item_skills", "dialogue_history"}
        if not required_keys.issubset(conversation_record.keys()):
            raise ValueError(f"conversation_record must contain: {required_keys}")
        conversation_record["learner_id"] = self.id
        conversation_record["knowledge_items_snapshot"] = self.knowledge_items.copy()
        try:
            with open(self.practice_conversations_file, "a", encoding="utf-8") as f:
                f.write(f"{json.dumps(conversation_record)}\n")
        except Exception as exc:
            print(f"Error saving conversation: {exc}")
