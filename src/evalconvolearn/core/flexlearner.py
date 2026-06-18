"""FlexLearner: transparent simulated learner with skill-guardrail machinery."""

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, model_validator

from ..models.practice_item import PracticeItem
from ..models.skill import Skill
from ..utils.data_loaders import render_conversation_messages
from .base_learner import BaseLearner
from .base_tutor import BaseTutor

logger = logging.getLogger(__name__)


class LearningPromptResponse(BaseModel):
    reasoning: str
    learned_skills: list[int]


class AnswerResponse(BaseModel):
    reasoning: str
    answer: str


class FlexLearner(BaseLearner, ABC):
    """Abstract class for Flex learners.

    Extends BaseLearner with
    the full simulation machinery: skill-list guardrails, conversation
    prompting hooks, and an LLM-driven learning loop.

    Any custom Flex learner must inherit from this class and implement
    the abstract methods. The skill-based mastery state is always
    maintained as hidden guardrails, but the learner's VISIBLE
    knowledge representation is defined by each subclass.

    Subclasses MUST call ``super().__init__(**data)`` and respect
    the skill-space prerequisite constraints when updating knowledge.

    Abstract methods to implement for simulation backends:
    - get_knowledge_description
    - get_knowledge_for_problem
    - get_required_knowledge_to_answer_practice_item
    - update_knowledge_from_conversation
    - initialize_learner_knowledge

    Evaluation-only methods inherited from BaseLearner:
    - has_skill: implemented here via mastered_skills lookup
    - start_or_continue_conversation: stub, override for black-box eval
    - end_conversation: stub, override for black-box eval
    """

    mastered_skills: list[str] = []
    practice_history: list[dict] = []
    practice_conversations_file: str | Path | None = ""
    persona: dict = {}
    active_misconceptions: dict[str, str] = {}

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _expand_prerequisite_skills(self) -> "FlexLearner":
        """Ensure all ancestors of every mastered skill are also marked mastered.

        A skill cannot be mastered without its prerequisites, so any subclass
        initialized with a partial mastered_skills list (e.g. ["MA.6.NSO.4.2"])
        should automatically include all prerequisite ancestors.
        """
        if not self.mastered_skills:
            return self
        skills_to_check = self.mastered_skills.copy()
        while skills_to_check:
            sid = skills_to_check.pop()
            if sid not in self.skill_space:
                continue
            skill = self.skill_space[sid]
            for preq_id in skill.prerequisites:
                if preq_id not in self.mastered_skills:
                    self.mastered_skills.append(preq_id)
                    skills_to_check.append(preq_id)
        return self

    # ------------------------------------------------------------------ #
    #  BaseLearner evaluation-only method implementations
    # ------------------------------------------------------------------ #

    def has_skill(self, skill: str | Skill, **kwargs: Any) -> bool:
        """Return ``True`` if the learner has mastered *skill*.

        For this transparent simulated learner, mastery is a direct
        look-up in the ``mastered_skills`` list.

        Parameters
        ----------
        skill : str | Skill
            A skill ID string or a :class:`~evalconvolearn.models.skill.Skill`
            object.

        Returns
        -------
        bool

        """
        skill_id = skill.id if isinstance(skill, Skill) else skill
        return skill_id in self.mastered_skills

    def start_or_continue_conversation(
        self,
        conversation_history: list[dict],
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "FlexLearner does not implement start_or_continue_conversation. "
            "Override this method to generate responses based on the conversation history.",
        )

    def end_conversation(self, conversation_history: list[dict], **kwargs: Any) -> None:
        raise NotImplementedError(
            "FlexLearner does not implement end_conversation. "
            "Override this method to perform any cleanup or final processing at the end of a conversation.",
        )

    # ------------------------------------------------------------------ #
    #  Knowledge representation (override in subclasses)
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_knowledge_description(self) -> str:
        """Return a natural-language description of what the learner currently
        knows.  This is injected into LLM prompts instead of the raw
        skill list so the learner's responses are grounded in its own
        knowledge configuration.

        Examples
        --------
        - For the default skill-binary learner this would list mastered
        skill descriptions.
        - For a conversation-history learner this
        would summarize or list past dialogues.

        """

    @abstractmethod
    def get_knowledge_for_problem(
        self,
        practice_item: str | PracticeItem,
        item_skills: list[Skill],
        knowledge_attrs: dict | None = None,
    ) -> str:
        """Return the subset of knowledge relevant to the given problem.
        Used when generating the learner's practice / solution prompts.

        Args:
        ----
            practice_item: The problem text or PracticeItem.
            item_skills: Skills associated with the item.

        Returns:
        -------
            A string to be inserted into prompts describing what the
            learner knows that is relevant to this problem.

        """

    @abstractmethod
    def get_required_knowledge_to_answer_practice_item(
        self,
        practice_item: str | PracticeItem,
        practice_item_skills: list[Skill],
        knowledge_attrs: dict | None = None,
    ) -> str:
        """Return the knowledge from the skill space that the learner would need to answer the problem correctly.
        Used for skill guardrails when deciding whether the learner can learn from a conversation or not.

        Args:
        ----
            practice_item: The problem text or PracticeItem.
            practice_item_skills: Skills associated with the item.
            knowledge_attrs: Optional dict of additional attributes relevant for determining the required knowledge

        """

    @abstractmethod
    def update_knowledge_from_conversation(
        self,
        dialogue_history: str,
    ) -> None:
        """Update the learner's internal knowledge representation after a
        conversation. Called AFTER the skill guardrails have already
        decided that knowledge CAN be updated, i.e. prerequisite skills are already mastered.

        Args:
        ----
            dialogue_history: Rendered conversation transcript.

        """

    def initialize_from_skills(
        self,
        mastered_skill_ids: list[str],
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement initialize_from_skills. "
            "Override this method to set up the learner's initial knowledge state from a list of skill IDs.",
        )

    def set_up_initialization_tutor(self, **kwargs: Any) -> BaseTutor | None:
        pass

    @abstractmethod
    def initialize_learner_knowledge(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the learner's knowledge representation based on the current mastered skills and any additional kwargs.

        This is called after the learner is initialized with its initial set
        of mastered skills, and can be used to set up the learner's internal
        knowledge state to match what it should know based on those skills.

        For example, a simulated learner with a knowledge base might populate
        that knowledge base with information related to the initially mastered skills. A conversation-history-based learner might initialize its conversation history with past dialogues related to the initially mastered skills.
        """
        ...

    # ------------------------------------------------------------------ #
    #  Prompt generation hooks (override for custom prompt style)
    # ------------------------------------------------------------------ #

    def get_misconceptions_for_skills(self, item_skills: list[Skill]) -> str:
        """Return a formatted string of active misconceptions relevant to the given skills.

        Only includes misconceptions for skills that the learner has NOT yet mastered.
        Returns an empty string when no relevant misconceptions are active.
        """
        relevant = []
        for skill in item_skills:
            if skill.id not in self.mastered_skills and skill.id in self.active_misconceptions:
                relevant.append(
                    f"- {skill.description}: {self.active_misconceptions[skill.id]}",
                )
        if not relevant:
            return ""
        return "\n".join(relevant)

    def _get_response_length_instruction(self) -> str:
        """Return a response-length style instruction based on the learner's persona.

        Returns an empty string when no ``response_length`` persona trait is set,
        so callers that do not provide a persona are unaffected.
        """
        response_length = self.persona.get("response_length", "")
        if response_length == "wordy":
            return (
                "Respond in a verbose, elaborate way: "
                "write multiple sentences, and explain your thinking "
                "-even if it is wrong- in detail before asking your question or making your attempt."
            )
        if response_length == "concise":
            return (
                "Respond briefly, in a conversational way, to the point: "
                "use as few words as possible while still being clear."
            )
        return ""

    def get_practice_prompt(
        self,
        practice_item_text: str,
        item_skills: list[Skill],
        conversation_history: str,
        current_confusion: str,
        knowledge_attrs: dict | None = None,
    ) -> dict[str, str]:
        """Return system and user prompts for the practice conversation node.

        Override this to change how the learner formulates questions /
        shows confusion during practice.

        Returns
        -------
            Dict with keys ``"system"`` and ``"user"``.

        """
        knowledge = self.get_knowledge_for_problem(
            practice_item_text,
            item_skills,
            knowledge_attrs,
        )

        response_length_instruction = self._get_response_length_instruction()
        style_line = f"\nResponse style: {response_length_instruction}" if response_length_instruction else ""

        misconceptions_text = self.get_misconceptions_for_skills(item_skills)
        misconceptions_line = (
            f"\nYou also have the following misconceptions that affect your reasoning:\n{misconceptions_text}"
            if misconceptions_text
            else ""
        )

        if current_confusion == "":
            user_prompt = f"""
You are a novice student learning mathematics and working on the math problem below.
Considering your current knowledge below, and your current conversation with a tutor, either:
- Ask a simple naive question about the content if you have very little relevant knowledge.
- If you lack specific knowledge, ask a focused, related question based on the problem.
- If you have most of the relevant knowledge, attempt to solve the problem but make a mistake related to what you don't know.
{style_line}

Math problem:
{practice_item_text}

Your current relevant knowledge:
{knowledge}
{misconceptions_line}

Current conversation:
{conversation_history}

Return your reasoning, your current confusion, and your response.
"""
        else:
            user_prompt = f"""
You are a novice student learning mathematics and working on the math problem below.
Considering your current knowledge, respond to the tutor by continuing with your current confusion.
You are currently confused about: {current_confusion}.
{style_line}

Math problem:
{practice_item_text}

Your current relevant knowledge:
{knowledge}
{misconceptions_line}

Current conversation:
{conversation_history}

Return your reasoning and your response to the tutor.
"""
        return {
            "system": "You are a novice learner with growing knowledge working on a math problem.",
            "user": user_prompt,
        }

    def get_solution_prompt(
        self,
        practice_item_text: str,
        item_skills: list[Skill],
        conversation_history: str,
        knowledge_attrs: dict | None = None,
    ) -> dict[str, str]:
        """Return system and user prompts for the solution proposal node.

        Override this to change how the learner proposes solutions.

        Returns
        -------
            Dict with keys ``"system"`` and ``"user"``.

        """
        knowledge = self.get_knowledge_for_problem(
            practice_item_text,
            item_skills,
            knowledge_attrs,
        )

        response_length_instruction = self._get_response_length_instruction()
        style_line = f"\nResponse style: {response_length_instruction}" if response_length_instruction else ""

        user_prompt = f"""
You are a novice student learning mathematics and working on the math problem below.
Considering your current knowledge and your conversation with a tutor, propose a solution to the problem.
{style_line}

Math problem:
{practice_item_text}

Your current relevant knowledge:
{knowledge}

Current conversation:
{conversation_history}

Return your reasoning and your response to the tutor including your problem solution.
"""
        return {
            "system": "You are a novice learner with a growing set of knowledge.",
            "user": user_prompt,
        }

    # ------------------------------------------------------------------ #
    #  Skill guardrail helpers
    # ------------------------------------------------------------------ #

    def can_learn_skill(self, skill: Skill | str) -> bool:
        """Check whether all prerequisites for *skill* are already mastered.
        This is the guardrail that prevents knowledge updates when
        prerequisites are missing.
        """
        if isinstance(skill, Skill):
            skill_id = skill.id
        else:
            skill_id = skill
        # if skill itself in mastered_skills, we can't learn it again, so return False
        if skill_id in self.mastered_skills:
            return False
        skill = self.skill_space.get_skill(skill_id)
        if skill:
            for preq_id in skill.prerequisites:
                if preq_id not in self.mastered_skills:
                    return False
        else:
            logger.warning(
                f"Skill {skill_id} not found in skill space. Assuming it has no prerequisites.",
            )
        return True

    def get_learnable_skills(self) -> list[Skill]:
        """Return skills whose prerequisites are all mastered but that are
        not yet mastered themselves.
        """
        return [
            skill for skill in self.skill_space if skill.id not in self.mastered_skills and self.can_learn_skill(skill)
        ]

    def forget_skill(self, skill_id: str | Skill) -> None:
        if isinstance(skill_id, Skill):
            skill_id = skill_id.id
        self.mastered_skills.remove(skill_id)

    # ------------------------------------------------------------------ #
    #  Core methods with default implementations (override as needed)
    # ------------------------------------------------------------------ #

    def learn_root_skills(self) -> list[Skill]:
        """Initialize with root skills."""
        root_skills: list[Skill] = []
        if len(self.mastered_skills) == 0:
            root_skills = self.skill_space.get_root_skills()
            for rs in root_skills:
                self.master_new_skill(rs)
        return root_skills

    def master_new_skill(self, skill_id: str | Skill) -> None:
        """Master a new skill. Override to add custom side-effects."""
        if isinstance(skill_id, Skill):
            sid = skill_id.id
        else:
            sid = skill_id
        assert sid in self.skill_space, f"Skill {sid} is not in the SkillSpace."
        if sid not in self.mastered_skills:
            self.mastered_skills.append(sid)
            self.active_misconceptions.pop(sid, None)

    def log_new_practice(self, session_mastered_skills: dict) -> None:
        self.practice_history.append(session_mastered_skills)

    def learns_from_conversation(
        self,
        dialogue_history: list[dict] | str,
        item_skills: list[Skill],
        llm_client: Any | None = None,
        correct_answer: str = "",
        check_if_should_learn: bool = True,
        use_past_conversations: bool = False,
        solved_problem: bool = False,
    ) -> list[Skill] | None:
        """Learns skills after a conversation using skill-binary logic.

        Must respect ``can_learn_skill`` guardrails.
        Must call ``update_knowledge_from_conversation`` after skill mastering to simulate
        the learner updating their knowledge representation after the conversation.

        The learner should update its knowledge only when the learner demonstrated mastery in the conversation
        And not just when the tutor shows the correct answer if the learner does not follow up with a correct attempt.
        The correct attempt demonstration should happen upstream in the conversation, the learning update focus on the learner's mastery demonstration.
        """
        if llm_client is None:
            load_dotenv()
            llm_client = OpenAI()
            if llm_client is None:
                raise ValueError(
                    "LLM client is required to answer practice items. Please provide an OpenAI client instance.",
                )
        try:
            if check_if_should_learn:
                learnable_skills = self.get_learnable_skills()
                learnable_skills = [sk for sk in learnable_skills if sk in item_skills]

                assert not (solved_problem and use_past_conversations), (
                    "If the problem was solved, we do not allow using past conversations."
                )
                if not use_past_conversations and solved_problem:
                    learned = []
                    for sk in learnable_skills:
                        self.master_new_skill(sk)
                        learned.append(sk)
                    if isinstance(dialogue_history, list):
                        dialogue_history_str = render_conversation_messages(
                            dialogue_history,
                            roles_names={"user": "Learner", "assistant": "Tutor"},
                        )
                    else:
                        dialogue_history_str = dialogue_history
                    self.update_knowledge_from_conversation(
                        dialogue_history=dialogue_history_str,
                    )
                    return learned

                mastery_specific_response_guidelines = (
                    "Your response list may be empty or contain one or more skills. "
                    "Focus on the LEARNER's messages: only include a skill if the learner's "
                    "own responses demonstrate that they understood and can apply the concept — "
                    "e.g. the learner correctly restates the procedure, applies it to the problem, "
                    "or gives a correct answer. "
                    "Do NOT include a skill simply because the tutor explained it well; "
                    "the tutor's explanation alone is not evidence of learner mastery."
                )
                if isinstance(dialogue_history, list):
                    conversation_history = render_conversation_messages(
                        messages=dialogue_history,
                        roles_names={"user": "Learner", "assistant": "Tutor"},
                    )
                else:
                    conversation_history = dialogue_history

                past_conversations_string = ""
                if use_past_conversations:
                    past_conversations = []
                    for lsk in learnable_skills:
                        skill_conversations = self.retrieve_practice_conversations_by_skill(
                            skill=lsk,
                        )
                        for conv in skill_conversations:
                            if conv not in past_conversations:
                                past_conversations.append(conv)
                    past_conversations_string = (
                        "\nPast conversations where the learner practiced some of the learnable skills:\n"
                    )
                    if len(past_conversations) > 0:
                        past_conversations_string = "\n\n".join(
                            [
                                f"Past conversation {i + 1}:\n"
                                + render_conversation_messages(
                                    messages=conv,
                                    roles_names={
                                        "user": "Learner",
                                        "assistant": "Tutor",
                                    },
                                )
                                for i, conv in enumerate(past_conversations)
                            ],
                        )
                learnable_skills_list = (
                    "\n".join(
                        [f"{i + 1}. {sk.description}" for i, sk in enumerate(learnable_skills)],
                    )
                    if learnable_skills
                    else "No learnable skills. Answer with []"
                )

                correct_answer_text = ""
                if correct_answer:
                    correct_answer_text = f"\nCorrect answer to the practice item: {correct_answer}\n"

                learning_prompt = f"""
                You are an experienced educator evaluating whether a learner has demonstrated mastery of skills after a tutoring exchange.
                Using the learner's learnable skills list below and the conversation history, identify which skills the LEARNER themselves demonstrated understanding of.
                {mastery_specific_response_guidelines}
                Judge the LEARNER's messages, not the tutor's: a skill is mastered only when the learner's own words show they can correctly apply or explain it.
                Return a list of numbers corresponding to skill numbers in the list below.

                Learnable Skills:
                {learnable_skills_list}
                {correct_answer_text}
                Practice conversation history with the tutor:
                {conversation_history}
                {past_conversations_string}

                Respond with your reasoning and the list of mastered skills' numbers if any or an empty list if none.
                """
                completion = llm_client.beta.chat.completions.parse(
                    model="gpt-4.1-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an experience math teacher who evaluates students' skill mastery levels.",
                        },
                        {"role": "user", "content": learning_prompt},
                    ],
                    response_format=LearningPromptResponse,
                )

                logger.info(
                    "[learns_from_conversation] learned_skills=%s",
                    completion.choices[0].message.parsed.learned_skills,
                )

                learned_skills_ids = [int(i) - 1 for i in completion.choices[0].message.parsed.learned_skills]
                learned_skills = [learnable_skills[i] for i in learned_skills_ids]

                for sk in learned_skills:
                    self.master_new_skill(sk)
                self.update_knowledge_from_conversation(
                    dialogue_history=conversation_history,
                )
                return learned_skills
            else:
                if isinstance(dialogue_history, list):
                    dialogue_history_str = render_conversation_messages(
                        dialogue_history,
                        roles_names={"user": "Learner", "assistant": "Tutor"},
                    )
                else:
                    dialogue_history_str = dialogue_history
                self.update_knowledge_from_conversation(
                    dialogue_history=dialogue_history_str,
                )
                return None
        except Exception:
            logger.exception("[learns_from_conversation] error")
            return None

    def answer_practice_item(
        self,
        practice_item_text: str | None = None,
        practice_item_skills: list[Skill] | None = None,
        answer_choices: list[str] | None = None,
        should_answer_correctly: bool | None = None,
        prompt: str | None = None,
        return_prompt: bool = False,
        llm_client: Any | None = None,
    ) -> str | dict:
        """Answer a practice item."""
        if llm_client is None:
            load_dotenv()
            llm_client = OpenAI()
            if llm_client is None:
                raise ValueError(
                    "LLM client is required to answer practice items. Please provide an OpenAI client instance.",
                )
        try:
            valid_letters: list[str] = []
            if prompt:
                answer_prompt = prompt
                letter_pattern = r"^([A-Z])\.\s"
                for line in prompt.split("\n"):
                    match = re.match(letter_pattern, line.strip())
                    if match:
                        valid_letters.append(match.group(1))
            else:
                if practice_item_text is None:
                    raise ValueError(
                        "Either prompt or practice_item_text must be provided",
                    )
                answer_choices_text = ""
                if answer_choices:
                    valid_letters = [chr(ord("A") + i) for i in range(len(answer_choices))]
                    answer_choices_text = "\n".join(
                        [
                            f"{letter}. {choice}"
                            for letter, choice in zip(
                                valid_letters,
                                answer_choices,
                                strict=False,
                            )
                        ],
                    )
                logger.info(
                    "[answer_practice_item] Building prompt from practice_item_text. "
                    "should_answer_correctly=%s, answer_choices=%s",
                    should_answer_correctly,
                    answer_choices or "none",
                )
                if should_answer_correctly is True:
                    answer_prompt = f"""
You are a student taking a placement test.

Question:
{practice_item_text}

Answer choices:
{answer_choices_text if answer_choices_text else "None provided"}

Solve this problem carefully and correctly.
Select the correct answer choice if choices are provided.
Your final answer MUST be a single letter from the answer choices (e.g., A, B, C).
Reply with ONLY the letter, no other text.
If no choices are provided, return a specific number or measurement (include units).
"""
                elif should_answer_correctly is False:
                    answer_prompt = f"""
You are a student taking a placement test. You do NOT have the skills needed to solve this problem.

Question:
{practice_item_text}

Answer choices:
{answer_choices_text if answer_choices_text else "None provided"}

Make a reasonable attempt that shows errors typical of students who lack this skill.
Select an INCORRECT answer choice.
Your final answer MUST be a single letter from the answer choices (e.g., A, B, C).
Reply with ONLY the letter, no other text.
If no choices are provided, return a specific number or measurement (include units) that is incorrect.
"""
                else:
                    required_knowledge_text = self.get_required_knowledge_to_answer_practice_item(
                        practice_item=practice_item_text,
                        practice_item_skills=practice_item_skills or [],
                        knowledge_attrs=None,
                    )
                    if answer_choices and answer_choices_text:
                        answer_choices_text = (
                            f"Answer choices, preceeded by their A, B, C, or D labels:\n{answer_choices_text}"
                        )
                        response_instructions = (
                            "CRITICAL RULES:\n"
                            "1. If you have the required skill in your mastered list above:\n"
                            "- Solve the problem carefully and correctly\n"
                            "- Select the correct answer choice if choices are provided\n"
                            "- Otherwise, provide the exact numerical answer\n"
                            "2. If you do NOT have the required skill (even if you have prerequisites):\n"
                            "- Make a reasonable attempt that shows errors\n"
                            "- Get the answer wrong\n"
                            "- Show typical student misconceptions\n"
                            "- Select the incorrect answer choice if choices are provided\n"
                            "Your final answer MUST be a single letter **and a single letter only** from the answer choices (e.g., A, B, C) if choices are provided.\n"
                            "Reply with ONLY the letter, no other text."
                        )
                    else:
                        answer_choices_text = ""
                        response_instructions = (
                            "If no choices are provided, return your reasoning and your response with explanations."
                        )
                    answer_prompt = f"""
You are a student taking a placement test.
Answer the question correctly if your knowledge contains enough knowledge related to the skills being tested.
Otherwise, make a reasonable attempt that shows an error typical of students who lack the required skills, and get the answer wrong.

{required_knowledge_text}

Question:
{practice_item_text}

{answer_choices_text}

{response_instructions}
"""

            completion = llm_client.beta.chat.completions.parse(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You generate plausible student responses based on the student's knowledge.",
                    },
                    {"role": "user", "content": answer_prompt},
                ],
                response_format=AnswerResponse,
            )

            parsed = completion.choices[0].message.parsed
            if parsed is None:
                logger.warning(
                    "[answer_practice_item] LLM returned no parsed response.",
                )
                if return_prompt:
                    return {"answer": "", "prompt": answer_prompt}
                return ""

            answer = parsed.answer.strip()
            logger.info(
                "[answer_practice_item] raw answer: '%s' | reasoning: '%s'",
                answer,
                (parsed.reasoning or "")[:120],
            )
            if valid_letters:
                answer = answer.upper()
                if answer not in valid_letters:
                    logger.warning(
                        "[answer_practice_item] Answer '%s' not in valid_letters %s — returning empty.",
                        answer,
                        valid_letters,
                    )
                    if return_prompt:
                        return {"answer": "", "prompt": answer_prompt}
                    return ""

            logger.info("[answer_practice_item] Final answer: '%s'", answer)

            if return_prompt:
                return {"answer": answer, "prompt": answer_prompt}
            return answer

        except Exception:
            logger.exception("[answer_practice_item] error")
            if return_prompt:
                return {"answer": "", "prompt": ""}
            return ""
