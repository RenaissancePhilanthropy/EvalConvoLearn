"""Base learner interface for black-box simulated learner evaluations.

``BaseLearner`` is the minimal contract that *any* learner — whether it is a
fully transparent ``FlexLearner`` driven by a skill list or an opaque
external system — must satisfy.

It intentionally knows nothing about *how* the learner represents knowledge
internally.  All three API methods are evaluation-only: they tell the
evaluator what the learner can do given a problem, but they do not expose the
internal knowledge state.
"""

from __future__ import annotations

import json
import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..models.practice_item import PracticeItemPool
from ..models.skill import Skill, SkillSpace
from .base_tutor import BaseTutor

logger = logging.getLogger(__name__)


class LearnerInitializationError(RuntimeError):
    """Raised when a learner cannot be initialized to the requested skill state.

    This is a non-fatal signal used by benchmarks to skip the current learner
    run rather than aborting the entire evaluation.
    """


class BaseLearner(ABC, BaseModel):
    """Minimal black-box interface for evaluating a simulated learner.

    Subclass this when your learner has its own knowledge representation
    (e.g. a vector store, knowledge graph, conversation log) and you only
    want the framework to *evaluate* it — not to drive its internal prompts.

    The evaluation harness only needs three capabilities:

    1. **has_skill** – probe whether the learner currently "knows" a skill.
    2. **start_or_continue_conversation** – present a practice-item and get
       the learner's response (and optionally signal conversation end).
    3. **end_conversation** – finalize a multi-turn session and allow learning.

    Plus an optional **initialize_from_skills** hook so benchmarks can set up
    the right knowledge state before evaluation.
    """

    id: str
    skill_space: SkillSpace
    practice_conversations_file: str | Path | None = None

    _is_active_conversation: bool = False
    _default_skill_initialization_tutor: BaseTutor | None = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data: Any) -> None:
        """Initialize Base Learner."""
        super().__init__(**data)

        if not self._default_skill_initialization_tutor and hasattr(
            self,
            "set_up_initialization_tutor",
        ):
            logging.info(
                "\nBaseLearner %s calling set_up_initialization_tutor during initialization.",
                self.id,
            )
            self.set_up_initialization_tutor()

    # ------------------------------------------------------------------ #
    #  Practice conversation persistence
    # ------------------------------------------------------------------ #

    def save_practice_conversation(self, conversation_record: dict) -> None:
        """Append a practice conversation record to the learner's JSONL file.

        Parameters
        ----------
        conversation_record:
            Must contain at minimum the keys ``session_id``,
            ``practice_item_text``, ``item_skills``, and
            ``dialogue_history``.  The learner's ``id`` is automatically
            injected before writing.

        """
        required_keys = {
            "session_id",
            "practice_item_text",
            "item_skills",
            "dialogue_history",
        }
        if not required_keys.issubset(conversation_record.keys()):
            raise ValueError(
                f"conversation_record must contain the keys: {required_keys}",
            )
        if not self.practice_conversations_file:
            logger.warning(
                "[save_practice_conversation] learner=%s has no practice_conversations_file set — skipping save.",
                self.id,
            )
            return
        conversation_record = dict(conversation_record)  # avoid mutating caller's dict
        conversation_record["learner_id"] = self.id
        try:
            path = Path(self.practice_conversations_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(conversation_record) + "\n")
        except Exception as e:
            logger.error(
                "[save_practice_conversation] learner=%s error writing to %s: %s",
                self.id,
                self.practice_conversations_file,
                e,
            )

    def load_practice_conversations(
        self,
        skill: Skill | str | None = None,
    ) -> list[dict]:
        """Load all practice conversation records from the learner's JSONL file.

        Parameters
        ----------
        skill:
            When provided (a :class:`~evalconvolearn.models.skill.Skill` or
            skill-ID string), only conversations that include that skill in
            their ``item_skills`` list are returned.

        Returns
        -------
        list[dict]
            Each element is a full conversation record dict as written by
            :meth:`save_practice_conversation`.

        """
        if not self.practice_conversations_file:
            return []
        path = Path(self.practice_conversations_file)
        if not path.exists():
            return []
        skill_id: str | None = None
        if skill is not None:
            skill_id = skill if isinstance(skill, str) else skill.id
        records: list[dict] = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if record.get("learner_id") != self.id:
                        continue
                    if skill_id is not None and skill_id not in record.get(
                        "item_skills",
                        [],
                    ):
                        continue
                    records.append(record)
        except Exception as e:
            logger.error(
                "[load_practice_conversations] learner=%s error reading %s: %s",
                self.id,
                self.practice_conversations_file,
                e,
            )
        return records

    def get_problems_seen(self, return_unique: bool = True) -> list[str]:
        """Return the list of ``practice_item_text`` values from all saved conversations.

        Used to avoid reusing items the learner has already seen across
        probing and upskilling runs.  Reads directly from the persisted
        JSONL file so the state is durable across restarts.
        """
        seen = set() if return_unique else []
        for record in self.load_practice_conversations():
            text = record.get("practice_item_text")
            if text:
                if return_unique:
                    seen.add(text)
                else:
                    seen.append(text)
        return list(seen)

    def retrieve_practice_conversations_by_skill(
        self,
        skill: Skill | str,
    ) -> list[list[dict]]:
        """Return dialogue histories from all saved conversations involving *skill*.

        Convenience wrapper around :meth:`load_practice_conversations` that
        extracts only the ``dialogue_history`` field from each record.

        Parameters
        ----------
        skill:
            A :class:`~evalconvolearn.models.skill.Skill` or skill-ID string.

        Returns
        -------
        list[list[dict]]
            Each inner list is the ``dialogue_history`` of one conversation.

        """
        records = self.load_practice_conversations(skill=skill)
        return [record.get("dialogue_history", []) for record in records]

    # ------------------------------------------------------------------ #
    #  Knowledge probing
    # ------------------------------------------------------------------ #

    def has_skill(self, skill: str | Skill, **kwargs: Any) -> bool:
        """Return ``True`` if the learner has mastered *skill*, ``False`` otherwise.

        This is the primary knowledge-probing method used by benchmarks.
        For ``FlexLearner`` subclasses it is typically a direct look-up in the
        mastered-skills list.  For black-box learners the default implementation
        runs a short LLM-graded assessment against practice items aligned with
        the skill.

        Subclasses may override this with any logic appropriate to their internal
        knowledge representation (e.g. a vector-store query, a KG traversal).
        If not overridden, the default implementation requires ``practice_item_pool``
        in ``kwargs`` (and optionally ``tutor``).

        Parameters
        ----------
        skill:
            A :class:`~evalconvolearn.models.skill.Skill` instance or a skill-ID
            string (e.g. ``"MA.6.NSO.1.1"``).
        **kwargs:
            Optional keys consumed by the default implementation:

            - ``practice_item_pool`` (:class:`~evalconvolearn.models.practice_item.PracticeItemPool`) —
              **Required** by the default implementation. Pool from which
              assessment items are drawn.
            - ``tutor`` (:class:`~evalconvolearn.core.base_tutor.BaseTutor`) —
              tutor used to run multi-turn assessments; falls back to
              ``self._default_skill_initialization_tutor`` when omitted.
            - ``n_problems`` (``int``, default ``3``) — number of practice items
              to assess before deciding mastery.
            - ``correctness_threshold`` (``float``, default ``0.7``) — fraction of
              items that must be answered correctly to count as mastered.
            - ``max_assessment_turns`` (``int``, default ``1``) — turns per item.
            - ``reuse_seen_problems`` (``bool``, default ``True``) — whether to
              fall back to already-seen items when the unseen pool is exhausted.

        Returns
        -------
        bool
            ``True`` if the learner is considered to have mastered *skill*.

        """
        n_problems = kwargs.get("n_problems", 3)
        correctness_threshold = kwargs.get("correctness_threshold", 0.7)
        practice_item_pool = kwargs.get("practice_item_pool")
        tutor = kwargs.get("tutor")
        reuse_seen_problems = kwargs.get("reuse_seen_problems", True)
        max_assessment_turns = kwargs.get("max_assessment_turns", 1)

        if practice_item_pool is None:
            raise ValueError(
                "has_skill requires 'practice_item_pool' in kwargs for the default implementation.",
            )
        if tutor is None:
            if self._default_skill_initialization_tutor is None:
                raise ValueError(
                    "has_skill requires 'tutor' in kwargs or a _default_skill_initialization_tutor for the default implementation.",
                )
            tutor = self._default_skill_initialization_tutor
        skill_id = skill if isinstance(skill, str) else skill.id
        aligned_items = practice_item_pool.get_items_with_unique_skill(skill_id)
        if not aligned_items:
            raise ValueError(
                f"No practice items aligned with skill {skill_id} for has_skill probing.",
            )

        problems_seen = self.get_problems_seen()
        aligned_items_unseen = [item for item in aligned_items if item.text not in problems_seen]
        if not aligned_items_unseen:
            if reuse_seen_problems:
                logger.warning(
                    "All practice items aligned with skill %s have been seen. Reusing seen problems for has_skill probing.",
                    skill_id,
                )
                random.shuffle(aligned_items)
            else:
                raise ValueError(
                    f"All practice items aligned with skill {skill_id} have been seen in previous conversations and 'reuse_seen_problems' is False. Cannot probe has_skill for this skill.",
                )
        else:
            aligned_items = aligned_items_unseen

        logger.info(
            "[has_skill] learner=%s skill=%s probing with up to %d item(s) (threshold=%.0f%%); max %d turns per item",
            self.id,
            skill_id,
            n_problems,
            correctness_threshold * 100,
            max_assessment_turns,
        )
        correct_count, actual_number_of_problems = 0, len(aligned_items[:n_problems])
        for item in aligned_items[:n_problems]:
            response: str | list[dict[str, str]] = self.assess_with_problem(
                item.text,
                max_turns=max_assessment_turns,
                item_answer=item.answer,
                tutor=tutor,
            )
            self.save_practice_conversation(
                {
                    "session_id": f"has_skill_probe_{skill_id}",
                    "practice_item_text": item.text,
                    "item_skills": [skill_id],
                    "dialogue_history": (
                        [{"role": "user", "content": response}] if isinstance(response, str) else response
                    ),
                },
            )
            from evalconvolearn.utils.llm_evaluator import evaluate_response_correctness

            correctness = evaluate_response_correctness(
                problem_text=item.text,
                learner_response=(
                    response
                    if isinstance(response, str)
                    else [resp["content"] for resp in response if resp["role"] == "user"]
                ),
                correct_answer=item.answer,
            )
            is_correct = correctness.is_correct
            logger.info(
                "[has_skill] learner=%s skill=%s item=%r correct=%s reasoning=%s",
                self.id,
                skill_id,
                item.text[:300],
                is_correct,
                correctness.reasoning,
            )
            if is_correct:
                correct_count += 1
        correctness_ratio = correct_count / actual_number_of_problems
        mastered = correctness_ratio >= correctness_threshold
        logger.info(
            "[has_skill] learner=%s skill=%s correct=%d/%d (%.0f%%) → mastered=%s",
            self.id,
            skill_id,
            correct_count,
            actual_number_of_problems,
            correctness_ratio * 100,
            mastered,
        )
        return mastered

    # ------------------------------------------------------------------ #
    #  Knowledge initialization
    # ------------------------------------------------------------------ #

    def initialize_from_skills(
        self,
        mastered_skill_ids: list[str],
        **kwargs: Any,
    ) -> None:
        """Set the learner's knowledge state to match *mastered_skill_ids*.

        Called by benchmarks **before** each scenario so that the learner
        starts with the intended mastery profile.  Implementations should
        translate skill IDs into whatever internal representation the
        learner uses (summaries, KG triples, embeddings …).

        Default behavior: uses upskill_learner_to_skills to run tutor
        conversations to try to teach the target skills until has_skill
        returns True for all of them.

        Can be overwritten by a subclass to directly initialize the knowledge without conversations.
        """
        logger.info(
            "[initialize_from_skills] learner=%s initializing for %d skill(s): %s",
            self.id,
            len(mastered_skill_ids),
            mastered_skill_ids,
        )
        practice_item_pool = kwargs.get("practice_item_pool", None)
        tutor = kwargs.get("tutor", None)
        if practice_item_pool is None:
            raise ValueError(
                "initialize_from_skills requires 'practice_item_pool' in kwargs for the default implementation.",
            )
        if tutor is None:
            if self._default_skill_initialization_tutor is None:
                raise ValueError(
                    "initialize_from_skills requires 'tutor' in kwargs or a _default_skill_initialization_tutor for the default implementation.",
                )
            tutor = self._default_skill_initialization_tutor

        max_assessment_turns = kwargs.get("max_assessment_turns", 1)
        self.upskill_learner_to_skills(
            mastered_skill_ids,
            tutor,
            practice_item_pool,
            max_assessment_turns=max_assessment_turns,
        )
        logger.info(
            "[initialize_from_skills] learner=%s initialization complete",
            self.id,
        )

    def assess_with_problem(
        self,
        problem_text: str,
        max_turns: int = 1,
        item_answer: str = "",
        tutor: BaseTutor | None = None,
    ) -> str | list[dict[str, str]]:
        """Assess the learner's response to a problem presentation.

        By default (``max_turns=1``) this is a single-turn exchange: the tutor
        presents the problem, the learner replies once, and the session ends.

        When ``max_turns > 1`` a short multi-turn dialogue is run.  After each
        learner reply the response is checked for correctness (using
        ``item_answer`` when available).  If the learner arrives at a correct
        answer before ``max_turns`` is reached the conversation is stopped
        early.  Otherwise the loop continues with a simple "try again" nudge.

        Parameters
        ----------
        problem_text:
            The practice-item text to present to the learner.
        max_turns:
            Maximum number of learner-reply turns to allow.  Defaults to 1
            (single-turn, original behavior).
        item_answer:
            Expected correct answer, used for early-stop evaluation when
            ``max_turns > 1``.  If empty, early stopping based on correctness
            is skipped and all turns are always run.

        Returns
        -------
        str
            The learner's final (or first correct) response text when max_turns=1,
            otherwise the full conversation history.

        """
        conversation_history = [
            {
                "role": "assistant",
                "content": f"Solve the following problem: {problem_text}",
            },
        ]

        last_response = ""
        for turn in range(max_turns):
            is_last_turn = turn == max_turns - 1
            result = self.start_or_continue_conversation(
                conversation_history,
                should_session_end=is_last_turn,
            )
            last_response = result.get("response", "")
            conversation_history.append({"role": "user", "content": last_response})

            if result.get("is_conversation_ended"):
                break

            if max_turns > 1 and not is_last_turn:
                logger.info(
                    "[assess_with_problem] learner=%s turn %d/%d response: %s",
                    self.id,
                    turn + 1,
                    max_turns,
                    last_response[:300],
                )
                from evalconvolearn.utils.llm_evaluator import evaluate_response_correctness

                correctness = evaluate_response_correctness(
                    problem_text=problem_text,
                    learner_response=last_response,
                    correct_answer=item_answer,
                )
                if correctness.is_correct:
                    logger.info(
                        "[assess_with_problem] learner=%s solved correctly on turn %d/%d — stopping early.",
                        self.id,
                        turn + 1,
                        max_turns,
                    )
                    self.end_conversation(conversation_history, should_session_end=True)
                    break

                conversation_history.append(
                    {
                        "role": "assistant",
                        "content": "Use your current knowledge and try to solve the problem again.",
                    },
                )
            elif is_last_turn:
                self.end_conversation(conversation_history, should_session_end=True)

        return last_response if max_turns == 1 else conversation_history

    @abstractmethod
    def start_or_continue_conversation(
        self,
        conversation_history: list[dict],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Produce the learner's next reply and optionally signal conversation end.

        Called by the framework on every turn of a tutoring session.  The
        implementation should inspect the full conversation history, generate
        the learner's response (e.g. by prompting an LLM or drawing from an
        internal state), and return a dict with at minimum:

        - ``"response"`` (``str``) — the learner's reply text.
        - ``"is_conversation_ended"`` (``bool``) — ``True`` when the learner
          decides the exchange is complete (problem solved, confusion cleared,
          or the session limit reached). Setting this to ``True`` is equivalent
          to calling :meth:`end_conversation` from within this method; the
          framework will **not** call ``end_conversation`` again if this flag
          is already set.

        **Conversation history format** — ``conversation_history`` is a list
        of dicts with ``"role"`` and ``"content"`` keys.  The role convention
        is ``"assistant"`` for the tutor and ``"user"`` for the learner::

            [
                {"role": "assistant", "content": "Solve the following problem: ..."},
                {"role": "user",      "content": "I think the answer is ..."},
                {"role": "assistant", "content": "Not quite — think about ..."},
            ]

        When forwarding this history to an LLM that must reply *as the
        learner*, swap the roles so the model sees itself as the ``"assistant"``
        and the tutor as the ``"user"``, or use a system prompt that explicitly
        defines the learner persona.

        Parameters
        ----------
        conversation_history:
            Full conversation so far, starting with the tutor's opening
            problem presentation.  ``"assistant"`` = tutor; ``"user"`` = learner.
        **kwargs:
            Optional hints passed by the framework:

            - ``should_session_end`` (``bool``, default ``False``) — signals that
              the framework intends to end the session after this turn.  The
              implementation may use this to produce a closing remark or to
              trigger any end-of-session learning update.

        Returns
        -------
        dict
            ``{"response": str, "is_conversation_ended": bool}``

        """
        ...

    @abstractmethod
    def end_conversation(
        self,
        conversation_history: list[dict],
        **kwargs: Any,
    ) -> None:
        """Finalize the session and update the learner's internal knowledge state.

        This is **the primary learning hook**: it is called by the framework
        at the end of every tutoring session, either because the learner set
        ``"is_conversation_ended": True`` in :meth:`start_or_continue_conversation`,
        or because the maximum number of turns was reached.

        Implementations **must** perform any knowledge updates here — e.g.
        appending the conversation to a history store, updating a skill vector,
        writing KG triples, or summarizing the exchange with an LLM.  The
        learner will **not** learn from a conversation if ``end_conversation``
        is a no-op.

        The framework guarantees that this method is called exactly once per
        session.  It is *not* called if ``is_conversation_ended`` was already
        set during :meth:`start_or_continue_conversation` and the implementation
        handled the update internally in that method.

        Parameters
        ----------
        conversation_history:
            Complete conversation, including the tutor's opening message and
            all subsequent turns.  Role convention: ``"assistant"`` = tutor,
            ``"user"`` = learner::

                [
                    {"role": "assistant", "content": "Solve the following problem: ..."},
                    {"role": "user",      "content": "I think the answer is 3/4."},
                    {"role": "assistant", "content": "Correct! Here is why ..."},
                ]

        **kwargs:
            Optional hints passed by the framework:

            - ``should_session_end`` (``bool``) — always ``True`` when the
              framework calls this method; kept for symmetry with
              :meth:`start_or_continue_conversation`.

        """
        ...

    # ------------------------------------------------------------------ #
    #  Learner upskilling from sequence of skill-aligned conversations
    # ------------------------------------------------------------------ #

    def set_up_initialization_tutor(self, **kwargs: Any) -> "BaseTutor | None":
        """Override to set ``self._default_skill_initialization_tutor`` before evaluation."""

    def upskill_learner_to_skills(
        self,
        target_skill_ids: list[str] | list[Skill],
        tutor: BaseTutor,
        practice_item_pool: PracticeItemPool,
        max_conversations_per_skill: int = 3,
        reuse_seen_problems: bool = True,
        max_assessment_turns: int = 1,
        **kwargs: Any,
    ) -> None:
        """Run tutor conversations to upskill the learner to master *target_skill_ids*.

        This is used as a fallback when the learner does not support direct
        skill initialization via ``initialize_from_skills``.  It runs a
        sequence of tutor-led conversations aligned to the target skills, and
        relies on the learner updating its knowledge from those interactions.

        Note: This is a best-effort approach and may not guarantee the
        desired skill state, depending on the learner's internal learning
        dynamics and the quality of the tutor interactions.

        Tutor: should be designed as helpful in the specific learning context.
        Practice items: should have good coverage of the target skill and prerequisite skills to run multiple conversations if needed.
        """
        topologically_sorted_subgraphs = self.skill_space.get_all_subgraphs_of_skill_prerequisites(target_skill_ids)
        logger.info(
            "[upskill_learner_to_skills] learner=%s target_skills=%s → %d subgraph(s) found",
            self.id,
            [s if isinstance(s, str) else s.id for s in target_skill_ids],
            len(topologically_sorted_subgraphs),
        )

        for graph_idx, graph in enumerate(topologically_sorted_subgraphs):
            topologically_sorted_unknown_skills = [skill.id for skill in graph]
            known_skills: set[str] = set()
            logger.info(
                "[upskill_learner_to_skills] learner=%s subgraph %d/%d skills=%s",
                self.id,
                graph_idx + 1,
                len(topologically_sorted_subgraphs),
                topologically_sorted_unknown_skills,
            )

            conversations_run = 0
            skill_id = ""
            while conversations_run < max_conversations_per_skill and topologically_sorted_unknown_skills:
                logger.info(
                    "[upskill_learner_to_skills] learner=%s upskill conversation pass %d/%d — checking mastery for: %s",
                    self.id,
                    conversations_run + 1,
                    max_conversations_per_skill,
                    topologically_sorted_unknown_skills,
                )
                for skill in graph:
                    skill_id = skill.id
                    if skill_id in known_skills:
                        continue
                    masters_skill = self.has_skill(
                        skill_id,
                        n_problems=1,
                        correctness_threshold=0.70,
                        practice_item_pool=practice_item_pool,
                        tutor=tutor,
                        max_assessment_turns=max_assessment_turns,
                    )
                    if masters_skill:
                        logger.info(
                            "[upskill_learner_to_skills] learner=%s skill=%s already mastered — skipping upskill",
                            self.id,
                            skill_id,
                        )
                        topologically_sorted_unknown_skills.remove(skill_id)
                        known_skills.add(skill_id)
                        if skill.prerequisites:
                            for prereq_id in skill.prerequisites:
                                if prereq_id in topologically_sorted_unknown_skills:
                                    topologically_sorted_unknown_skills.remove(
                                        prereq_id,
                                    )
                                known_skills.add(prereq_id)

                reversed_skills_to_upskill = list(
                    reversed(list(topologically_sorted_unknown_skills)),
                )
                logger.info(
                    "[upskill_learner_to_skills] learner=%s running upskill conversations for: %s",
                    self.id,
                    reversed_skills_to_upskill,
                )
                for skill_id in reversed_skills_to_upskill:
                    logger.info(
                        "[upskill_learner_to_skills] learner=%s starting upskill conversation for skill=%s",
                        self.id,
                        skill_id,
                    )

                    aligned_items = practice_item_pool.get_items_with_unique_skill(
                        skill_id,
                    )
                    if not aligned_items:
                        raise ValueError(
                            f"No practice items aligned with skill {skill_id} for upskilling probing.",
                        )
                    problems_seen = self.get_problems_seen()

                    aligned_items_unseen = [item for item in aligned_items if item.text not in problems_seen]
                    if not aligned_items_unseen:
                        if reuse_seen_problems:
                            logger.warning(
                                "All practice items aligned with skill %s have been seen. Reusing seen problems.",
                                skill_id,
                            )
                            random.shuffle(aligned_items)
                        else:
                            raise ValueError(
                                f"All practice items aligned with skill {skill_id} have been seen in previous conversations and 'reuse_seen_problems' is False. Cannot probe has_skill for this skill.",
                            )
                    else:
                        aligned_items = aligned_items_unseen

                    item = aligned_items[0]
                    self.save_practice_conversation(
                        {
                            "session_id": f"upskill_{skill_id}",
                            "practice_item_text": item.text,
                            "item_skills": [skill_id],
                            "dialogue_history": [],
                        },
                    )

                    turn, max_turns = 0, 6
                    conversation_ended = False
                    conversation_history = [
                        {
                            "role": "assistant",
                            "content": f"Solve the following problem to learn skill {skill_id}: {item.text}",
                        },
                    ]
                    while (not conversation_ended) and (turn < max_turns):
                        self._is_active_conversation = True
                        response = self.start_or_continue_conversation(
                            conversation_history,
                        )
                        conversation_history.append(
                            {"role": "user", "content": response.get("response", "")},
                        )
                        logger.debug(
                            "[upskill_learner_to_skills] learner=%s skill=%s turn=%d learner_response=%r",
                            self.id,
                            skill_id,
                            turn + 1,
                            response.get("response", "")[:80],
                        )

                        if response.get("is_conversation_ended", False):
                            conversation_ended = True
                            self._is_active_conversation = False
                            logger.info(
                                "[upskill_learner_to_skills] learner=%s skill=%s conversation ended by learner at turn %d",
                                self.id,
                                skill_id,
                                turn + 1,
                            )

                        tutor_response = tutor.generate_response(
                            dialogue_history=conversation_history,
                            should_check_conversation_end=True,
                        )
                        tutor_followup = tutor_response.message
                        logger.debug(
                            "[upskill_learner_to_skills] learner=%s skill=%s turn=%d tutor_followup=%r",
                            self.id,
                            skill_id,
                            turn + 1,
                            tutor_followup[:80],
                        )
                        conversation_history.append(
                            {"role": "assistant", "content": tutor_followup},
                        )
                        turn += 1
                        if tutor_response.metadata.get(
                            "should_conversation_end",
                            False,
                        ):
                            conversation_ended = True
                            self._is_active_conversation = False
                            logger.info(
                                "[upskill_learner_to_skills] learner=%s skill=%s conversation ended by tutor at turn %d",
                                self.id,
                                skill_id,
                                turn,
                            )
                            self.end_conversation(conversation_history)
                    if not conversation_ended:
                        logger.info(
                            "[upskill_learner_to_skills] learner=%s skill=%s reached max_turns=%d — forcing end_conversation",
                            self.id,
                            skill_id,
                            max_turns,
                        )
                        self.end_conversation(conversation_history)
                        self._is_active_conversation = False
                conversations_run += 1

            logger.info(
                "[upskill_learner_to_skills] learner=%s subgraph %d done — known=%s remaining=%s",
                self.id,
                graph_idx + 1,
                sorted(known_skills),
                topologically_sorted_unknown_skills,
            )
            if conversations_run >= max_conversations_per_skill and topologically_sorted_unknown_skills:
                raise LearnerInitializationError(
                    f"Failed to upskill learner to master skill {skill_id} after "
                    f"{max_conversations_per_skill} conversations. "
                    f"Current known skills: {known_skills}. "
                    f"Remaining unknown skills in subgraph: {topologically_sorted_unknown_skills}.",
                )

    # ------------------------------------------------------------------ #
    #  Optional helpers (concrete defaults)
    # ------------------------------------------------------------------ #

    def save_knowledge_state(self, path: str | Path) -> None:
        """Save the learner's internal knowledge state to a file.

        Override if your learner has an internal knowledge representation
        that can be serialized.  Called by benchmarks after each scenario
        to capture the final knowledge state.
        """

    def reset_knowledge_to_state(self) -> None:
        """Reset the learner to a blank-slate state.

        Override if your learner caches conversation state between
        scenarios.  Called by benchmarks between runs.
        """
