"""Lightweight conversation runner for BaseLearner evaluations.

Unlike :class:`ConversationGraph` (which delegates prompt construction to
:class:`FlexLearner`), this module drives conversations entirely
through the :class:`BaseLearner` surface:

    start_conversation_with_problem  →  continue_conversation  →  end_conversation
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from ..core.base_learner import BaseLearner
from ..core.base_tutor import BaseTutor
from ..models.practice_item import PracticeItem
from ..models.tutor import TutorResponse

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """One learner–tutor exchange."""

    turn_number: int
    learner_response: str
    tutor_response: str | None = None


@dataclass
class BaseConversationResult:
    """Outcome of a full base-learner conversation."""

    session_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    solution_found: bool = False
    final_learner_response: str = ""
    conversation_ended_reason: str = ""
    tokens_used: dict[str, int] = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0},
    )

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    def to_history(self) -> list[dict[str, str]]:
        """Render as a flat message list (assistant = tutor, user = learner)."""
        messages: list[dict[str, str]] = []
        for t in self.turns:
            messages.append({"role": "user", "content": t.learner_response})
            if t.tutor_response:
                messages.append({"role": "assistant", "content": t.tutor_response})
        return messages


def run_base_learner_conversation(
    learner: BaseLearner,
    practice_item: PracticeItem | str,
    tutor: BaseTutor | None = None,
    max_turns: int = 6,
    session_id: str | None = None,
    tutor_responses: list[str] | None = None,
    item_skills: list[str] | None = None,
    save_conversation: bool = True,
    correct_answer: str | None = None,
    tutor_generation_metadata: dict | None = None,
) -> BaseConversationResult:
    """Run a multi-turn conversation using only the BaseLearner interface.

    Parameters
    ----------
    learner:
        The black-box learner.
    practice_item:
        The problem to solve.
    tutor:
        Optional ``BaseTutor`` for generating responses.  If *None*,
        ``tutor_responses`` must supply pre-built replies.
    max_turns:
        Maximum learner–tutor exchanges.
    session_id:
        Optional session identifier (auto-generated if omitted).
    tutor_responses:
        Pre-built tutor replies (used for mocked / scripted conversations).
        When provided, ``tutor`` is ignored.
    item_skills:
        Skill IDs associated with *practice_item*.  Stored in the saved
        conversation record so the item appears in :meth:`get_problems_seen`
        and :meth:`load_practice_conversations` look-ups.
        Falls back to ``practice_item.associated_skills`` when *practice_item*
        is a :class:`~flexlearner.models.practice_item.PracticeItem`.
    save_conversation:
        When ``True`` (default), persist the completed conversation to the
        learner's ``practice_conversations_file`` via
        :meth:`~BaseLearner.save_practice_conversation`.

    Returns
    -------
    BaseConversationResult

    """
    session_id = session_id or str(uuid.uuid4())
    if isinstance(practice_item, PracticeItem):
        problem_text = practice_item.text
        resolved_skills: list[str] = (
            item_skills
            if item_skills is not None
            else list(practice_item.associated_skills)
        )
    else:
        problem_text = practice_item
        resolved_skills = item_skills or []

    result = BaseConversationResult(session_id=session_id)

    # Seed the history with the tutor's opening problem presentation
    opening_message = f"Let's work on the following problem together: {problem_text}"
    conversation_history: list[dict] = [
        {"role": "assistant", "content": opening_message},
    ]

    # --- Turn 1: start ---
    learner_out = learner.start_or_continue_conversation(
        conversation_history=conversation_history,
    )
    learner_response = learner_out.get("response", "")
    conversation_history.append({"role": "user", "content": learner_response})

    # Get first tutor response
    first_tutor_reply = _get_tutor_reply(
        tutor,
        tutor_responses,
        turn_index=0,
        dialogue_history=conversation_history,
        session_id=session_id,
        tutor_generation_metadata=tutor_generation_metadata,
    )
    # do not check for conversation end after first turn reply
    result.turns.append(
        ConversationTurn(
            turn_number=1,
            learner_response=learner_response,
            tutor_response=(
                first_tutor_reply
                if isinstance(first_tutor_reply, str)
                else (first_tutor_reply.message if first_tutor_reply else None)
            ),
        ),
    )

    # --- Subsequent turns ---
    for turn_idx in range(1, max_turns):
        if first_tutor_reply is None and turn_idx == 1:
            # Single-shot mode: no tutor, stop after first response
            break

        tutor_msg = result.turns[-1].tutor_response
        if tutor_msg is None:
            break

        conversation_history.append({"role": "assistant", "content": tutor_msg})
        learner_out = learner.start_or_continue_conversation(
            conversation_history=conversation_history,
        )

        learner_response = learner_out.get("response", "")
        # conversation may end from the learner side.
        is_conversation_ended = learner_out.get("is_conversation_ended", False)
        conversation_history.append({"role": "user", "content": learner_response})

        if is_conversation_ended:
            result.conversation_ended_reason = "learner_ended"
            # still add the final learner turn to the results
            result.turns.append(
                ConversationTurn(
                    turn_number=turn_idx + 1,
                    learner_response=learner_response,
                    tutor_response=None,
                ),
            )
            break

        tutor_reply = _get_tutor_reply(
            tutor,
            tutor_responses,
            turn_index=turn_idx,
            dialogue_history=conversation_history,
            session_id=session_id,
            tutor_generation_metadata=tutor_generation_metadata,
        )

        # conversation may end from the tutor side.
        if hasattr(tutor_reply, "metadata") and tutor_reply.metadata.get(
            "should_conversation_end",
            False,
        ):
            is_conversation_ended = True
            result.conversation_ended_reason = tutor_reply.metadata.get(
                "should_end_reasoning",
                "tutor_ended",
            )
            tutor_reply = tutor_reply.message  # unwrap if using response with end check
            # depending on the tutor, the tutor may end the conversation even when the learner has not found a solution...

        result.turns.append(
            ConversationTurn(
                turn_number=turn_idx + 1,
                learner_response=learner_response,
                tutor_response=(
                    tutor_reply
                    if isinstance(tutor_reply, str)
                    else (tutor_reply.message if tutor_reply else None)
                ),
            ),
        )
        if is_conversation_ended:
            # conversation ended from either side, stop the conversation loop
            break

    # --- End ---
    result.final_learner_response = result.turns[-1].learner_response
    result.conversation_ended_reason = (
        "max_turns"
        if len(result.turns) >= max_turns
        else (result.conversation_ended_reason or "tutor_no_reply")
    )

    logger.info(
        "[Base Learner Conversation] Conversation ended after %d turns. Reason: %s",
        result.num_turns,
        result.conversation_ended_reason,
    )

    learner.end_conversation(conversation_history=conversation_history)

    # we assume that if learner ended the conversation, we can evaluate correctness.
    # if the tutor ended the conversation early, we can also evaluate the correctness.
    if correct_answer or result.conversation_ended_reason != "max_turns":
        from ..utils.llm_evaluator import did_learner_find_solution_in_turns

        learner_turns = [t.learner_response for t in result.turns if t.learner_response]
        verdict = did_learner_find_solution_in_turns(
            problem_text=problem_text,
            learner_turns=learner_turns,
            correct_answer=correct_answer,
        )
        result.solution_found = verdict.solution_found

    # Persist the conversation so subsequent has_skill / upskill calls
    # can detect this item as already seen and avoid reusing it.
    if save_conversation:
        learner.save_practice_conversation(
            {
                "session_id": session_id,
                "practice_item_text": problem_text,
                "item_skills": resolved_skills,
                "dialogue_history": conversation_history,
                "num_turns": result.num_turns,
                "conversation_ended_reason": result.conversation_ended_reason,
            },
        )

    return result


def _get_tutor_reply(
    tutor: BaseTutor | None,
    scripted: list[str] | None,
    turn_index: int,
    dialogue_history: list[dict],
    session_id: str | None = None,
    tutor_generation_metadata: dict | None = None,
) -> str | None | TutorResponse:
    """Return the next tutor reply from a script or a live tutor."""
    if scripted is not None:
        if turn_index < len(scripted):
            return scripted[turn_index]
        return None
    if tutor is None:
        return None
    # Use the tutor's generate response (simplified; adapt to your Tutor API)
    try:
        # Subclass 'BaseTutor' and implement the response generation with extra kwargs
        # CAN NOT use 'Tutor' here because it uses student_pool_id etc. which are not available for a BaseLearner.
        extra_kwargs: dict = {
            "should_check_conversation_end": True,
            "session_id": session_id or f"session_{uuid.uuid4()}",
        }
        if tutor_generation_metadata:
            extra_kwargs["tutor_generation_metadata"] = tutor_generation_metadata
        response = tutor.generate_response(
            dialogue_history=dialogue_history,
            **extra_kwargs,
        )
        return response
    except Exception as e:
        logger.warning("Tutor response generation failed: %s", e)
        return None
