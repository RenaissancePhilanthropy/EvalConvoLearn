"""Utilities for running FlexLearner conversations to completion."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evalconvolearn.models.flexlearner_conversation import ConversationGraph
    from evalconvolearn.models.practice_item import PracticeItem
    from evalconvolearn.models.tutor import Tutor, TutorResponse


def run_conversation_to_completion(
    conversation: ConversationGraph,
    practice_item: PracticeItem,
    session_id: str,
    tutor: Tutor,
    max_turns: int = 10,
) -> dict:
    """Run a conversation to completion and extract metrics."""
    metrics = {
        "turns_to_solution": 0,
        "solution_found": False,
        "max_turns_reached": False,
        "messages": [
            {"role": "system", "content": f"Practice Item: {practice_item.text}"},
        ],
        "final_state": None,
        "tokens_used": {"input_tokens": 0, "output_tokens": 0},
    }
    conversation_ended = False

    full_response = ""
    for chunk in conversation.run_conversation(
        session_id=session_id,
        start_or_resume_conversation="start",
    ):
        full_response += str(chunk)
    metrics["messages"].append({"role": "user", "content": full_response})
    if "Conversation ended" in full_response:
        conversation_ended = True
        metrics["solution_found"] = True

    turn_count = 1
    while not conversation_ended and turn_count < max_turns:
        tutor_response: TutorResponse = tutor.get_teacher_followup_message(
            dialogue_history=metrics["messages"],
        )
        metrics["messages"].append(
            {"role": "assistant", "content": tutor_response.message},
        )

        full_response = ""
        for chunk in conversation.run_conversation(
            session_id=session_id,
            start_or_resume_conversation="resume",
            tutor_message=tutor_response.message,
        ):
            full_response += str(chunk)

        metrics["messages"].append({"role": "user", "content": full_response})
        turn_count += 1
        conversation_ended = conversation_ended or (
            "Conversation ended" in full_response
        )

    if turn_count < max_turns:
        metrics["solution_found"] = True

    if turn_count >= max_turns:
        metrics["max_turns_reached"] = True
        metrics["turns_to_solution"] = max_turns
    else:
        metrics["turns_to_solution"] = turn_count

    final_state = conversation.get_final_state(session_id)
    if final_state and "tokens_used" in final_state:
        metrics["tokens_used"] = final_state["tokens_used"]

    return metrics
