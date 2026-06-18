"""LLM-based evaluation helpers for base-learner benchmarks."""

from __future__ import annotations

import logging

from pydantic import BaseModel

from .llm_client import make_client

logger = logging.getLogger(__name__)


class CorrectnessVerdict(BaseModel):
    reasoning: str
    is_correct: bool
    asked_followup: bool = False


class KnowledgeSufficiencyVerdict(BaseModel):
    """LLM verdict on whether a learner's knowledge is sufficient to answer a problem."""

    reasoning: str
    can_answer_correctly: bool


# class LearningVerdict(BaseModel):
#     reasoning: str
#     skill_likely_learned: bool


class SolutionFoundVerdict(BaseModel):
    """LLM verdict on whether the learner found a correct solution across their turns."""

    reasoning: str
    solution_found: bool


class ReusedResolvedMisconceptionVerdict(BaseModel):
    """Whether the learner reused a misconception previously corrected by the tutor."""

    reasoning: str
    reused_incorrectly: bool
    repeated_misconception: str = ""


class ConversationErrorLabels(BaseModel):
    """Conversation-level math error labels."""

    numerical_calculation: bool = False
    conceptual_understanding: bool = False
    problem_comprehension: bool = False
    strategic_decision: bool = False
    step_omission: bool = False


class ConversationTalkMoveLabels(BaseModel):
    """Conversation-level learner talk-move labels."""

    asking_for_more_information: bool = False
    making_a_claim: bool = False
    providing_evidence_or_reasoning: bool = False


class ConversationBehaviorLabelsVerdict(BaseModel):
    """Structured conversation-level labels for errors and talk moves."""

    reasoning: str
    errors: ConversationErrorLabels
    talk_moves: ConversationTalkMoveLabels


def did_learner_find_solution_in_turns(
    problem_text: str,
    learner_turns: list[str],
    correct_answer: str = "",
    model: str = "gpt-4.1-mini",
    computed_conversation_ended_reason: str | None = None,
) -> SolutionFoundVerdict:
    """Judge whether the learner found a correct solution across their conversation turns.

    Unlike ``evaluate_response_correctness`` which checks a single response,
    this examines *all* learner messages in order to determine if the learner
    arrived at a correct solution at any point during the conversation.

    Parameters
    ----------
    problem_text:
        The problem the learner was working on.
    learner_turns:
        All of the learner's messages (in order) from the conversation.
    correct_answer:
        Optional reference answer for the problem.
    model:
        LLM model to use for evaluation.
    computed_conversation_ended_reason:
        The conversation ended reason computed by the system. This can be provided as additional context to the evaluator.

    Returns
    -------
    SolutionFoundVerdict
        With ``solution_found`` flag and ``reasoning``.

    """
    if not learner_turns:
        return SolutionFoundVerdict(
            reasoning="No learner turns provided.",
            solution_found=False,
        )

    client = make_client(model)

    numbered_turns = "\n".join(f"Turn {i + 1}: {turn}" for i, turn in enumerate(learner_turns))

    system = (
        "You are an expert math teacher reviewing a student's messages during a "
        "tutored conversation. Determine whether the student produced a correct "
        "and complete solution to the problem at any point across their messages. "
        "The solution does not need to appear in a single message — the student "
        "may build up to it across turns.\n"
        "If provided, consider the computed conversation ended reason as additional context for your judgment."
    )
    user = f"Problem:\n{problem_text}\n\nStudent messages (in order):\n{numbered_turns}\n\n"
    if correct_answer:
        user += f"Reference correct answer:\n{correct_answer}\n\n"
    if computed_conversation_ended_reason:
        user += f"Computed conversation ended reason:\n{computed_conversation_ended_reason}\n\n"
    user += "Did the student find a correct solution? Return your reasoning and verdict."

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=SolutionFoundVerdict,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        msg = "LLM structured-output call returned no parsed result"
        raise RuntimeError(msg)
    return result


def evaluate_knowledge_sufficiency(
    problem_text: str,
    learner_knowledge: str,
    model: str = "gpt-4.1-mini",
) -> KnowledgeSufficiencyVerdict:
    """Judge whether a learner's knowledge is sufficient to correctly answer a problem.

    Uses the learner's ``get_required_knowledge_to_answer_practice_item`` output
    to determine if the learner has enough knowledge to answer the problem.

    Returns a ``KnowledgeSufficiencyVerdict`` with ``can_answer_correctly`` flag.
    """
    client = make_client(model)

    system = (
        "You are an expert math teacher evaluating whether a student has "
        "sufficient knowledge to correctly answer a problem. "
        "Based on the student's current knowledge description, determine "
        "whether they have enough understanding to solve the problem correctly. "
        "Be strict: if the knowledge is vague, incomplete, or only tangentially "
        "related, the student likely cannot answer correctly."
    )
    user = (
        f"Problem:\n{problem_text}\n\n"
        f"Student's current knowledge:\n{learner_knowledge}\n\n"
        "Based on this knowledge, can the student correctly solve this problem? "
        "Return your reasoning and verdict."
    )

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=KnowledgeSufficiencyVerdict,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        msg = "LLM structured-output call returned no parsed result"
        raise RuntimeError(msg)
    return result


def evaluate_response_correctness(
    problem_text: str,
    learner_response: str | list[str],
    correct_answer: str = "",
    model: str = "gpt-4.1-mini",
) -> CorrectnessVerdict:
    """Judge whether *learner_response* correctly answers *problem_text*.

    Returns a `CorrectnessVerdict` with ``is_correct`` and
    ``asked_followup`` flags.
    """
    client = make_client(model)

    system = (
        "You are an expert math teacher evaluating a student's response to a "
        "problem. Determine whether the response is a correct and complete answer.\n"
        "If the student asks a follow-up question instead of answering, set "
        "asked_followup=True and is_correct=False."
    )
    if isinstance(learner_response, list):
        for lr in learner_response:
            if not isinstance(lr, str):
                raise ValueError(
                    "All elements in learner_response list must be strings.",
                )
        learner_response = "Learner messages:\n" + "\n".join(
            [f"{i + 1}: {lr}" for i, lr in enumerate(learner_response)],
        )
    user = f"Problem:\n{problem_text}\n\nStudent response:\n{learner_response}\n\n"
    if correct_answer:
        user += f"Reference correct answer:\n{correct_answer}\n\n"
    user += "Return your reasoning and verdict."

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=CorrectnessVerdict,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        msg = "LLM structured-output call returned no parsed result"
        raise RuntimeError(msg)
    return result


def _format_dialogue_history(dialogue_history: list[dict[str, str]]) -> str:
    if not dialogue_history:
        return "(empty conversation)"

    # rename user/assistant roles to learner/tutor for clarity in evaluation context:
    roles_mapping = {"user": "Learner", "assistant": "Tutor"}

    return "\n".join(
        f"{index + 1}. {roles_mapping.get(message.get('role', 'unknown'), message.get('role', 'unknown'))}: {message.get('content', '')}"
        for index, message in enumerate(dialogue_history)
    )


def _format_prior_conversations(prior_conversations: list[list[dict[str, str]]]) -> str:
    if not prior_conversations:
        return "No prior conversations."

    formatted_conversations: list[str] = []
    for index, conversation in enumerate(prior_conversations, start=1):
        formatted_conversations.append(
            f"Conversation {index}:\n{_format_dialogue_history(conversation)}",
        )
    return "\n\n".join(formatted_conversations)


def classify_conversation_behaviors(
    dialogue_history: list[dict[str, str]] | str,
    problem_text: str = "",
    correct_answer: str = "",
    model: str = "gpt-4.1-mini",
) -> ConversationBehaviorLabelsVerdict:
    """Classify conversation-level learner errors and talk moves."""
    if isinstance(dialogue_history, list):
        formatted_dialogue = _format_dialogue_history(dialogue_history)
    else:
        formatted_dialogue = dialogue_history.strip() or "(empty conversation)"

    if formatted_dialogue == "(empty conversation)":
        return ConversationBehaviorLabelsVerdict(
            reasoning="No usable conversation was provided.",
            errors=ConversationErrorLabels(),
            talk_moves=ConversationTalkMoveLabels(),
        )

    client = make_client(model)
    system = (
        "You are an expert math learning scientist reviewing a full tutor-learner conversation. "
        "Label only what is supported by the learner's messages. Multiple labels may be true. "
        "Use the tutor responses only as context for interpreting the learner."
    )
    user = (
        f"Problem:\n{problem_text or '(not provided)'}\n\n"
        f"Reference correct answer:\n{correct_answer or '(not provided)'}\n\n"
        f"Conversation:\n{formatted_dialogue}\n\n"
        "Classify whether the learner shows any of these error types anywhere in the conversation:\n"
        "- numerical_calculation: arithmetic or computation mistakes\n"
        "- conceptual_understanding: misunderstanding of the underlying math concept\n"
        "- problem_comprehension: misunderstanding what the problem is asking or what information matters\n"
        "- strategic_decision: choosing an inappropriate method or next step\n"
        "- step_omission: skipping a necessary step or justification\n\n"
        "Also classify whether the learner uses any of these talk moves anywhere in the conversation:\n"
        "- asking_for_more_information\n"
        "- making_a_claim\n"
        "- providing_evidence_or_reasoning\n\n"
        "Return concise reasoning and the boolean labels."
    )

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=ConversationBehaviorLabelsVerdict,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        msg = "LLM structured-output call returned no parsed result"
        raise RuntimeError(msg)
    return result
