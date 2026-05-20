"""Data loading utilities for benchmarks."""

import csv
from pathlib import Path


def get_data_dir() -> Path:
    """Get the data directory path.

    Returns
    -------
        Path to the data directory

    """
    return Path(__file__).parent.parent.parent.parent / "data"


def get_florida_doe_data_dir() -> Path:
    """Get the Florida DOE data directory path.

    Returns
    -------
        Path to the florida-doe data directory

    """
    return get_data_dir() / "florida-doe"


def load_tagged_skill_ids() -> set[str]:
    """Load skill IDs from tagged practice items CSV.

    Returns
    -------
        set of skill IDs found in the tagged items file

    """
    tagged_items_path = (
        get_florida_doe_data_dir()
        / "tagged-practice-items-with-responses.csv"
    )

    skill_ids = set()
    with tagged_items_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            skill_id = (row.get("skill_id") or "").strip()
            if skill_id:
                skill_ids.add(skill_id)
    return skill_ids


def get_tutor_responses_csv_path(
    mocked_tutor_responses_csv_path: str | Path | None = None,
) -> Path:
    """Get path to CSV file with generated helpful tutor responses.

    Returns
    -------
        Path to the helpful responses CSV file

    """
    if mocked_tutor_responses_csv_path:
        return Path(mocked_tutor_responses_csv_path)
    return get_florida_doe_data_dir() / "practice-items-with-mock-responses.csv"


def load_tutor_responses_mapping(
    mocked_tutor_responses_csv_path: str | Path | None = None,
) -> dict:
    """Load helpful and unhelpful responses from CSV and create a mapping from problem text to responses.

    Returns
    -------
        dict: Mapping from problem text to response dict with 'helpful_response', 'unhelpful_response', and 'skill_id'

    """
    csv_path = get_tutor_responses_csv_path(
        mocked_tutor_responses_csv_path=mocked_tutor_responses_csv_path,
    )
    responses_map = {}

    if not csv_path.exists():
        # If the CSV doesn't exist yet, return empty mapping
        return responses_map

    with csv_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            problem = row.get("problem", "").strip()
            helpful_response = row.get("helpful_response", "").strip()
            unhelpful_response = row.get("unhelpful_response", "").strip()
            learner_response_helpful = row.get("learner_response_helpful", "").strip()
            learner_response_unhelpful = row.get(
                "learner_response_unhelpful",
                "",
            ).strip()
            skill_id = row.get("skill_id", "").strip()

            if problem:
                responses_map[problem] = {
                    "helpful_response": helpful_response,
                    "unhelpful_response": unhelpful_response,
                    "learner_response_helpful": learner_response_helpful,
                    "learner_response_unhelpful": learner_response_unhelpful,
                    "skill_id": skill_id,
                }

    return responses_map


def get_benchmark_output_dir() -> Path:
    """Get the benchmark evaluation output directory.

    Creates the directory if it doesn't exist.

    Returns
    -------
        Path to the evaluations directory

    """
    output_dir = Path(__file__).parent.parent.parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def render_conversation_messages(
    messages: list[dict | object],
    roles_names: dict[str, str] | None = None,
) -> str:
    """Render conversation messages as a formatted string for LLM prompts.

    Supports dict format (``[{"role": ..., "content": ...}]``) and LangGraph
    message objects (``HumanMessage``, ``AIMessage``, etc.).

    Args:
    ----
        messages: List of message dicts or LangGraph message objects
        roles_names: Optional mapping from standard roles to display names.
            Defaults to ``{"user": "user", "assistant": "assistant"}``.

    Returns:
    -------
        String representation of the conversation

    """
    if roles_names is None:
        roles_names = {"user": "user", "assistant": "assistant"}

    if not messages:
        return "<Empty conversation history>"

    conv = ""
    for i, msg in enumerate(messages):
        if isinstance(msg, dict):
            role = roles_names.get(msg.get("role", "no_role"), "undefined")
            content = msg.get("content", "no content")
        elif hasattr(msg, "content"):
            msg_type = type(msg).__name__
            if "Human" in msg_type:
                role = roles_names.get("user", "user")
            elif "AI" in msg_type or "Assistant" in msg_type:
                role = roles_names.get("assistant", "assistant")
            else:
                role = msg_type
            content = msg.content
        else:
            role = "unknown"
            content = str(msg)

        conv += f"<<<{i}. {role}: {content}>>>\n"

    return conv
