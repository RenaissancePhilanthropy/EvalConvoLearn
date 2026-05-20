"""Extract, skill-tag, and save Eedi tutoring conversations as JSONL.

This script:
1. Loads the two Eedi tutoring datasets used in
    `process_tutoring_data.ipynb`.
2. Groups messages by `InterventionId`, reads the tutor identifier from
    `UserId`, and keeps only dialogues where at least a minimum share of turns
    are learner turns (`IsTutor == 0`).
3. Computes how many eligible conversations each tutor has, samples tutors,
    and keeps at least a configured minimum number of conversations per sampled
    tutor.
4. Merges each dialogue with its corresponding question text (`Sequence == 1`).
4.5. Uses an LLM to discard any sampled question that refers to an external
    image, file, or graph that is required to answer it.
5. Uses an LLM to align each question to the closest skill in the current
    Florida DOE skill space, keeping only matched items.  The two example
    problems stored for each skill are included in the prompt to help the LLM.
6. Uses an LLM to infer whether the learner appears to master that skill by
    the end of the tutoring exchange.
7. Filters the tagged output so only tutors with at least a configured minimum
    number of tagged conversations remain.
8. Saves the remaining conversations in an `all_conversations.jsonl`-style
    format, including the true `tutor_id` separately from `learner_id`.

Example:
-------
```bash
uv run python data/evaluations/source_data/eedi_tutoring/extract_tag_store_eedi_tutoring_conversations.py \
  --sample-size 100 \
  --output data/evaluations/source_data/eedi_tutoring/all_conversations.jsonl
```

"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

EEDI_DATASET_NAME = "Eedi/Question-Anchored-Tutoring-Dialogues-2k"
EEDI_DIALOGUE_CONFIG = "anchored-dialogues"
EEDI_QUESTION_CONFIG = "dq-question-metadata"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ROOT_DIR = Path(__file__).resolve().parents[4]
DEFAULT_SKILL_SPACE = ROOT_DIR / "data" / "florida-doe" / "skill-space.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("all_conversations.jsonl")

StructuredResponseT = TypeVar("StructuredResponseT", bound=BaseModel)
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SkillSpec:
    """Represents one skill-space entry."""

    skill_id: str
    description: str
    prerequisite_skills: list[str]
    problem_1: str = ""
    problem_2: str = ""


class SkillTagVerdict(BaseModel):
    """LLM verdict for matching a question to the skill space."""

    reasoning: str
    matched: bool
    skill_id: str = ""


class ExternalReferenceItemVerdict(BaseModel):
    """Per-question verdict within a batch external-reference check."""

    question_index: int
    reasoning: str
    requires_external_reference: bool


class ExternalReferenceBatchVerdict(BaseModel):
    """LLM batch verdict for whether questions require an external reference."""

    verdicts: list[ExternalReferenceItemVerdict]


class LearningOutcomeVerdict(BaseModel):
    """LLM verdict for whether the learner appears to have learned the skill."""

    reasoning: str
    skill_learned: bool


@dataclass(slots=True)
class LLMConfig:
    """Runtime configuration for repeated LLM calls."""

    model: str
    max_retries: int
    sleep_seconds: float


@dataclass(slots=True)
class SamplingConfig:
    """Runtime configuration for tutor-aware dialogue sampling."""

    sample_size: int
    seed: int
    min_student_ratio: float
    min_conversations_per_tutor: int
    min_tagged_conversations_per_tutor: int


def setup_logging(log_level: str) -> None:
    """Configure simple console logging for the extraction pipeline."""
    level_name = log_level.upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        msg = f"Unsupported log level: {log_level}"
        raise TypeError(msg)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Eedi tutoring dialogues, align them to the local skill "
            "space with an LLM, and save matched conversations as JSONL."
        ),
    )
    parser.add_argument(
        "--skill-space",
        type=Path,
        default=DEFAULT_SKILL_SPACE,
        help=f"Path to the skill-space CSV (default: {DEFAULT_SKILL_SPACE}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help=(
            "Maximum number of eligible dialogues to sample before skill filtering "
            "(default: 100)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dialogue sampling (default: 42).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model for skill tagging and outcome judging (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--min-student-ratio",
        type=float,
        default=0.5,
        help="Minimum ratio of student turns required to keep a dialogue (default: 0.5).",
    )
    parser.add_argument(
        "--min-conversations-per-tutor",
        type=int,
        default=5,
        help=(
            "Minimum number of eligible sampled conversations to keep per sampled tutor "
            "before skill filtering (default: 5)."
        ),
    )
    parser.add_argument(
        "--min-tagged-conversations-per-tutor",
        type=int,
        default=5,
        help=(
            "Minimum number of tagged conversations a tutor must retain in the final "
            "output set (default: 5)."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for each LLM call (default: 3).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Seconds to wait between failed LLM attempts (default: 1.0).",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help=("Console log verbosity " f"(default: {DEFAULT_LOG_LEVEL})."),
    )
    return parser.parse_args()


def _write_stderr(message: str) -> None:
    sys.stderr.write(f"{message}\n")


def _write_stdout(message: str) -> None:
    sys.stdout.write(f"{message}\n")


def _validate_args(args: argparse.Namespace) -> str | None:
    checks = (
        (args.sample_size <= 0, "Error: --sample-size must be positive."),
        (
            not 0 <= args.min_student_ratio <= 1,
            "Error: --min-student-ratio must be between 0 and 1.",
        ),
        (
            args.min_conversations_per_tutor <= 0,
            "Error: --min-conversations-per-tutor must be positive.",
        ),
        (
            args.min_tagged_conversations_per_tutor <= 0,
            "Error: --min-tagged-conversations-per-tutor must be positive.",
        ),
        (
            args.sample_size < args.min_conversations_per_tutor,
            "Error: --sample-size must be at least --min-conversations-per-tutor.",
        ),
        (
            not os.getenv("OPENAI_API_KEY"),
            "Error: OPENAI_API_KEY environment variable is not set.",
        ),
    )
    return next((message for condition, message in checks if condition), None)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _sanitize_identifier(value: Any, prefix: str) -> str:
    text = _normalize_text(value)
    if not text:
        return prefix
    text = re.sub(r"^-", "neg_", text)
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    cleaned = cleaned.strip("_")
    return cleaned or prefix


def _parse_prerequisites(raw_value: str) -> list[str]:
    text = raw_value.strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def load_skill_space(skill_space_path: Path) -> dict[str, SkillSpec]:
    """Load the local skill space from CSV."""
    if not skill_space_path.exists():
        msg = f"Skill-space CSV not found: {skill_space_path}"
        raise FileNotFoundError(msg)

    skills: dict[str, SkillSpec] = {}
    with skill_space_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            skill_id = _normalize_text(row.get("skill_id"))
            if not skill_id:
                continue
            skills[skill_id] = SkillSpec(
                skill_id=skill_id,
                description=_normalize_text(row.get("skill_description")),
                prerequisite_skills=_parse_prerequisites(
                    _normalize_text(row.get("prerequisite_skills")),
                ),
                problem_1=_normalize_text(row.get("problem_1")),
                problem_2=_normalize_text(row.get("problem_2")),
            )
    if not skills:
        msg = f"No skills were loaded from {skill_space_path}"
        raise ValueError(msg)
    logger.info("Loaded %d skills from %s", len(skills), skill_space_path)
    return skills


def _build_skill_catalogue(skills: dict[str, SkillSpec]) -> str:
    lines: list[str] = []
    for skill_id in sorted(skills):
        skill = skills[skill_id]
        prereqs = ", ".join(skill.prerequisite_skills) or "none"
        entry = f"- {skill.skill_id}: {skill.description} | prerequisites: {prereqs}"
        if skill.problem_1:
            entry += f"\n    Example 1: {skill.problem_1}"
        if skill.problem_2:
            entry += f"\n    Example 2: {skill.problem_2}"
        lines.append(entry)
    return "\n".join(lines)


def load_eedi_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Eedi dialogue and question metadata splits."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - import guard
        msg = (
            "The `datasets` package is required. Install project dependencies or "
            "`pip install datasets`."
        )
        raise ImportError(msg) from exc

    logger.info(
        "Loading Eedi dialogue split: %s / %s",
        EEDI_DATASET_NAME,
        EEDI_DIALOGUE_CONFIG,
    )
    ad = load_dataset(
        EEDI_DATASET_NAME,
        EEDI_DIALOGUE_CONFIG,
        split="train",
    )
    logger.info(
        "Loading Eedi question split: %s / %s",
        EEDI_DATASET_NAME,
        EEDI_QUESTION_CONFIG,
    )
    dq = load_dataset(
        EEDI_DATASET_NAME,
        EEDI_QUESTION_CONFIG,
        split="train",
    )
    dialogues_frame = cast("pd.DataFrame", ad.to_pandas())
    question_frame = cast("pd.DataFrame", dq.to_pandas())
    logger.info(
        "Loaded %d dialogue rows and %d question rows",
        len(dialogues_frame),
        len(question_frame),
    )
    return dialogues_frame, question_frame


def _coerce_is_tutor(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    text = _normalize_text(value).lower()
    return text in {"1", "true", "t", "yes"}


def _format_dialogue_history(messages: list[dict[str, str]]) -> str:
    return "\n".join(
        f"<<<{index}. {message['speaker']}: {message['content']}>>>"
        for index, message in enumerate(messages)
    )


def _cap_dialogue_history(
    messages: list[dict[str, str]],
    max_turns_per_side: int = 7,
) -> str:
    """Return a formatted dialogue history capped at *max_turns_per_side* turns per speaker.

    Consecutive messages from the same speaker count as a single turn.
    Messages are kept as-is (not merged); the cap only controls how many
    speaker-alternation groups are included for each side.
    """
    turn_counts: dict[str, int] = {}
    prev_speaker: str | None = None
    capped: list[dict[str, str]] = []

    for message in messages:
        speaker = message["speaker"]
        if speaker != prev_speaker:
            turn_counts[speaker] = turn_counts.get(speaker, 0) + 1
            prev_speaker = speaker
        if turn_counts.get(speaker, 0) <= max_turns_per_side:
            capped.append(message)

    return _format_dialogue_history(capped)


def _aggregate_dialogues(
    dialogues_frame: pd.DataFrame,
    *,
    min_student_ratio: float,
) -> list[dict[str, Any]]:
    required_columns = {
        "InterventionId",
        "UserId",
        "QuestionId_DQ",
        "MessageSequence",
        "IsTutor",
        "MessageString",
    }
    missing = required_columns.difference(dialogues_frame.columns)
    if missing:
        msg = f"Dialogue frame is missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    aggregated: list[dict[str, Any]] = []
    sorted_frame = dialogues_frame.sort_values(
        ["InterventionId", "MessageSequence"],
        kind="stable",
    )
    for intervention_id, intervention_group in sorted_frame.groupby(
        "InterventionId",
        sort=False,
    ):
        sorted_group = intervention_group.sort_values("MessageSequence", kind="stable")
        messages: list[dict[str, str]] = []
        student_messages = 0

        for _, row in sorted_group.iterrows():
            message_text = _normalize_text(row.get("MessageString"))
            if not message_text:
                continue
            is_tutor = _coerce_is_tutor(row.get("IsTutor"))
            if not is_tutor:
                student_messages += 1
            messages.append(
                {
                    "speaker": "Tutor" if is_tutor else "Learner",
                    "content": message_text,
                },
            )

        total_messages = len(messages)
        if total_messages == 0:
            continue

        student_ratio = student_messages / total_messages
        if student_ratio < min_student_ratio:
            continue

        first_row = sorted_group.iloc[0]
        aggregated.append(
            {
                "intervention_id": _normalize_text(intervention_id),
                "tutor_id": _normalize_text(first_row.get("UserId")),
                "question_id_dq": _normalize_text(first_row.get("QuestionId_DQ")),
                "messages": messages,
                "dialogue_history": _format_dialogue_history(messages),
                "capped_dialogue_history": _cap_dialogue_history(messages),
                "student_message_ratio": student_ratio,
                "student_messages": student_messages,
                "message_count": total_messages,
            },
        )
    logger.info(
        "Aggregated %d eligible dialogues from %d raw rows using min student ratio %.2f",
        len(aggregated),
        len(dialogues_frame),
        min_student_ratio,
    )
    return aggregated


def _build_question_text_map(question_frame: pd.DataFrame) -> dict[str, str]:
    required_columns = {"QuestionId_DQ", "Text", "Sequence"}
    missing = required_columns.difference(question_frame.columns)
    if missing:
        msg = f"Question frame is missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    filtered = question_frame.copy()
    filtered["Sequence_numeric"] = pd.to_numeric(filtered["Sequence"], errors="coerce")
    filtered = filtered.loc[filtered["Sequence_numeric"] == 1]
    filtered = filtered.sort_values(["QuestionId_DQ", "InterventionId"], kind="stable")

    question_map: dict[str, str] = {}
    for _, row in filtered.iterrows():
        question_id = _normalize_text(row.get("QuestionId_DQ"))
        question_text = _normalize_text(row.get("Text"))
        if question_id and question_text and question_id not in question_map:
            question_map[question_id] = question_text
    logger.info(
        "Built question text map with %d sequence-1 questions",
        len(question_map),
    )
    return question_map


def select_sampled_dialogues(
    dialogues_frame: pd.DataFrame,
    question_frame: pd.DataFrame,
    *,
    sampling_config: SamplingConfig,
) -> list[dict[str, Any]]:
    """Filter, merge, and sample eligible Eedi dialogues by tutor."""
    aggregated = _aggregate_dialogues(
        dialogues_frame,
        min_student_ratio=sampling_config.min_student_ratio,
    )
    question_text_map = _build_question_text_map(question_frame)

    eligible = [
        {
            **dialogue,
            "practice_item_text": question_text_map[dialogue["question_id_dq"]],
        }
        for dialogue in aggregated
        if dialogue["question_id_dq"] in question_text_map
    ]

    logger.info(
        "Found %d dialogues with matching question text before sampling",
        len(eligible),
    )

    dialogues_by_tutor: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for dialogue in eligible:
        dialogues_by_tutor[dialogue["tutor_id"]].append(dialogue)

    eligible_tutor_ids = sorted(
        tutor_id
        for tutor_id, tutor_dialogues in dialogues_by_tutor.items()
        if len(tutor_dialogues) >= sampling_config.min_conversations_per_tutor
    )

    logger.info(
        "Found %d tutors with at least %d eligible conversations",
        len(eligible_tutor_ids),
        sampling_config.min_conversations_per_tutor,
    )

    if not eligible_tutor_ids:
        logger.warning(
            "No tutors meet the minimum pre-tagging conversation threshold of %d",
            sampling_config.min_conversations_per_tutor,
        )
        return []

    max_tutors_by_budget = max(
        1,
        sampling_config.sample_size // sampling_config.min_conversations_per_tutor,
    )
    sampled_tutor_count = min(len(eligible_tutor_ids), max_tutors_by_budget)
    sampled_tutor_ids = (
        pd.Series(eligible_tutor_ids)
        .sample(
            n=sampled_tutor_count,
            random_state=sampling_config.seed,
            replace=False,
        )
        .tolist()
    )

    logger.info(
        "Sampled %d tutors under sample-size budget %d with minimum %d conversations each",
        len(sampled_tutor_ids),
        sampling_config.sample_size,
        sampling_config.min_conversations_per_tutor,
    )

    sampled_dialogues: list[dict[str, Any]] = []
    remaining_dialogues_by_tutor: dict[str, list[dict[str, Any]]] = {}
    remaining_budget = sampling_config.sample_size

    for offset, tutor_id in enumerate(sampled_tutor_ids):
        tutor_dialogues = (
            pd.Series(dialogues_by_tutor[tutor_id])
            .sample(
                n=len(dialogues_by_tutor[tutor_id]),
                random_state=sampling_config.seed + offset,
                replace=False,
            )
            .tolist()
        )
        base_selection = tutor_dialogues[: sampling_config.min_conversations_per_tutor]
        sampled_dialogues.extend(base_selection)
        remaining_dialogues_by_tutor[tutor_id] = tutor_dialogues[
            sampling_config.min_conversations_per_tutor :
        ]
        remaining_budget -= len(base_selection)

    tutor_cycle = sampled_tutor_ids.copy()
    cycle_index = 0
    while remaining_budget > 0 and tutor_cycle:
        tutor_id = tutor_cycle[cycle_index % len(tutor_cycle)]
        remaining_for_tutor = remaining_dialogues_by_tutor[tutor_id]
        if remaining_for_tutor:
            sampled_dialogues.append(remaining_for_tutor.pop(0))
            remaining_budget -= 1
        else:
            tutor_cycle = [
                item for item in tutor_cycle if remaining_dialogues_by_tutor[item]
            ]
            cycle_index = 0
            continue
        cycle_index += 1

    logger.info(
        "Sampled %d dialogues from %d eligible dialogues across %d tutors with seed %d",
        len(sampled_dialogues),
        len(eligible),
        len(sampled_tutor_ids),
        sampling_config.seed,
    )
    return sampled_dialogues


def filter_tagged_records_by_tutor_count(
    records: list[dict[str, Any]],
    *,
    min_tagged_conversations_per_tutor: int,
) -> list[dict[str, Any]]:
    """Keep only tagged records from tutors with enough tagged conversations."""
    counts_by_tutor = collections.Counter(
        _normalize_text(record.get("tutor_id"))
        for record in records
        if _normalize_text(record.get("tutor_id"))
    )
    kept_tutors = {
        tutor_id
        for tutor_id, count in counts_by_tutor.items()
        if count >= min_tagged_conversations_per_tutor
    }
    filtered_records = [
        record
        for record in records
        if _normalize_text(record.get("tutor_id")) in kept_tutors
    ]
    logger.info(
        "Retained %d/%d tagged conversations across %d tutors after applying final tutor threshold %d",
        len(filtered_records),
        len(records),
        len(kept_tutors),
        min_tagged_conversations_per_tutor,
    )
    return filtered_records


def _call_structured_llm(
    client: OpenAI,
    *,
    messages: list[dict[str, str]],
    response_format: type[StructuredResponseT],
    label: str,
    llm_config: LLMConfig,
) -> StructuredResponseT:
    last_error: Exception | None = None
    for attempt in range(1, llm_config.max_retries + 1):
        try:
            completion = client.beta.chat.completions.parse(
                model=llm_config.model,
                messages=messages,
                response_format=response_format,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                msg = "LLM returned no parsed response."
                raise ValueError(msg)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "[%s] attempt %d/%d failed: %s",
                label,
                attempt,
                llm_config.max_retries,
                exc,
            )
            if attempt < llm_config.max_retries:
                time.sleep(llm_config.sleep_seconds)
        else:
            logger.debug("[%s] succeeded on attempt %d", label, attempt)
            return parsed
    msg = f"[{label}] failed after {llm_config.max_retries} attempts: {last_error}"
    raise RuntimeError(msg)


def check_questions_require_external_reference_batch(
    client: OpenAI,
    *,
    questions: list[tuple[str, str]],
    llm_config: LLMConfig,
    batch_size: int = 5,
) -> dict[str, bool]:
    """Check whether questions need an external image/file/graph, in batches.

    Args:
    ----
        questions: Sequence of ``(question_id, question_text)`` pairs to check.
        batch_size: Number of questions to send in each LLM call (default 5).

    Returns:
    -------
        Mapping from ``question_id`` to ``True`` if an external reference is
        required, ``False`` otherwise.  Questions whose index is missing from
        the LLM response default to ``False`` with a warning.

    """
    results: dict[str, bool] = {}
    for batch_start in range(0, len(questions), batch_size):
        batch = questions[batch_start : batch_start + batch_size]
        numbered_questions = "\n\n".join(
            f"[{i}] {text}" for i, (_, text) in enumerate(batch)
        )
        batch_label = f"external-reference-batch-{batch_start // batch_size}"
        logger.debug(
            "[%s] Checking %d question(s) for external references",
            batch_label,
            len(batch),
        )
        verdict: ExternalReferenceBatchVerdict = _call_structured_llm(
            client,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert at reviewing math questions. "
                        "For each numbered question below, determine whether it "
                        "explicitly requires the student to refer to an external image, "
                        "graph, diagram, table, or file that is not embedded as plain "
                        "text within the question itself. "
                        "Set requires_external_reference=true only when the question "
                        "cannot be answered without such a missing external element. "
                        "Questions that merely describe a graph or table in words do not "
                        "require an external reference. "
                        "Return one verdict per question using the question's index."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Questions:\n{numbered_questions}\n\n"
                        "Return a verdict for every question index listed above."
                    ),
                },
            ],
            response_format=ExternalReferenceBatchVerdict,
            label=batch_label,
            llm_config=llm_config,
        )
        for item in verdict.verdicts:
            if 0 <= item.question_index < len(batch):
                qid = batch[item.question_index][0]
                results[qid] = item.requires_external_reference
            else:
                logger.warning(
                    "[%s] Received out-of-range question_index %d (batch size %d); ignoring",
                    batch_label,
                    item.question_index,
                    len(batch),
                )
        for i, (qid, _) in enumerate(batch):
            if qid not in results:
                logger.warning(
                    "[%s] No verdict returned for question index %d (%s); defaulting to False",
                    batch_label,
                    i,
                    qid or "<unknown>",
                )
                results[qid] = False
    return results


def tag_question_to_skill(
    client: OpenAI,
    *,
    question_text: str,
    skill_catalogue: str,
    skills: dict[str, SkillSpec],
    llm_config: LLMConfig,
) -> SkillTagVerdict:
    """Use an LLM to pick the closest matching highest skill, if any."""
    verdict = _call_structured_llm(
        client,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a curriculum alignment expert for middle-school math. "
                    "Given a math question and a skill-space catalogue, choose at most one "
                    "skill that the question is most directly assessing. Prefer the most "
                    "specific/highest skill the student is actively working on, not a mere "
                    "prerequisite. Only set matched=true when the fit is direct and clear. "
                    "If nothing fits, set matched=false and leave skill_id empty."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Skill catalogue:\n{skill_catalogue}\n\n"
                    f"Math question:\n{question_text}\n\n"
                    "Return a structured verdict with reasoning, matched, and skill_id."
                ),
            },
        ],
        response_format=SkillTagVerdict,
        label="skill-tag",
        llm_config=llm_config,
    )
    skill_id = verdict.skill_id.strip()
    if verdict.matched and skill_id in skills:
        return verdict.model_copy(update={"skill_id": skill_id})
    return verdict.model_copy(update={"matched": False, "skill_id": ""})


def infer_learning_outcome(
    client: OpenAI,
    *,
    question_text: str,
    dialogue_history: str,
    skill: SkillSpec,
    llm_config: LLMConfig,
) -> LearningOutcomeVerdict:
    """Use an LLM to judge whether the learner demonstrated the target skill."""
    return _call_structured_llm(
        client,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert math learning scientist reviewing a tutoring "
                    "conversation. Judge whether the learner demonstrates understanding "
                    "of the target skill by the end of the exchange. Be strict: only mark "
                    "skill_learned=true if the learner's own messages show they can now "
                    "apply or explain the target skill correctly. Tutor explanations alone "
                    "are not enough."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Target skill: {skill.skill_id}\n"
                    f"Skill description: {skill.description}\n"
                    "Assumed prerequisite skills already mastered: "
                    f"{', '.join(skill.prerequisite_skills) or 'none'}\n\n"
                    f"Question:\n{question_text}\n\n"
                    f"Conversation:\n{dialogue_history}\n\n"
                    "Return reasoning and whether the learner appears to have learned the target skill."
                ),
            },
        ],
        response_format=LearningOutcomeVerdict,
        label="learning-outcome",
        llm_config=llm_config,
    )


def _build_conversation_record(
    dialogue: dict[str, Any],
    *,
    skill: SkillSpec,
    learned_skill: bool,
    tagging_metadata: dict[str, Any],
    capped_dialogue_history: str,
) -> dict[str, Any]:
    tutor_id_raw = _normalize_text(dialogue.get("tutor_id"))
    tutor_id = _sanitize_identifier(tutor_id_raw, prefix="tutor_unknown")
    learner_id = _sanitize_identifier(
        dialogue.get("intervention_id"),
        prefix="learner_unknown",
    )
    learner_id = f"eedi_learner_{learner_id}"
    intervention_id = _sanitize_identifier(
        dialogue.get("intervention_id"),
        prefix="intervention_unknown",
    )

    return {
        "session_id": f"eedi_tutor_{tutor_id}__{intervention_id}",
        "learner_details": {
            "source": "eedi_tutoring",
            "student_message_ratio": round(
                float(dialogue.get("student_message_ratio", 0.0)),
                3,
            ),
            "message_count": int(dialogue.get("message_count", 0)),
            "student_messages": int(dialogue.get("student_messages", 0)),
            "intervention_id": _normalize_text(dialogue.get("intervention_id")),
            "question_id_dq": _normalize_text(dialogue.get("question_id_dq")),
            "tutor_id": tutor_id_raw,
        },
        "practice_item_text": dialogue["practice_item_text"],
        "dialogue_history": dialogue["dialogue_history"],
        "capped_dialogue_history": capped_dialogue_history,
        "item_skills": [skill.skill_id],
        "item_skill_prerequisites": skill.prerequisite_skills,
        "mastered_skills_before_conversation": skill.prerequisite_skills,
        "mastered_skills_from_conversation": [skill.skill_id] if learned_skill else [],
        "correct_answer": "",
        "learner_id": learner_id,
        "tutor_id": tutor_id_raw,
        "tagging_metadata": tagging_metadata,
    }


def save_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write all records to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), output_path)


def main() -> int:
    """Run the Eedi conversation extraction and tagging pipeline."""
    args = _parse_args()
    setup_logging(args.log_level)
    sampling_config = SamplingConfig(
        sample_size=args.sample_size,
        seed=args.seed,
        min_student_ratio=args.min_student_ratio,
        min_conversations_per_tutor=args.min_conversations_per_tutor,
        min_tagged_conversations_per_tutor=args.min_tagged_conversations_per_tutor,
    )
    llm_config = LLMConfig(
        model=args.model,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
    )

    logger.info(
        "Starting Eedi extraction with sample_size=%d, seed=%d, model=%s",
        args.sample_size,
        args.seed,
        args.model,
    )

    validation_error = _validate_args(args)
    if validation_error is not None:
        _write_stderr(validation_error)
        return 1

    skills = load_skill_space(args.skill_space)
    skill_catalogue = _build_skill_catalogue(skills)
    dialogues_frame, question_frame = load_eedi_frames()
    sampled_dialogues = select_sampled_dialogues(
        dialogues_frame,
        question_frame,
        sampling_config=sampling_config,
    )

    if not sampled_dialogues:
        _write_stderr("No eligible Eedi dialogues were found.")
        return 1

    client = OpenAI()

    # -----------------------------------------------------------------------
    # Step 4.5 – discard questions that require an external image / file / graph
    # -----------------------------------------------------------------------
    seen_question_ids: set[str] = set()
    unique_questions: list[tuple[str, str]] = []
    for dialogue in sampled_dialogues:
        qid = dialogue["question_id_dq"]
        if qid not in seen_question_ids:
            seen_question_ids.add(qid)
            unique_questions.append((qid, dialogue["practice_item_text"]))

    logger.info(
        "Checking %d unique questions for external-reference requirements (batch size 5)",
        len(unique_questions),
    )
    external_ref_flags = check_questions_require_external_reference_batch(
        client,
        questions=unique_questions,
        llm_config=llm_config,
    )

    external_ref_discarded = sum(1 for v in external_ref_flags.values() if v)
    sampled_dialogues = [
        d
        for d in sampled_dialogues
        if not external_ref_flags.get(d["question_id_dq"], False)
    ]
    logger.info(
        "Discarded %d dialogues whose question requires an external reference; "
        "%d remain",
        external_ref_discarded,
        len(sampled_dialogues),
    )

    if not sampled_dialogues:
        _write_stderr(
            "No eligible dialogues remain after external-reference filtering.",
        )
        return 1

    question_skill_cache: dict[str, SkillTagVerdict] = {}
    output_records: list[dict[str, Any]] = []
    unmatched_count = 0
    total_dialogues = len(sampled_dialogues)

    logger.info(
        "Processing %d sampled dialogues across %d unique questions",
        total_dialogues,
        len({dialogue["question_id_dq"] for dialogue in sampled_dialogues}),
    )

    for index, dialogue in enumerate(sampled_dialogues, start=1):
        question_id = dialogue["question_id_dq"]
        if question_id not in question_skill_cache:
            logger.info(
                "[%d/%d] Tagging question %s",
                index,
                total_dialogues,
                question_id or "<unknown>",
            )
            question_skill_cache[question_id] = tag_question_to_skill(
                client,
                question_text=dialogue["practice_item_text"],
                skill_catalogue=skill_catalogue,
                skills=skills,
                llm_config=llm_config,
            )
        else:
            logger.debug(
                "[%d/%d] Reusing cached skill tag for question %s",
                index,
                total_dialogues,
                question_id or "<unknown>",
            )

        skill_verdict = question_skill_cache[question_id]
        if not skill_verdict.matched or skill_verdict.skill_id not in skills:
            unmatched_count += 1
            logger.info(
                "[%d/%d] No matching skill for intervention %s",
                index,
                total_dialogues,
                dialogue["intervention_id"] or "<unknown>",
            )
            continue

        skill = skills[skill_verdict.skill_id]
        logger.info(
            "[%d/%d] Generating learning outcome for intervention %s on skill %s",
            index,
            total_dialogues,
            dialogue["intervention_id"] or "<unknown>",
            skill.skill_id,
        )
        capped_history = dialogue["capped_dialogue_history"]
        learning_verdict = infer_learning_outcome(
            client,
            question_text=dialogue["practice_item_text"],
            dialogue_history=capped_history,
            skill=skill,
            llm_config=llm_config,
        )
        output_records.append(
            _build_conversation_record(
                dialogue,
                skill=skill,
                learned_skill=learning_verdict.skill_learned,
                capped_dialogue_history=capped_history,
                tagging_metadata={
                    "skill_tagging": {
                        "model": llm_config.model,
                        "matched": skill_verdict.matched,
                        "skill_id": skill.skill_id,
                        "reasoning": skill_verdict.reasoning,
                    },
                    "learning_outcome": {
                        "model": llm_config.model,
                        "skill_learned": learning_verdict.skill_learned,
                        "reasoning": learning_verdict.reasoning,
                    },
                },
            ),
        )
        logger.info(
            "[%d/%d] Added conversation record for learner %s (learned=%s)",
            index,
            total_dialogues,
            dialogue["tutor_id"] or "<unknown>",
            learning_verdict.skill_learned,
        )

    filtered_output_records = filter_tagged_records_by_tutor_count(
        output_records,
        min_tagged_conversations_per_tutor=sampling_config.min_tagged_conversations_per_tutor,
    )

    save_jsonl(filtered_output_records, args.output)
    logger.info(
        "Finished processing: %d saved after final tutor filtering, %d discarded after skill tagging",
        len(filtered_output_records),
        unmatched_count,
    )

    _write_stdout(
        f"Eligible sampled dialogues (after external-reference filter): {len(sampled_dialogues)}",
    )
    _write_stdout(f"Discarded due to external reference: {external_ref_discarded}")
    _write_stdout(
        f"Matched dialogues before final tutor filtering: {len(output_records)}",
    )
    _write_stdout(f"Matched and saved dialogues: {len(filtered_output_records)}")
    _write_stdout(f"Discarded after skill tagging: {unmatched_count}")
    _write_stdout(f"Output written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
