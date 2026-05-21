"""Export markdown packets for manually reviewing Eedi auto-labeling outputs.

This script samples conversations from a metrics-tagged Eedi conversations JSONL
file and writes three markdown review packets:
- item skill-tagging review
- learner mastery-from-conversation review
- learner talk-move and error-type review

Each packet is saved into the corresponding review folder under
`data/eedi_tutoring/`.

Example:
-------
```bash
uv run python data/eedi_tutoring/export_eedi_tagging_review_samples.py \
  --input data/eedi_tutoring/all_conversations_with_metrics.jsonl \
  --sample-size 20
```

"""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_LOG_LEVEL = "INFO"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "all_conversations_with_metrics.jsonl"
REVIEW_DIRS = {
    "skill": BASE_DIR / "review_item_skill_tagging_process",
    "mastery": BASE_DIR / "review_learner_mastery_from_conv_decision",
    "behavior": BASE_DIR / "review_talk_moves_and_error_types_tagging",
}

logger = logging.getLogger(__name__)


def setup_logging(log_level: str) -> None:
    """Configure simple console logging."""
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
    # uv run python data/eedi_tutoring/export_eedi_tagging_review_samples.py --input data/eedi_tutoring/conversations_sampled_v2_metrics.jsonl --sample-size 30
    parser = argparse.ArgumentParser(
        description="Sample metrics-tagged Eedi conversations and export markdown review packets.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Metrics-tagged conversations JSONL (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of examples to include in each review packet (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    parser.add_argument(
        "--output-prefix",
        default="manual_review",
        help="Filename prefix for generated markdown files (default: manual_review).",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help=f"Console log verbosity (default: {DEFAULT_LOG_LEVEL}).",
    )
    return parser.parse_args()


def _read_jsonl(input_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as handle:
        for line_idx, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"Invalid JSON on line {line_idx} of {input_path}: {exc}"
                raise ValueError(msg) from exc
            if not isinstance(payload, dict):
                msg = f"Expected a JSON object on line {line_idx} of {input_path}."
                raise TypeError(msg)
            records.append(payload)
    return records


def _sample_records(
    records: list[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if sample_size <= 0:
        msg = "--sample-size must be positive."
        raise ValueError(msg)
    if len(records) <= sample_size:
        return sorted(records, key=lambda record: str(record.get("session_id", "")))
    rng = random.Random(seed)
    sampled = rng.sample(records, sample_size)
    return sorted(sampled, key=lambda record: str(record.get("session_id", "")))


def _format_bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _format_list(values: Any) -> str:
    if not isinstance(values, list):
        return "(not available)"
    if not values:
        return "(none)"
    return ", ".join(str(value) for value in values)


def _format_json_block(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)


def _get_tagging_metadata(record: dict[str, Any], key: str) -> dict[str, Any]:
    tagging_metadata = record.get("tagging_metadata")
    if not isinstance(tagging_metadata, dict):
        return {}
    payload = tagging_metadata.get(key)
    return payload if isinstance(payload, dict) else {}


def _render_skill_review(records: list[dict[str, Any]], *, input_path: Path) -> str:
    lines = [
        "# Manual Review Packet: Item Skill Tagging",
        "",
        f"Source JSONL: {input_path}",
        f"Generated at: {datetime.now(tz=UTC).isoformat()}",
        f"Number of examples: {len(records)}",
        "",
        "Review question: is the problem text aligned with the stored skill tag?",
        "",
    ]
    for index, record in enumerate(records, start=1):
        skill_metadata = _get_tagging_metadata(record, "skill_tagging")
        item_skills = record.get("item_skills", [])
        chosen_skill = item_skills[0] if item_skills else "(missing)"
        lines.extend(
            [
                f"## Example {index}: session `{record.get('session_id', '(missing)')}`",
                "",
                f"- learner_id: {record.get('learner_id', '(missing)')}",
                f"- chosen skill: {chosen_skill}",
                f"- item skills: {_format_list(item_skills)}",
                f"- skill prerequisites: {_format_list(record.get('item_skill_prerequisites', []))}",
                f"- saved skill-tag match: {_format_bool(skill_metadata.get('matched', bool(item_skills)))}",
                f"- skill-tag model: {skill_metadata.get('model', '(not available)')}",
                "",
                "### Problem text",
                "",
                record.get("practice_item_text", "(missing)"),
                "",
                "### Stored skill-tagging rationale",
                "",
                skill_metadata.get(
                    "reasoning",
                    "(not available in saved tagging metadata)",
                ),
                "",
                "### Conversation (capped)",
                "",
                "```text",
                str(
                    record.get(
                        "capped_dialogue_history",
                        record.get("dialogue_history", "(missing)"),
                    ),
                ),
                "```",
                "",
                "### Reviewer notes",
                "",
                "- Verdict: [ ] aligned  [ ] not aligned  [ ] unsure",
                "- Notes:",
                "",
                "---",
                "",
            ],
        )
    return "\n".join(lines)


def _render_mastery_review(records: list[dict[str, Any]], *, input_path: Path) -> str:
    lines = [
        "# Manual Review Packet: Learner Mastery From Conversation",
        "",
        f"Source JSONL: {input_path}",
        f"Generated at: {datetime.now(tz=UTC).isoformat()}",
        f"Number of examples: {len(records)}",
        "",
        "Review question: does the stored mastery / no-mastery outcome match the conversation evidence?",
        "",
    ]
    for index, record in enumerate(records, start=1):
        mastery_metadata = _get_tagging_metadata(record, "learning_outcome")
        mastered_from_conv = record.get("mastered_skills_from_conversation", [])
        item_skills = record.get("item_skills", [])
        target_skill = item_skills[0] if item_skills else "(missing)"
        lines.extend(
            [
                f"## Example {index}: session `{record.get('session_id', '(missing)')}`",
                "",
                f"- learner_id: {record.get('learner_id', '(missing)')}",
                f"- target skill: {target_skill}",
                f"- mastered before conversation: {_format_list(record.get('mastered_skills_before_conversation', []))}",
                f"- mastered from conversation: {_format_list(mastered_from_conv)}",
                f"- stored mastery decision: {_format_bool(bool(mastered_from_conv))}",
                f"- mastery model: {mastery_metadata.get('model', '(not available)')}",
                "",
                "### Problem text",
                "",
                record.get("practice_item_text", "(missing)"),
                "",
                "### Stored mastery rationale",
                "",
                mastery_metadata.get(
                    "reasoning",
                    "(not available in saved tagging metadata)",
                ),
                "",
                "### Conversation (capped)",
                "",
                "```text",
                str(
                    record.get(
                        "capped_dialogue_history",
                        record.get("dialogue_history", "(missing)"),
                    ),
                ),
                "```",
                "",
                "### Reviewer notes",
                "",
                "- Verdict: [ ] mastery label looks correct  [ ] incorrect  [ ] unsure",
                "- Notes:",
                "",
                "---",
                "",
            ],
        )
    return "\n".join(lines)


def _render_behavior_review(records: list[dict[str, Any]], *, input_path: Path) -> str:
    lines = [
        "# Manual Review Packet: Learner Talk Moves And Error Types",
        "",
        f"Source JSONL: {input_path}",
        f"Generated at: {datetime.now(tz=UTC).isoformat()}",
        f"Number of examples: {len(records)}",
        "",
        "Review question: do the stored error-type and talk-move labels match the conversation evidence?",
        "",
    ]
    for index, record in enumerate(records, start=1):
        conversation_metrics = record.get("conversation_metrics")
        if not isinstance(conversation_metrics, dict):
            conversation_metrics = {}
        lines.extend(
            [
                f"## Example {index}: session `{record.get('session_id', '(missing)')}`",
                "",
                f"- learner_id: {record.get('learner_id', '(missing)')}",
                f"- target skill(s): {_format_list(record.get('item_skills', []))}",
                f"- questions per interrogative learner turn: {conversation_metrics.get('questions_per_interrogative_turn', '(missing)')}",
                f"- average learner turn length (words): {conversation_metrics.get('avg_words_per_learner_turn', '(missing)')}",
                f"- average learner turn string length (characters): {conversation_metrics.get('avg_learner_turn_string_length', '(missing)')}",
                f"- error count bucket: {conversation_metrics.get('error_count_bucket', '(missing)')}",
                f"- has any talk move: {_format_bool(conversation_metrics.get('has_any_talk_move', False))}",
                f"- classification model: {conversation_metrics.get('classification_model', '(not available)')}",
                "",
                "### Problem text",
                "",
                record.get("practice_item_text", "(missing)"),
                "",
                "### Stored error labels",
                "",
                "```json",
                _format_json_block(conversation_metrics.get("errors", {})),
                "```",
                "",
                "### Stored talk-move labels",
                "",
                "```json",
                _format_json_block(conversation_metrics.get("talk_moves", {})),
                "```",
                "",
                "### Stored classification rationale",
                "",
                conversation_metrics.get(
                    "classification_reasoning",
                    "(not available in saved conversation metrics)",
                ),
                "",
                "### Conversation (capped)",
                "",
                "```text",
                str(
                    record.get(
                        "capped_dialogue_history",
                        record.get("dialogue_history", "(missing)"),
                    ),
                ),
                "```",
                "",
                "### Reviewer notes",
                "",
                "- Verdict: [ ] labels look correct  [ ] incorrect  [ ] partly correct  [ ] unsure",
                "- Notes:",
                "",
                "---",
                "",
            ],
        )
    return "\n".join(lines)


def _write_review_packet(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def main() -> int:
    """Create markdown review packets for sampled tagged conversations."""
    args = _parse_args()
    setup_logging(args.log_level)

    if not args.input.exists():
        logger.error("Input JSONL not found: %s", args.input)
        return 1

    records = _read_jsonl(args.input)
    if not records:
        logger.error("No records found in %s", args.input)
        return 1

    sampled_records = _sample_records(records, args.sample_size, args.seed)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")

    output_paths = {
        "skill": REVIEW_DIRS["skill"]
        / f"{args.output_prefix}_skill_tagging_{timestamp}.md",
        "mastery": REVIEW_DIRS["mastery"]
        / f"{args.output_prefix}_mastery_{timestamp}.md",
        "behavior": REVIEW_DIRS["behavior"]
        / f"{args.output_prefix}_behavior_{timestamp}.md",
    }

    _write_review_packet(
        output_paths["skill"],
        _render_skill_review(sampled_records, input_path=args.input),
    )
    _write_review_packet(
        output_paths["mastery"],
        _render_mastery_review(sampled_records, input_path=args.input),
    )
    _write_review_packet(
        output_paths["behavior"],
        _render_behavior_review(sampled_records, input_path=args.input),
    )

    logger.info("Wrote %d sampled examples to review packets.", len(sampled_records))
    for label, output_path in output_paths.items():
        logger.info("%s review packet → %s", label, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
