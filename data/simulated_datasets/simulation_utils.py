"""Utility functions for post-simulation analysis of FlexLearner datasets.

These helpers operate on the ``all_conversations.jsonl`` file produced by a
simulation run and generate human-readable summaries for inspection.

Usage
-----
    python simulation_utils.py --conversations-file path/to/all_conversations.jsonl
    python simulation_utils.py --conversations-file path/to/all_conversations.jsonl \\
                               --output-file path/to/summary.md
    python simulation_utils.py --pool-dir path/to/pool_directory/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------


def load_conversations(jsonl_path: str | Path) -> list[dict]:
    """Load all conversation records from an ``all_conversations.jsonl`` file.

    Each line is a JSON object expected to contain at least:
    - ``session_id``           – encodes learner ID and conversation index
    - ``item_skills``          – list of skill IDs associated with the item
    - ``mastered_skills_before_conversation`` – skill IDs mastered before this turn
    - ``mastered_skills_from_conversation``   – skill IDs newly mastered after this turn

    Lines that are malformed or missing required fields are skipped with a
    warning printed to stdout.
    """
    records: list[dict] = []
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Conversation file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"  [warn] line {lineno}: JSON parse error — {exc}")
                continue

            # Validate minimum required fields
            missing = [
                f
                for f in (
                    "session_id",
                    "mastered_skills_before_conversation",
                    "mastered_skills_from_conversation",
                )
                if f not in record
            ]
            # item_skill_prerequisites is optional; default to empty list if absent
            if "item_skill_prerequisites" not in record:
                record["item_skill_prerequisites"] = []
            if missing:
                print(
                    f"  [warn] line {lineno}: missing fields {missing}, skipping.",
                )
                continue

            records.append(record)

    return records


# ---------------------------------------------------------------------------
# Grouping & sorting helpers
# ---------------------------------------------------------------------------


def _learner_id_from_session(session_id: str) -> str:
    """Extract learner ID from a session ID like ``'learner_002__conv_001'``."""
    return session_id.split("__conv_")[0] if "__conv_" in session_id else session_id


def _conv_index_from_session(session_id: str) -> int:
    """Extract the conversation index from a session ID.

    Falls back to 0 for session IDs that don't follow the expected pattern.
    """
    if "__conv_" in session_id:
        try:
            return int(session_id.split("__conv_")[1])
        except ValueError:
            pass
    return 0


def group_by_learner(records: list[dict]) -> dict[str, list[dict]]:
    """Return a mapping of learner_id → list[record], ordered by conv index."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        lid = _learner_id_from_session(rec["session_id"])
        groups[lid].append(rec)

    # Sort each learner's conversations by their sequential index
    for lid in groups:
        groups[lid].sort(key=lambda r: _conv_index_from_session(r["session_id"]))

    return dict(groups)


# ---------------------------------------------------------------------------
# Markdown summary generation
# ---------------------------------------------------------------------------


def generate_learning_sequence_summary(
    jsonl_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Parse ``all_conversations.jsonl`` and write a Markdown summary.

    The summary shows, for every learner:
    - their initial skill set (skills mastered before the very first conversation)
    - for each conversation in order: the item's associated skills, and any
      skills newly learned during that conversation

    Parameters
    ----------
    jsonl_path:
        Path to the ``all_conversations.jsonl`` file produced by the simulation.
    output_path:
        Where to write the ``.md`` file.  Defaults to a file named
        ``learning_sequence_summary.md`` placed next to *jsonl_path*.

    Returns
    -------
    Path
        The path to the written Markdown file.

    """
    jsonl_path = Path(jsonl_path)

    if output_path is None:
        output_path = jsonl_path.parent / "learning_sequence_summary.md"
    output_path = Path(output_path)

    records = load_conversations(jsonl_path)
    if not records:
        output_path.write_text(
            "# Learning Sequence Summary\n\n_No conversation records found._\n",
            encoding="utf-8",
        )
        return output_path

    learner_groups = group_by_learner(records)

    lines: list[str] = [
        "# Learning Sequence Summary",
        "",
        f"Source: `{jsonl_path}`  ",
        f"Learners: **{len(learner_groups)}** | Total conversations: **{len(records)}**",
        "",
        "---",
        "",
    ]

    for learner_id, convs in sorted(learner_groups.items()):
        lines.append(f"## Learner `{learner_id}`")
        lines.append("")

        # Initial skills = skills mastered before the very first conversation
        first_conv = convs[0]
        initial_skills: list[str] = sorted(
            first_conv.get("mastered_skills_before_conversation", []),
        )

        if initial_skills:
            lines.append(
                f"**Initial skills ({len(initial_skills)}):** " + ", ".join(f"`{s}`" for s in initial_skills),
            )
        else:
            lines.append("**Initial skills:** _none_")
        lines.append("")

        for conv in convs:
            session_id: str = conv["session_id"]
            conv_idx = _conv_index_from_session(session_id)
            item_skills: list[str] = sorted(conv.get("item_skills", []))
            before: set[str] = set(conv.get("mastered_skills_before_conversation", []))
            after_new: list[str] = sorted(
                conv.get("mastered_skills_from_conversation", []),
            )
            # Only skills that weren't already mastered before this conversation
            newly_learned: list[str] = sorted(s for s in after_new if s not in before)

            lines.append(
                f"### Conversation {conv_idx + 1} — `{session_id}`",
            )
            item_prereqs: list[str] = sorted(conv.get("item_skill_prerequisites", []))

            if item_skills:
                lines.append(
                    "- **Item skills:** " + ", ".join(f"`{s}`" for s in item_skills),
                )
            else:
                lines.append("- **Item skills:** _none recorded_")

            if item_prereqs:
                lines.append(
                    "- **Item skill prerequisites:** " + ", ".join(f"`{s}`" for s in item_prereqs),
                )
            else:
                lines.append("- **Item skill prerequisites:** _none_")

            if newly_learned:
                lines.append(
                    "- **Newly learned:** ✅ " + ", ".join(f"`{s}`" for s in newly_learned),
                )
            else:
                lines.append("- **Newly learned:** —")
            lines.append("")

        # Final skills = skills mastered before last conv + skills learned in last conv
        last_conv = convs[-1]
        final_skills: list[str] = sorted(
            set(last_conv.get("mastered_skills_before_conversation", []))
            | set(last_conv.get("mastered_skills_from_conversation", [])),
        )
        if final_skills:
            lines.append(
                f"**Final skills ({len(final_skills)}):** " + ", ".join(f"`{s}`" for s in final_skills),
            )
        else:
            lines.append("**Final skills:** _none_")
        lines.append("")

        lines.append("---")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Learning sequence summary written to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Markdown learning-sequence summary from an "
            "all_conversations.jsonl file produced by a simulation run."
        ),
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--conversations-file",
        type=Path,
        metavar="FILE",
        help=(
            "Path to the all_conversations.jsonl file to summarise. "
            "Example: data/student_pools/my_pool_20260401-120000/all_conversations.jsonl"
        ),
    )
    source.add_argument(
        "--pool-dir",
        type=Path,
        metavar="DIR",
        help=("Path to a pool directory; the script will look for all_conversations.jsonl inside it automatically."),
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Where to write the .md summary. Defaults to learning_sequence_summary.md next to the conversations file."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for generate_learning_sequence_summary."""
    args = _parse_args()

    if args.conversations_file:
        jsonl_path = args.conversations_file
    else:
        jsonl_path = args.pool_dir / "all_conversations.jsonl"

    if not jsonl_path.exists():
        logger.error("Conversations file not found: %s", jsonl_path)
        sys.exit(1)

    summary_path = generate_learning_sequence_summary(
        jsonl_path=jsonl_path,
        output_path=args.output_file,
    )
    logger.info("Done. Summary written to: %s", summary_path)


if __name__ == "__main__":
    main()
