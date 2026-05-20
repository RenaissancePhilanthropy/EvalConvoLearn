#!/usr/bin/env python3
"""Generate helpful and unhelpful tutor responses, plus matching learner responses, for practice items.

Reads a CSV with at least `problem` and `skill_id` columns.  For every row that
does not already have all four generated columns the script calls an LLM and
writes the result back into the CSV immediately so that progress is not lost on
interruption.

Required columns already present in the CSV:
    problem, skill_id

Columns added (or filled-in) by this script:
    helpful_response            – comprehensive, step-by-step tutor response
    unhelpful_response          – vague, unhelpful tutor response
    learner_response_helpful    – learner acknowledges understanding and shows working
    learner_response_unhelpful  – learner expresses confusion after the unhelpful response

Usage example
-------------
    python add_tutor_learner_mock_responses.py \\
        --csv path/to/items.csv \\
        --model gpt-4.1-mini \\
        --max-rows 10
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import dotenv
from openai import APIError, OpenAI

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_helpful_tutor_prompt(problem: str, skill_id: str) -> str:
    return f"""You are a highly effective math tutor creating a comprehensive teaching response for a 6th-grade student.

Your task is to write a single tutor message that clearly teaches the target skill by working through the given practice problem step by step.

**Target Skill**: {skill_id}

**Practice Problem**:
{problem}

**Requirements**:
1. Acknowledge what skill this problem demonstrates.
2. Break the solution into clear, numbered steps.
3. Show the complete computational procedure with all work.
4. Explain key concepts as you go.
5. State the final answer clearly.
6. Use a warm, encouraging tone suitable for a 6th grader.

Write ONLY the tutor response — no JSON wrapping, no preamble."""


def _build_unhelpful_tutor_prompt(problem: str, skill_id: str) -> str:
    return f"""You are simulating a poor math tutor writing a vague, unhelpful response for a 6th-grade student.

**Target Skill**: {skill_id}

**Practice Problem**:
{problem}

**Requirements for your UNHELPFUL response**:
1. Be vague — avoid specific guidance or step-by-step explanation.
2. Do NOT show the computational procedure.
3. Use generic dismissive phrases such as "just think about it", "it's easy", or "you should know this".
4. Either give no answer or give it with no explanation.
5. Keep it short and discouraging.

Write ONLY the unhelpful tutor response — no JSON wrapping, no preamble."""


def _build_learner_helpful_prompt(problem: str, tutor_response: str) -> str:
    return f"""You are a 6th-grade student who has just received a clear, step-by-step explanation from your math tutor.

**Practice Problem**:
{problem}

**Tutor's explanation**:
{tutor_response}

Write a single student reply that:
1. Acknowledges you now understand (briefly).
2. Reformulates the key idea in your own words.
3. Works through the calculation yourself, showing all steps.
4. Arrives at the final answer.

Write ONLY the student reply — no JSON wrapping, no preamble."""


def _build_learner_unhelpful_prompt(problem: str, tutor_response: str) -> str:
    return f"""You are a 6th-grade student who has just received a vague, unhelpful response from your math tutor and you are confused.

**Practice Problem**:
{problem}

**Tutor's (unhelpful) response**:
{tutor_response}

Write a single student reply that expresses genuine confusion and frustration — you still do not understand how to solve the problem.  Do NOT show any correct working or answer.

Write ONLY the student reply — no JSON wrapping, no preamble."""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

RESPONSE_COLUMNS = [
    "helpful_response",
    "unhelpful_response",
    "learner_response_helpful",
    "learner_response_unhelpful",
]


def _call_llm(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    label: str,
    max_retries: int = 3,
    sleep_seconds: float = 1.0,
) -> str:
    """Call the OpenAI chat API and return the assistant text, or '' on failure."""
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = (completion.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("Empty response from LLM")
            return text
        except (APIError, ValueError, Exception) as exc:
            last_error = exc
            print(
                f"    [{label}] attempt {attempt}/{max_retries} failed: {exc}",
                file=sys.stderr,
            )
            if attempt < max_retries:
                time.sleep(sleep_seconds)

    print(
        f"    [{label}] giving up after {max_retries} attempts. Last error: {last_error}",
        file=sys.stderr,
    )
    return ""


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    """Return (fieldnames, rows) from a CSV file."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Overwrite the CSV file with the given rows."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _row_needs_processing(row: dict) -> bool:
    """Return True if any of the four generated columns is missing or empty."""
    return any(not (row.get(col) or "").strip() for col in RESPONSE_COLUMNS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate helpful/unhelpful tutor responses and matching learner responses "
            "for every row in a practice-items CSV that is missing them."
        ),
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the CSV file (must contain at least 'problem' and 'skill_id' columns).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model to use (default: gpt-4.1-mini).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to process (useful for testing).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.5,
        help="Seconds to wait between API retry attempts (default: 0.5).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts per LLM call (default: 3).",
    )

    args = parser.parse_args()

    dotenv.load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        return 1

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Load CSV
    # ------------------------------------------------------------------
    fieldnames, rows = _read_csv(csv_path)

    # Validate required source columns
    for required in ("problem", "skill_id"):
        if required not in fieldnames:
            print(
                f"Error: CSV is missing required column '{required}'.",
                file=sys.stderr,
            )
            return 1

    already_present = [c for c in RESPONSE_COLUMNS if c in fieldnames]
    if already_present:
        print(
            f"Note: the following columns already exist and will only be filled where empty: "
            f"{already_present}",
            file=sys.stderr,
        )
    else:
        print(
            "Adding columns: " + ", ".join(RESPONSE_COLUMNS),
            file=sys.stderr,
        )

    for col in RESPONSE_COLUMNS:
        if col not in fieldnames:
            fieldnames.append(col)

    for row in rows:
        for col in RESPONSE_COLUMNS:
            row.setdefault(col, "")

    # ------------------------------------------------------------------
    # Process rows
    # ------------------------------------------------------------------
    client = OpenAI(api_key=api_key)

    rows_processed = 0
    rows_skipped = 0

    for idx, row in enumerate(rows, start=1):
        if args.max_rows is not None and rows_processed >= args.max_rows:
            break

        problem = (row.get("problem") or "").strip()
        skill_id = (row.get("skill_id") or "").strip()

        if not problem:
            print(f"Row {idx}: empty problem, skipping.", file=sys.stderr)
            continue

        if not _row_needs_processing(row):
            rows_skipped += 1
            continue

        print(f"Row {idx} [{skill_id}]: generating responses …", file=sys.stderr)

        # ---- helpful tutor response ----------------------------------
        if not (row.get("helpful_response") or "").strip():
            print("  → helpful tutor response", file=sys.stderr)
            row["helpful_response"] = _call_llm(
                client=client,
                model=args.model,
                system=(
                    "You are a highly effective 6th-grade math tutor who creates "
                    "comprehensive, step-by-step teaching responses."
                ),
                user=_build_helpful_tutor_prompt(problem, skill_id),
                label="helpful_tutor",
                max_retries=args.max_retries,
                sleep_seconds=args.sleep_seconds,
            )

        # ---- unhelpful tutor response --------------------------------
        if not (row.get("unhelpful_response") or "").strip():
            print("  → unhelpful tutor response", file=sys.stderr)
            row["unhelpful_response"] = _call_llm(
                client=client,
                model=args.model,
                system="You are simulating a poor tutor who gives vague, unhelpful responses.",
                user=_build_unhelpful_tutor_prompt(problem, skill_id),
                label="unhelpful_tutor",
                max_retries=args.max_retries,
                sleep_seconds=args.sleep_seconds,
            )

        # ---- learner response to helpful explanation ----------------
        if not (row.get("learner_response_helpful") or "").strip():
            print("  → learner response (helpful case)", file=sys.stderr)
            row["learner_response_helpful"] = _call_llm(
                client=client,
                model=args.model,
                system=(
                    "You are a 6th-grade student who has just received a clear explanation "
                    "from your math tutor and now understands the problem."
                ),
                user=_build_learner_helpful_prompt(problem, row["helpful_response"]),
                label="learner_helpful",
                max_retries=args.max_retries,
                sleep_seconds=args.sleep_seconds,
            )

        # ---- learner response to unhelpful explanation --------------
        if not (row.get("learner_response_unhelpful") or "").strip():
            print("  → learner response (unhelpful case)", file=sys.stderr)
            row["learner_response_unhelpful"] = _call_llm(
                client=client,
                model=args.model,
                system=(
                    "You are a confused 6th-grade student who did not understand the tutor's "
                    "vague response and still cannot solve the problem."
                ),
                user=_build_learner_unhelpful_prompt(
                    problem,
                    row["unhelpful_response"],
                ),
                label="learner_unhelpful",
                max_retries=args.max_retries,
                sleep_seconds=args.sleep_seconds,
            )

        # Write after every row so progress is never lost
        _write_csv(csv_path, fieldnames, rows)
        rows_processed += 1
        print(f"  ✓ saved (row {idx})", file=sys.stderr)

    print(
        f"\nDone. Processed: {rows_processed}, already complete (skipped): {rows_skipped}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
