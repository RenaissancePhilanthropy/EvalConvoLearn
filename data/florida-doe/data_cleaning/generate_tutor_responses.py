#!/usr/bin/env python3
"""Generate both helpful and unhelpful tutor responses for Florida DOE practice items using Claude."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import dotenv
from openai import APIError, OpenAI


def _clean_text(text: str) -> str:
    """Clean and normalize text input."""
    return text.strip().strip('"').strip("'")


def _build_helpful_response_prompt(
    problem: str,
    skill_id: str,
    prerequisites: str,
) -> str:
    return f"""You are a highly effective math tutor creating a comprehensive teaching response for a 6th-grade student.

Your task is to create a single tutor response that clearly teaches the target skill by working through the given practice problem.

**Target Skill**: {skill_id}
**Prerequisites**: {prerequisites if prerequisites else "None specified"}

**Practice Problem**:
{problem}

**Requirements for your response**:
1. Start by acknowledging what skill this problem demonstrates
2. Break down the problem into clear, numbered steps
3. Show the complete computational procedure with all work
4. Explain key concepts and procedures as you go
5. Provide the final answer clearly
6. Focus on procedural fluency and understanding
7. Write in a warm, encouraging tone suitable for a 6th grader

**Example of the style you should emulate**:

"This juicing plant problem is a great example of computing quotients
of positive fractions with procedural fluency.

Let's break this down:
1. We have a container holding 6⅔ tons of oranges. This is a mixed number: 6 and 2/3 tons.
2. The plant juices 1½ tons per day. This is another mixed number: 1 and 1/2 tons per day.
3. To find how many days it takes to empty the container, we need to divide: (6 2/3) ÷ (1 1/2)

This problem specifically focuses on the computational procedure:
- First, convert mixed numbers to improper fractions:
  6 2/3 = (6×3 + 2)/3 = 20/3
  1 1/2 = (1×2 + 1)/2 = 3/2
- Division of fractions means multiply by the reciprocal:
  (20/3) ÷ (3/2) = (20/3) × (2/3) = 40/9
- Simplify the result: 40/9 = 4 4/9 days

And that's our answer! It will take 4 4/9 days to empty the container."

Now create your comprehensive helpful tutor response. Return ONLY a JSON object with this structure:
{{
    "content": "Your comprehensive tutor response here..."
}}"""


def _build_unhelpful_response_prompt(
    problem: str,
    skill_id: str,
    prerequisites: str,
) -> str:
    return f"""You are simulating a poor math tutor creating an unhelpful response for a 6th-grade student.

Your task is to create a single tutor response that is vague, unhelpful, and does NOT effectively teach the student.

**Target Skill**: {skill_id}
**Prerequisites**: {prerequisites if prerequisites else "None specified"}

**Practice Problem**:
{problem}

**Requirements for your UNHELPFUL response**:
1. Be vague and avoid specific guidance
2. Do NOT break down the problem into clear steps
3. Do NOT show the complete computational procedure
4. Use generic phrases like "just think about it", "it's easy", "you should know this"
5. Either provide no answer or provide it without explanation
6. Be discouraging, dismissive, or overly brief
7. Miss the key teaching moment

**Examples of unhelpful response styles**:

- "This is pretty simple. Just convert the fractions and divide. You should get the answer."
- "Well, you need to solve this using what you learned. Think about it."
- "The answer is 4 4/9 days. Next problem?"
- "Just follow the steps we did before. It's the same thing."
- "I don't know why you're struggling with this, it's basic fraction division."

Now create your vague and unhelpful tutor response. Return ONLY a JSON object with this structure:
{{
    "content": "Your vague and unhelpful tutor response here..."
}}"""


def _generate_response(
    client: OpenAI,
    model: str,
    problem: str,
    skill_id: str,
    prerequisites: str,
    response_type: str,
    max_retries: int = 3,
    sleep_seconds: float = 1.0,
) -> str:
    if response_type == "helpful":
        prompt = _build_helpful_response_prompt(problem, skill_id, prerequisites)
        system_message = "You are a highly effective 6th-grade math tutor who creates comprehensive teaching responses."
    else:
        prompt = _build_unhelpful_response_prompt(problem, skill_id, prerequisites)
        system_message = (
            "You are simulating a poor tutor who gives vague, unhelpful responses."
        )

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                max_tokens=2048,
                messages=[
                    {
                        "role": "system",
                        "content": system_message,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            response_text = completion.choices[0].message.content

            if not response_text:
                raise ValueError("Empty response from LLM")

            try:
                # LLMs sometimes wrap JSON in markdown code blocks
                if "```json" in response_text:
                    json_start = response_text.index("```json") + 7
                    json_end = response_text.index("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.index("```") + 3
                    json_end = response_text.index("```", json_start)
                    response_text = response_text[json_start:json_end].strip()

                parsed = json.loads(response_text)

                if "content" not in parsed:
                    raise ValueError("Missing 'content' key in response")

                return parsed["content"]

            except (json.JSONDecodeError, ValueError) as json_err:
                # If JSON parsing fails, return raw response
                print(
                    f"Warning: JSON parsing failed (attempt {attempt}): {json_err}. "
                    f"Using raw response.",
                    file=sys.stderr,
                )
                return response_text

        except APIError as exc:
            last_error = exc
            print(
                f"API error (attempt {attempt}/{max_retries}): {exc}",
                file=sys.stderr,
            )
        except Exception as exc:
            last_error = exc
            print(
                f"Unexpected error (attempt {attempt}/{max_retries}): {exc}",
                file=sys.stderr,
            )

        if attempt < max_retries:
            time.sleep(sleep_seconds)

    print(
        f"Error: failed to generate {response_type} response after {max_retries} attempts. "
        f"Last error: {last_error}",
        file=sys.stderr,
    )
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate both helpful and unhelpful tutor responses for tagged practice items "
            "and output a CSV with the responses."
        ),
    )
    parser.add_argument(
        "--input",
        default="data/florida-doe/tagged-practice-items-with-responses.csv",
        help="Path to the input CSV with problem, skill_id, and prerequisites columns.",
    )
    parser.add_argument(
        "--output",
        default="data/florida-doe/tagged-practice-items-with-responses.csv",
        help="Path to the output CSV with generated responses.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model name to use (e.g., gpt-4o-mini, gpt-4o, gpt-4-turbo).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of rows to process (for testing).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Seconds to sleep between retries.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file (skip already processed rows).",
    )

    args = parser.parse_args()

    dotenv.load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key)

    processed_problems = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, newline="", encoding="utf-8") as existing_file:
            existing_reader = csv.DictReader(existing_file)
            for row in existing_reader:
                processed_problems.add(_clean_text(row.get("problem", "")))
        print(
            f"Resuming: skipping {len(processed_problems)} already processed problems.",
            file=sys.stderr,
        )

    output_mode = "a" if args.resume and os.path.exists(args.output) else "w"
    write_header = not (args.resume and os.path.exists(args.output))

    with (
        open(args.input, newline="", encoding="utf-8") as infile,
        open(
            args.output,
            output_mode,
            newline="",
            encoding="utf-8",
        ) as outfile,
    ):
        reader = csv.DictReader(infile)
        fieldnames = [
            "problem",
            "skill_id",
            "prerequisites",
            "helpful_response",
            "unhelpful_response",
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)

        if write_header:
            writer.writeheader()

        rows_processed = 0
        rows_skipped = 0

        for idx, row in enumerate(reader, start=1):
            if args.max_rows is not None and rows_processed >= args.max_rows:
                break

            problem = _clean_text(row.get("problem", ""))
            skill_id = _clean_text(row.get("skill_id", ""))
            prerequisites = _clean_text(row.get("prerequisites", ""))

            if not problem:
                print(
                    f"Warning: empty problem at row {idx}, skipping.",
                    file=sys.stderr,
                )
                continue

            if args.resume and problem in processed_problems:
                rows_skipped += 1
                continue

            print(f"Processing row {idx}: {skill_id}...", file=sys.stderr)

            print("  Generating helpful response...", file=sys.stderr)
            helpful_response = _generate_response(
                client=client,
                model=args.model,
                problem=problem,
                skill_id=skill_id,
                prerequisites=prerequisites,
                response_type="helpful",
                sleep_seconds=args.sleep_seconds,
            )

            print("  Generating unhelpful response...", file=sys.stderr)
            unhelpful_response = _generate_response(
                client=client,
                model=args.model,
                problem=problem,
                skill_id=skill_id,
                prerequisites=prerequisites,
                response_type="unhelpful",
                sleep_seconds=args.sleep_seconds,
            )

            writer.writerow(
                {
                    "problem": problem,
                    "skill_id": skill_id,
                    "prerequisites": prerequisites,
                    "helpful_response": helpful_response,
                    "unhelpful_response": unhelpful_response,
                },
            )

            rows_processed += 1
            outfile.flush()  # persist after each row in case of interruption

        print(f"\nCompleted! Processed {rows_processed} rows.", file=sys.stderr)
        if rows_skipped > 0:
            print(f"Skipped {rows_skipped} already-processed rows.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
