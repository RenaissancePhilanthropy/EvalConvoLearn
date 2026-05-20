"""Add conversation-metric tags to saved Eedi tutoring conversations.

This script reads the JSONL output produced by
`extract_tag_store_eedi_tutoring_conversations.py`, computes the same
conversation-level metrics used by
`DatasetFittedConversationalBenchmark`, and writes an updated JSONL file.

The added metrics include:
- questions per interrogative learner turn
- average learner turn length in words (benchmark-compatible)
- average learner turn string length in characters
- learner error types
- learner talk moves

Example:

```bash
uv run python data/evaluations/source_data/eedi_tutoring/tag_eedi_sampled_convs_with_metrics.py \
  --input data/evaluations/source_data/eedi_tutoring/all_conversations.jsonl \
  --output data/evaluations/source_data/eedi_tutoring/all_conversations_with_metrics.jsonl
```
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[4]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evalconvolearn.benchmarks.realistic_benchmarks_from_conversation_data.dataset_fitted_conversational_benchmark import (  # noqa: E402
    compute_conversation_metrics,
)

DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_INPUT = Path(__file__).resolve().with_name("all_conversations.jsonl")
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("all_conversations_with_metrics.jsonl")
DEFAULT_CACHE_PATH = Path(__file__).resolve().with_name("conversation_metrics_cache.json")

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
    parser = argparse.ArgumentParser(
        description=(
            "Add benchmark-compatible conversation metrics and behavior labels "
            "to stored Eedi tutoring conversations."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input conversations JSONL (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"Conversation-label cache path (default: {DEFAULT_CACHE_PATH}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model for behavior classification (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help=f"Console log verbosity (default: {DEFAULT_LOG_LEVEL}).",
    )
    return parser.parse_args()


def _load_json_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load cache %s: %s", cache_path, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def _persist_json_cache(cache: dict[str, dict[str, Any]], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)


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
                msg = f"Expected JSON object on line {line_idx} of {input_path}."
                raise TypeError(msg)
            records.append(payload)
    return records


def _save_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp_path.replace(output_path)


def main() -> int:
    """Run the Eedi conversation metric-tagging pipeline."""
    args = _parse_args()
    setup_logging(args.log_level)

    if not args.input.exists():
        logger.error("Input JSONL not found: %s", args.input)
        return 1
    cache = _load_json_cache(args.cache_path)
    records = _read_jsonl(args.input)

    logger.info("Loaded %d records from %s", len(records), args.input)

    tagged_records: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        metrics = compute_conversation_metrics(
            dialogue_history=record.get("capped_dialogue_history", record.get("dialogue_history", [])),
            problem_text=str(record.get("practice_item_text", "")),
            correct_answer=str(record.get("correct_answer", "")),
            cache_namespace="real",
            classification_model=args.model,
            cache=cache,
        )

        tagging_metadata = record.get("tagging_metadata")
        if not isinstance(tagging_metadata, dict):
            tagging_metadata = {}
        tagging_metadata["conversation_metrics_tagging"] = {
            "model": args.model,
            "metrics_version": "dataset_fitted_conversational_benchmark_compatible",
            "metrics_source": "evalconvolearn.benchmarks.realistic_benchmarks_from_conversation_data.dataset_fitted_conversational_benchmark.compute_conversation_metrics",
        }

        tagged_records.append(
            {
                **record,
                "conversation_metrics": metrics,
                "tagging_metadata": tagging_metadata,
            },
        )
        logger.info(
            "[%d/%d] Tagged session %s",
            index,
            len(records),
            record.get("session_id", "<unknown>"),
        )

    _save_jsonl(tagged_records, args.output)
    _persist_json_cache(cache, args.cache_path)

    logger.info("Wrote %d tagged records to %s", len(tagged_records), args.output)
    logger.info("Loaded conversations: %d", len(records))
    logger.info("Tagged conversations: %d", len(tagged_records))
    logger.info("Output written to: %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
