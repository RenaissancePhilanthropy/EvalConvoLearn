"""Evaluation result dataclasses and metric-merge helpers."""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .evaluation import BenchmarkName

# ---------------------------------------------------------------------------
# Structured-metrics merge helpers
# ---------------------------------------------------------------------------


def _merge_alignment_metrics(metrics_list: list[dict]) -> dict:
    overall_vals = [m.get("overall_avg_alignment", 0.0) for m in metrics_list]
    avg_overall = sum(overall_vals) / len(overall_vals) if overall_vals else 0.0

    merged_breakdowns: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list),
    )
    for m in metrics_list:
        for breakdown_name, breakdown_dict in (m.get("breakdowns") or {}).items():
            if isinstance(breakdown_dict, dict):
                for bucket_key, bucket_val in breakdown_dict.items():
                    if isinstance(bucket_val, dict):
                        val = bucket_val.get(
                            "avg_alignment",
                            bucket_val.get("expectation_met_pct", 0.0),
                        )
                    else:
                        val = float(bucket_val)
                    merged_breakdowns[breakdown_name][bucket_key].append(val)

    breakdowns_averaged = {
        name: {k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in buckets.items()}
        for name, buckets in merged_breakdowns.items()
    }

    return {
        "metric_type": "alignment",
        "avg_alignment_rate": avg_overall,
        "breakdowns": breakdowns_averaged,
        "breakdown_keys": metrics_list[0].get("breakdown_keys", []),
    }


def _merge_multi_conv_metrics(metrics_list: list[dict]) -> dict:
    avg_turns_vals = [
        m["overall_avg_turns_per_skill"]
        for m in metrics_list
        if m.get("overall_avg_turns_per_skill") is not None
        and m["overall_avg_turns_per_skill"] != float("inf")
    ]
    consol_vals = [
        m.get("overall_consolidation_solution_rate", 0.0) for m in metrics_list
    ]
    total_targets = sum(m.get("total_targets", 0) for m in metrics_list)
    total_mastered = sum(m.get("targets_mastered", 0) for m in metrics_list)
    return {
        "metric_type": "multi_conv_practice",
        "avg_turns_per_skill": (
            sum(avg_turns_vals) / len(avg_turns_vals)
            if avg_turns_vals
            else float("inf")
        ),
        "avg_consolidation_solution_rate": (
            sum(consol_vals) / len(consol_vals) if consol_vals else 0.0
        ),
        "total_targets": total_targets,
        "total_targets_mastered": total_mastered,
        "avg_alignment_rate": total_mastered / total_targets if total_targets else None,
    }


def _merge_structured_metrics(metrics_list: list[dict]) -> dict:
    if not metrics_list:
        return {}
    metric_type = metrics_list[0].get("metric_type", "alignment")
    if metric_type == "multi_conv_practice":
        return _merge_multi_conv_metrics(metrics_list)
    return _merge_alignment_metrics(metrics_list)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkRunSummary:
    """Per-benchmark, per-learner-config run summary."""

    benchmark_name: BenchmarkName
    learner_config_label: str
    status: str  # "success" | "error"
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "success"


@dataclass
class EvaluationResults:
    """Aggregated result returned by EvalConvoLearn.run_evaluation.

    Attributes
    ----------
    run_id:
        Short unique identifier for the evaluation run.
    run_dir:
        Root directory under which all benchmark artifacts were written.
    label:
        Human-readable label from the :class:`EvaluationConfig`, if any.
    started_at:
        ISO-format timestamp when the run started.
    finished_at:
        ISO-format timestamp when the run finished.
    summaries:
        Flat list of :class:`BenchmarkRunSummary` objects — one per
        (benchmark, learner-config) pair.
    raw:
        The full ``dict`` returned by :class:`EvaluationService`.

    """

    run_id: str
    run_dir: Path
    label: str | None
    started_at: str
    finished_at: str
    summaries: list[BenchmarkRunSummary] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        """``True`` when every benchmark run succeeded."""
        return all(s.passed for s in self.summaries)

    @property
    def failed_summaries(self) -> list[BenchmarkRunSummary]:
        """Summaries whose status is ``'error'``."""
        return [s for s in self.summaries if not s.passed]

    @property
    def output_paths(self) -> list[Path]:
        """Collect every output file / directory path across all runs."""
        paths: list[Path] = []
        for summary in self.summaries:
            for key in ("output_file", "output_dir"):
                value = summary.output.get(key)
                if value:
                    paths.append(Path(value))
        return paths

    def print_summary(self) -> None:
        """Print a compact pass/fail summary for every benchmark run to stdout."""
        print(f"\nAll passed: {self.all_passed}")
        for s in self.summaries:
            icon = "✓" if s.passed else "✗"
            print(
                f"  [{icon}] {s.benchmark_name} / {s.learner_config_label} — {s.status}",
            )
            if s.error:
                print(f"        Error: {s.error}")

    @classmethod
    def _from_raw(cls, raw: dict[str, Any]) -> EvaluationResults:
        summaries: list[BenchmarkRunSummary] = []
        for bname, lc_map in raw.get("benchmarks", {}).items():
            for lc_label, entry in lc_map.items():
                summaries.append(
                    BenchmarkRunSummary(
                        benchmark_name=bname,
                        learner_config_label=lc_label,
                        status=entry.get("status", "error"),
                        output=entry.get("output", {}),
                        error=entry.get("error"),
                    ),
                )
        return cls(
            run_id=raw.get("run_id", ""),
            run_dir=Path(raw.get("run_dir", ".")),
            label=raw.get("label"),
            started_at=raw.get("started_at", ""),
            finished_at=raw.get("finished_at", ""),
            summaries=summaries,
            raw=raw,
        )


@dataclass
class EvalSetResults:
    """Aggregated results from a set of evaluation runs across multiple configs.

    Produced by :meth:`EvalConvoLearn.aggregate_results`.

    Attributes
    ----------
    evalset_id:
        Short unique identifier for this evaluation set.
    evalset_label:
        Human-readable label (e.g. ``"20240101_120000_my_run"``).
    timestamp:
        ISO-format timestamp when the set was aggregated.
    all_results:
        The individual :class:`EvaluationResults` from each config run.
    benchmark_x_learner_summaries:
        Aggregated metrics grouped by benchmark x learner type.
    output_dir:
        Directory where the summary JSON was (or will be) saved.

    """

    evalset_id: str
    evalset_label: str
    timestamp: str
    all_results: list[EvaluationResults] = field(default_factory=list)
    benchmark_x_learner_summaries: list[dict[str, Any]] = field(default_factory=list)
    output_dir: Path | None = None

    @property
    def total_runs(self) -> int:
        """Total number of individual benchmark runs across all evaluation configs."""
        return sum(len(r.summaries) for r in self.all_results)

    @property
    def total_passed(self) -> int:
        """Number of benchmark runs that completed successfully."""
        return sum(sum(1 for s in r.summaries if s.passed) for r in self.all_results)

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        print("=" * 64)
        print(f"  Eval set    : {self.evalset_label}")
        print(f"  Total runs  : {self.total_runs}")
        print(f"  Total passed: {self.total_passed}")
        print("=" * 64)

        for entry in self.benchmark_x_learner_summaries:
            metric_type = entry.get("metric_type", "alignment")
            label = f"{entry['benchmark']} x {entry['learner_type']}"

            if metric_type == "multi_conv_practice":
                turns = entry.get("avg_turns_per_skill", float("inf"))
                consol = entry.get("avg_consolidation_solution_rate", 0.0)
                mastered = entry.get("total_targets_mastered", 0)
                total = entry.get("total_targets", 0)
                print(
                    f"  {label:<45} | pass {entry['n_passed']}/{entry['n_runs']} | "
                    f"mastered {mastered}/{total} | turns/skill {turns:.1f} | "
                    f"consol {consol * 100:.0f}%",
                )
            else:
                align_val = entry.get("avg_alignment_rate")
                align_str = f"{align_val:.1%}" if align_val is not None else "N/A"
                print(
                    f"  {label:<45} | pass {entry['n_passed']}/{entry['n_runs']} | "
                    f"alignment {align_str}",
                )
                for bd_name, bd_vals in (entry.get("breakdowns") or {}).items():
                    if isinstance(bd_vals, dict):
                        for bucket, val in bd_vals.items():
                            if isinstance(val, (int, float)):
                                print(f"    -> {bd_name} = {bucket} : {val * 100:.1f}%")

    def save(self, output_dir: Path | str | None = None) -> Path:
        """Write the eval set summary to ``<output_dir>/evalset_summary.json``.

        Returns the path to the written file.
        """
        save_dir = Path(output_dir) if output_dir else self.output_dir
        if save_dir is None:
            msg = "No output_dir provided. Pass one to save() or set output_dir on the instance."
            raise ValueError(msg)
        save_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "evalset_id": self.evalset_id,
            "evalset_label": self.evalset_label,
            "timestamp": self.timestamp,
            "total_benchmark_runs": self.total_runs,
            "total_passed": self.total_passed,
            "benchmark_x_learner_summaries": self.benchmark_x_learner_summaries,
            "individual_results": [
                {
                    "benchmark": s.benchmark_name,
                    "learner_config": s.learner_config_label,
                    "status": s.status,
                    "passed": s.passed,
                    "output_dir": str(
                        s.output.get("output_dir", s.output.get("output_file", "")),
                    ),
                    "error": s.error,
                }
                for r in self.all_results
                for s in r.summaries
            ],
        }

        summary_path = save_dir / "evalset_summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        return summary_path


def build_evalset_results(
    results: list[EvaluationResults],
    eval_configs=None,
    evalset_label: str | None = None,
    output_dir: Path | str | None = None,
) -> EvalSetResults:
    """Aggregate multiple :class:`EvaluationResults` into an :class:`EvalSetResults`.

    Groups individual benchmark run summaries by benchmark x learner type,
    merges structured metrics across runs, and optionally saves a summary JSON.

    Parameters
    ----------
    results:
        List of :class:`EvaluationResults` to aggregate.
    eval_configs:
        Optional list of :class:`EvaluationConfig` objects used to produce *results*.
        When provided, learner class names are derived from configs directly.
    evalset_label:
        Human-readable name for this evaluation set.
    output_dir:
        If provided, the summary JSON is written here automatically.

    """
    run_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().isoformat()  # noqa: DTZ005
    label = evalset_label or f"evalset_{run_id}"

    label_to_learner_type: dict[str, str] = {}
    if eval_configs:
        for ec in eval_configs:
            for lc in ec.learner_configs:
                label_to_learner_type[lc.label] = lc.learner_class.__name__

    grouped: dict[str, dict[str, list[BenchmarkRunSummary]]] = defaultdict(
        lambda: defaultdict(list),
    )
    for r in results:
        for s in r.summaries:
            learner_type = label_to_learner_type.get(
                s.learner_config_label,
                s.learner_config_label,
            )
            grouped[s.benchmark_name][learner_type].append(s)

    benchmark_summaries: list[dict[str, Any]] = []
    for bench_name, lt_map in grouped.items():
        for lt, summaries in lt_map.items():
            n_total = len(summaries)
            n_passed = sum(1 for s in summaries if s.passed)
            sm_list = [
                s.output["structured_metrics"]
                for s in summaries
                if s.output.get("structured_metrics")
            ]
            merged = _merge_structured_metrics(sm_list) if sm_list else {}
            result_paths = [
                str(s.output.get("output_dir", s.output.get("output_file", "")))
                for s in summaries
                if s.output.get("output_dir") or s.output.get("output_file")
            ]
            entry: dict[str, Any] = {
                "benchmark": bench_name,
                "learner_type": lt,
                "n_runs": n_total,
                "n_passed": n_passed,
                "pass_rate": n_passed / n_total if n_total else 0,
                "result_paths": result_paths,
            }
            entry.update(merged)
            benchmark_summaries.append(entry)

    evalset_dir = Path(output_dir) if output_dir else None
    evalset_results = EvalSetResults(
        evalset_id=run_id,
        evalset_label=label,
        timestamp=timestamp,
        all_results=results,
        benchmark_x_learner_summaries=benchmark_summaries,
        output_dir=evalset_dir,
    )
    if evalset_dir:
        evalset_results.save(evalset_dir)
    return evalset_results
