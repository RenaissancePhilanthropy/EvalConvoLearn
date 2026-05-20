"""Result tracking classes and result-printing helpers for benchmark evaluations."""

import json
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Console pretty-printers (shared across all eval scripts)
# ---------------------------------------------------------------------------


def print_placement_test_results(run_data: dict) -> None:
    """Pretty-print a single PlacementTestBenchmark run record from a JSONL file.

    Expected fields
    ---------------
    learner_level, run_id, knowledge_check_mode, mastered_skills,
    summary  (dict with correct / total_items / accuracy),
    alignment_accuracy, items  (list of per-item dicts).

    Per-item fields: skill, expected_correct, is_correct,
    learner_has_all_required_skills, is_aligned_to_matrix, learner_answer.
    """
    learner_level = run_data.get("learner_level", "?")
    run_id = run_data.get("run_id", "?")
    knowledge_mode = run_data.get("knowledge_check_mode", "unknown")
    mastered = run_data.get("mastered_skills", [])
    summary = run_data.get("summary", {})
    alignment_acc = run_data.get("alignment_accuracy")
    items = run_data.get("items", [])

    print(f"\n  Run {run_id}  |  level={learner_level}  |  mode={knowledge_mode}")
    print(f"  Mastered skills : {mastered}")
    print(
        f"  Score           : {summary.get('correct', '?')} / "
        f"{summary.get('total_items', '?')} "
        f"({summary.get('accuracy', 0.0):.1%})",
    )
    if alignment_acc is not None:
        print(f"  Alignment acc.  : {alignment_acc:.1%}")

    if not items:
        return

    print()
    print(
        f"  {'Skill':<20}  {'Expected':>10}  {'Got':>6}  {'CanLearn':>10}  {'Aligned':>8}  Answer",
    )
    print(f"  {'-'*20}  {'-'*10}  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*30}")
    for item in items:
        skill = (item.get("skill") or "")[:20]
        expected_correct = item.get("expected_correct")  # True / False / None
        if expected_correct is True:
            expected = "correct"
        elif expected_correct is False:
            expected = "wrong"
        else:
            expected = "n/a"
        got = "✓" if item.get("is_correct") else "✗"
        can_learn = item.get("learner_has_all_required_skills")  # True / False / None
        if can_learn is True:
            can_learn_str = "✓"
        elif can_learn is False:
            can_learn_str = "✗"
        else:
            can_learn_str = "n/a"
        is_aligned = item.get("is_aligned_to_matrix")  # True / False / None
        if is_aligned is True:
            aligned = "✓"
        elif is_aligned is False:
            aligned = "✗"
        else:
            aligned = "n/a"
        learner_answer = (item.get("learner_answer") or "—")[:40]
        print(
            f"  {skill:<20}  {expected:>10}  {got:>6}  {can_learn_str:>10}  {aligned:>8}  {learner_answer}",
        )


def print_lfc_results(output_file: Path, label: str) -> None:
    """Pretty-print all LearningFromConversationBenchmark records from a JSONL file.

    Records are grouped by ``check_mode`` and printed with a per-mode summary
    followed by a per-item detail table.

    Expected per-record fields
    --------------------------
    check_mode, check_if_should_learn, learner_level, response_type,
    item_index, problem, item_skills, mastered_skills, learnable_skills,
    expected_skills, learned_skills, alignment_accuracy,
    evaluation_mode  (optional),
    pre_test_can_answer / post_test_can_answer  (pre/post mode only).
    """
    if not output_file.exists():
        print(f"\n  [{label}] LFC output file not found: {output_file}")
        return

    records: list[dict] = []
    with output_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print(f"\n  [{label}] LFC output file is empty.")
        return

    # Collect distinct modes in the order they appear
    seen_modes: list[str] = []
    for rec in records:
        m = rec.get("check_mode", "with_skill_check")
        if m not in seen_modes:
            seen_modes.append(m)

    for mode in seen_modes:
        mode_records = [
            r for r in records if r.get("check_mode", "with_skill_check") == mode
        ]
        _print_lfc_mode_summary(mode, mode_records)


def _print_lfc_mode_summary(mode: str, records: list[dict]) -> None:
    """Print summary + per-item table for one check_mode group of LFC records."""
    if not records:
        print(f"\n  [check_mode={mode!r}] No records.")
        return

    total = len(records)
    aligned = sum(1 for r in records if r.get("alignment_accuracy", 0) == 1.0)
    avg_acc = aligned / total if total else 0.0

    # Break down by response_type
    helpful = [r for r in records if r.get("response_type") == "helpful"]
    unhelpful = [r for r in records if r.get("response_type") == "unhelpful"]
    h_aligned = sum(1 for r in helpful if r.get("alignment_accuracy", 0) == 1.0)
    u_aligned = sum(1 for r in unhelpful if r.get("alignment_accuracy", 0) == 1.0)

    check_flag = records[0].get("check_if_should_learn", True)
    eval_mode = records[0].get("evaluation_mode", "skill_alignment")
    print(
        f"\n  ── check_mode={mode!r}  (check_if_should_learn={check_flag},  eval={eval_mode}) ──",
    )
    print(f"     Total items  : {total}")
    print(f"     Aligned      : {aligned}/{total}  ({avg_acc:.1%})")
    if helpful:
        print(f"     Helpful resp : {h_aligned}/{len(helpful)} aligned")
    if unhelpful:
        print(f"     Unhelpful    : {u_aligned}/{len(unhelpful)} aligned")

    # Per-item detail — base columns
    use_pre_post = eval_mode == "pre_post_test"

    print()
    if use_pre_post:
        print(
            f"  {'#':>3}  {'level':<12}  {'resp':<10}  "
            f"{'expected':<28}  {'learned':<28}  "
            f"{'pre':>5}  {'post':>5}  {'ok':>4}",
        )
        print(
            f"  {'-'*3}  {'-'*12}  {'-'*10}  "
            f"{'-'*28}  {'-'*28}  "
            f"{'-'*5}  {'-'*5}  {'-'*4}",
        )
    else:
        print(
            f"  {'#':>3}  {'level':<12}  {'resp':<10}  "
            f"{'expected':<30}  {'learned':<30}  {'ok':>4}",
        )
        print(
            f"  {'-'*3}  {'-'*12}  {'-'*10}  {'-'*30}  {'-'*30}  {'-'*4}",
        )

    for rec in records:
        idx = rec.get("item_index", "?")
        level = (rec.get("learner_level") or "")[:12]
        rtype = (rec.get("response_type") or "")[:10]
        expected = ", ".join(rec.get("expected_skills") or []) or "—"
        learned = ", ".join(rec.get("learned_skills") or []) or "—"
        ok = "✓" if rec.get("alignment_accuracy", 0) == 1.0 else "✗"
        if use_pre_post:
            pre = "T" if rec.get("pre_test_can_answer") else "F"
            post = "T" if rec.get("post_test_can_answer") else "F"
            print(
                f"  {idx:>3}  {level:<12}  {rtype:<10}  "
                f"{expected[:28]:<28}  {learned[:28]:<28}  "
                f"{pre:>5}  {post:>5}  {ok:>4}",
            )
        else:
            print(
                f"  {idx:>3}  {level:<12}  {rtype:<10}  "
                f"{expected[:30]:<30}  {learned[:30]:<30}  {ok:>4}",
            )


def print_placement_results(output_file: Path, label: str) -> None:
    """Compact pretty-print of PlacementTestBenchmark records from a JSONL file.

    Prints one summary line per run: run_id, learner level, and score.
    For per-item detail use :func:`print_placement_test_results` instead.
    """
    if not output_file.exists():
        print(f"\n  [{label}] Placement test output file not found: {output_file}")
        return

    print(f"\n  Detailed results — {label}:")
    with output_file.open(encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            run_data = json.loads(stripped)
            summary = run_data.get("summary", {})
            print(
                f"    Run {run_data.get('run_id', '?')} | "
                f"level={run_data.get('learner_level', '?')} | "
                f"score={summary.get('correct', '?')}/{summary.get('total_items', '?')} "
                f"({summary.get('accuracy', 0.0):.0%})",
            )


def print_mcp_results(output_file: Path, label: str) -> None:
    """Pretty-print MultiConversationsPracticeBenchmark results from a JSON file.

    Prints aggregate metrics (targets, mastered, turns/skill, consolidation rate)
    and a per-target result row.
    """
    if not output_file.exists():
        print(f"\n  [{label}] MCP output file not found: {output_file}")
        return

    print(f"\n  Multi-Conversations Practice — {label}:")
    with output_file.open(encoding="utf-8") as fh:
        mcp_data = json.load(fh)

    agg = mcp_data.get("aggregate_metrics", {})
    print(f"    Targets evaluated  : {agg.get('total_targets', '?')}")
    print(f"    Targets mastered   : {agg.get('targets_mastered', '?')}")
    print(f"    Avg turns/skill    : {agg.get('overall_avg_turns_per_skill', 0):.2f}")
    print(
        f"    Consolidation rate : {agg.get('overall_consolidation_solution_rate', 0):.0%}",
    )
    for tr in mcp_data.get("per_target_results", []):
        icon = "✓" if tr.get("target_mastered") else "✗"
        print(
            f"    [{icon}] {tr['target_skill']} ({tr['tier']}) — "
            f"climb={tr['total_climb_conversations']} conversations, "
            f"skills learned={tr['total_skills_learned_in_climb']}, "
            f"consol={tr.get('consolidation_solution_rate', 0):.0%}",
        )


class PlacementTestResult:
    """Result tracking for placement test evaluations."""

    def __init__(self, test_name: str, test_id: str):
        self.test_name = test_name
        self.test_id = test_id
        self.timestamp = datetime.now().isoformat()
        self.results: list[dict] = []

    def add_result(
        self,
        learner_level: str,
        run_id: int,
        mastered_skills: list[str],
        summary: dict,
        alignment_accuracy: float,
        alignment_evaluated: int,
        items: list[dict],
    ):
        """Add a placement test evaluation result."""
        result = {
            "learner_level": learner_level,
            "run_id": run_id,
            "mastered_skills": mastered_skills,
            "summary": summary,
            "alignment_accuracy": alignment_accuracy,
            "alignment_evaluated": alignment_evaluated,
            "items": items,
            "timestamp": datetime.now().isoformat(),
        }
        self.results.append(result)

    def save_to_file(self, output_dir: Path, runs_per_level: int):
        """Save results to JSONL file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = (
            output_dir / f"placement_test_skill_alignment_{runs_per_level}runs.jsonl"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            for result in self.results:
                run_record = {
                    "test_name": self.test_name,
                    **result,
                }
                f.write(json.dumps(run_record) + "\n")

        return output_file
