"""Multi-conversation practice benchmark for FlexLearner implementations.

For each target skill the learner climbs from root skills through the prerequisite
graph, then validates mastery with consolidation runs.

Key metrics:
- ``avg_turns_per_skill``: total turns divided by number of skills learned.
- ``consolidation_solution_rate``: fraction of post-mastery conversations
  where the learner found a correct solution.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from evalconvolearn.benchmarks.flexlearners.flexlearner_benchmark import (
    FlexLearnerBenchmark,
)
from evalconvolearn.core.flexlearner import FlexLearner
from evalconvolearn.models.binary_skills_flexlearner import (
    BinarySkillsFlexLearner,
    StudentPool,
)
from evalconvolearn.models.evaluation import LearnerEvalConfig
from evalconvolearn.models.flexlearner_conversation import ConversationGraph
from evalconvolearn.models.practice_item import PracticeItem, PracticeItemPool
from evalconvolearn.models.skill import Skill, SkillSpace
from evalconvolearn.models.tutor import Tutor
from evalconvolearn.utils import (
    get_placement_test_skill_levels,
    run_conversation_to_completion,
)

DEFAULT_TARGET_SKILLS: dict[str, list[str]] = {
    "beginner": ["MA.6.NSO.1.2"],  # 1 prerequisite (MA.6.NSO.1.1)
    "intermediate": ["MA.6.NSO.2.3"],  # chain: 2.1 -> 2.2 -> 2.3
    "expert": ["MA.6.NSO.4.1"],  # 4 prerequisites (entire 1.x chain)
}

DEFAULT_CONSOLIDATION_RUNS = 5
DEFAULT_MAX_CONVERSATION_TURNS = 7
DEFAULT_MAX_CLIMB_ITEMS_PER_SKILL = 3  # max items attempted per skill when climbing


class MultiConversationsPracticeBenchmark(FlexLearnerBenchmark):
    """Benchmark that runs progressive multi-conversation practice sequences.

    For each target skill the learner climbs from root skills through the
    prerequisite graph, then validates mastery with consolidation runs.

    Target skills can be provided explicitly via ``target_skills`` or derived
    from ``skill_levels`` (a ``dict[str, list[str]]`` mapping tier names to
    lists of skill IDs).  When ``skill_levels`` is given it takes precedence
    over ``target_skills`` and is resolved through
    ``FlexLearnerBenchmark._resolve_skill_levels`` so that ``learner_config``
    overrides (``mastered_skills`` / ``learner_level``) are honoured.
    """

    def __init__(
        self,
        skill_space: SkillSpace,
        practice_item_pool: PracticeItemPool,
        oversampled_item_pool: PracticeItemPool | None = None,
        target_skills: dict[str, list[str]] | None = None,
        skill_levels: Mapping[str, Iterable[str]] | None = None,
        consolidation_runs: int = DEFAULT_CONSOLIDATION_RUNS,
        max_conversation_turns: int = DEFAULT_MAX_CONVERSATION_TURNS,
        max_climb_items_per_skill: int = DEFAULT_MAX_CLIMB_ITEMS_PER_SKILL,
        output_dir: Path | None = None,
        learner_pool: StudentPool | None = None,
        learner_config: LearnerEvalConfig | None = None,
        benchmark_extra_args: dict | None = None,
    ) -> None:
        super().__init__(
            skill_space=skill_space,
            practice_item_pool=practice_item_pool,
            output_dir=output_dir,
            learner_pool=learner_pool,
            learner_config=learner_config,
            benchmark_extra_args=benchmark_extra_args,
        )
        self.logger = logging.getLogger(__name__)

        # Oversampled item pool for consolidation phase (more items per skill)
        self.oversampled_item_pool = oversampled_item_pool or practice_item_pool

        # Target skills per difficulty tier — skill_levels takes precedence
        # Resolve and validate skill levels
        resolved = self._resolve_skill_levels(
            skill_levels or get_placement_test_skill_levels(),
        )
        self._validate_skill_levels(resolved)
        target_skills = {tier: list(skill_ids) for tier, skill_ids in resolved.items()}
        self.target_skills = target_skills or DEFAULT_TARGET_SKILLS

        self.logger.info(
            f"[Multi Conversations Practice INIT] Selected target skills {self.target_skills} ",
        )

        # Configuration
        self.consolidation_runs = self.benchmark_extra_args.get(
            "consolidation_runs",
            consolidation_runs,
        )
        self.max_conversation_turns = self.benchmark_extra_args.get(
            "max_conversation_turns",
            max_conversation_turns,
        )
        self.max_climb_items_per_skill = self.benchmark_extra_args.get(
            "max_climb_items_per_skill",
            max_climb_items_per_skill,
        )

        # Initialize a helpful tutor for all conversations
        self.tutor = Tutor(
            id=str(uuid.uuid4()),
            tutor_type="llm",
            tutor_characteristics={"helpfulness": True},
            practice_item_pool=self.practice_item_pool,
            response_interaction_mode="return_only",
        )
        self.tutor.initialize_strategy()

        self.test_run_id = f"multi_conv_practice_{str(uuid.uuid4())[:8]}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_bfs_skill_order(self, target_skill_id: str) -> list[Skill]:
        """Return a BFS-ordered list of skills from root skills to *target_skill_id*.

        Delegates to :meth:`SkillSpace.get_bfs_skill_order`.
        """
        return self.skill_space.get_bfs_skill_order(target_skill_id)

    def _get_items_for_skill(
        self,
        skill_id: str,
        pool: PracticeItemPool,
        exclude_texts: set[str] | None = None,
    ) -> list[PracticeItem]:
        """Return practice items for *skill_id* from *pool*, excluding already-used texts."""
        items = pool.get_items_having_skill(skill_id)
        if exclude_texts:
            items = [it for it in items if it.text not in exclude_texts]
        return items

    def _create_learner(
        self,
        learner_id: str,
        mastered_skills: list[str],
        practice_file: Path,
    ) -> FlexLearner:
        """Create a fresh learner instance.

        When a ``learner_pool`` / ``learner_config`` is set the learner is
        created via ``StudentPool.create_learner`` which also calls
        ``initialize_learner_knowledge`` with the configured kwargs.

        If the learner exposes ``initialize_learner_knowledge`` (i.e. it is
        *not* a plain :class:`BinarySkillsFlexLearner`) the caller is expected to have
        provided proper ``init_knowledge_kwargs`` in the
        :class:`LearnerEvalConfig`

        When no ``learner_pool`` is configured, a plain :class:`BinarySkillsFlexLearner` is
        instantiated
        """
        if self.learner_pool is not None and self.learner_config is not None:
            init_kwargs = dict(self.learner_config.init_knowledge_kwargs or {})
            # Forward the mastered skills so that initialize_learner_knowledge
            # implementations can seed knowledge for each root skill.
            mastered_skill_objects = [
                self.skill_space.get_skill(sid)
                for sid in mastered_skills
                if self.skill_space.get_skill(sid) is not None
            ]
            init_kwargs.setdefault("learner_mastered_skills", mastered_skill_objects)
            return self.learner_pool.create_learner(
                learner_id=learner_id,
                mastered_skills=mastered_skills,
                skill_space=self.skill_space,
                **init_kwargs,
            )
        # Fallback: plain BinarySkillsFlexLearner (no custom KG / knowledge backend)
        practice_file.touch()
        return BinarySkillsFlexLearner(
            id=learner_id,
            mastered_skills=mastered_skills,
            skill_space=self.skill_space,
            practice_conversations_file=practice_file,
        )

    def _run_single_conversation(
        self,
        practice_item: PracticeItem,
        learner: FlexLearner,
        session_id: str,
        db_folder: Path,
        learning_enabled: bool = True,
    ) -> dict:
        """Run a single conversation and return metrics dict."""
        conversation = ConversationGraph(
            id=str(uuid.uuid4()),
            practice_item=practice_item,
            skill_space=self.skill_space,
            learning_enabled=learning_enabled,
            learner=learner,
            max_turns=self.max_conversation_turns,
            graph_memory_db_path=db_folder / f"{session_id}.db",
        )

        conv_metrics = run_conversation_to_completion(
            conversation=conversation,
            practice_item=practice_item,
            session_id=session_id,
            tutor=self.tutor,
            max_turns=self.max_conversation_turns,
        )
        return conv_metrics

    @staticmethod
    def compute_structured_metrics(output_file: Path) -> dict:
        """Extract avg_turns_per_skill and consolidation_solution_rate from summary JSON."""
        with open(output_file, encoding="utf-8") as f:
            summary = json.load(f)

        agg = summary.get("aggregate_metrics", {})
        return {
            "metric_type": "multi_conv_practice",
            "overall_avg_turns_per_skill": agg.get(
                "overall_avg_turns_per_skill",
                float("inf"),
            ),
            "overall_consolidation_solution_rate": agg.get(
                "overall_consolidation_solution_rate",
                0.0,
            ),
            "targets_mastered": agg.get("targets_mastered", 0),
            "total_targets": agg.get("total_targets", 0),
            "total_skills_learned": agg.get("total_skills_learned", 0),
            "breakdown_keys": [],
        }

    # ------------------------------------------------------------------
    # Core evaluation for a single target skill
    # ------------------------------------------------------------------

    def run_evaluation_for_target_skill(
        self,
        tier: str,
        target_skill_id: str,
        tmp_dir: Path,
    ) -> dict:
        """Run the full climb + consolidation sequence for one target skill.

        Returns a result dict with climb and consolidation details.
        """
        self.logger.debug(
            f"\n{'=' * 80}\n  Target skill: {target_skill_id}  (tier: {tier})\n{'=' * 80}",
        )

        # 1. Compute BFS skill order from roots to target
        skill_order = self._get_bfs_skill_order(target_skill_id)
        skill_order_ids = [sk.id for sk in skill_order]
        self.logger.debug(
            f"Skill climb order ({len(skill_order)} skills): {skill_order_ids}",
        )

        # 2. Initialize learner with root skills that are part of the target skill's subgraph
        root_skill_ids = [sk.id for sk in self.skill_space.get_root_skills_for_target(target_skill_id)]
        learner_id = f"multi_conv_{tier}_{target_skill_id}_{self.test_run_id}"
        practice_file = tmp_dir / f"{learner_id}.jsonl"

        learner = self._create_learner(learner_id, root_skill_ids.copy(), practice_file)
        learner.learn_root_skills()

        self.logger.debug(
            f"Learner initialized with root skills: {learner.mastered_skills}",
        )

        # 3. Climb phase: work through skills in order
        used_item_texts: set[str] = set()
        climb_records: list[dict] = []
        total_climb_turns = 0
        total_skills_learned_in_climb = 0
        target_mastered = False

        for skill in skill_order:
            if skill.id in learner.mastered_skills:
                self.logger.debug(f"Skill {skill.id} already mastered, skipping.")
                continue

            # Get practice items for this skill from the main pool
            items = self._get_items_for_skill(
                skill.id,
                self.practice_item_pool,
                exclude_texts=used_item_texts,
            )
            # Fall back to oversampled pool if main pool has no items
            if not items:
                items = self._get_items_for_skill(
                    skill.id,
                    self.oversampled_item_pool,
                    exclude_texts=used_item_texts,
                )
            if not items:
                self.logger.warning(
                    f"No practice items found for skill {skill.id}, skipping.",
                )
                continue

            # Attempt up to max_climb_items_per_skill conversations for this skill
            skill_mastered_this_round = False
            for attempt_idx, item in enumerate(items[: self.max_climb_items_per_skill]):
                used_item_texts.add(item.text)

                current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_id = f"{uuid.uuid4()}_{current_time_str}"
                db_folder = self.output_dir / self.test_run_id / tier / target_skill_id / "climb" / current_time_str
                db_folder.mkdir(parents=True, exist_ok=True)

                mastered_before = set(learner.mastered_skills)

                self.logger.debug(
                    f"[Climb] skill={skill.id}, attempt={attempt_idx + 1}, item={item.text[:60]}...",
                )

                conv_metrics = self._run_single_conversation(
                    practice_item=item,
                    learner=learner,
                    session_id=session_id,
                    db_folder=db_folder,
                    learning_enabled=True,
                )

                mastered_after = set(learner.mastered_skills)
                newly_learned = mastered_after - mastered_before
                turns = conv_metrics.get("turns_to_solution", 0)
                solution_found = conv_metrics.get("solution_found", False)
                total_climb_turns += turns
                total_skills_learned_in_climb += len(newly_learned)

                climb_record = {
                    "phase": "climb",
                    "tier": tier,
                    "target_skill": target_skill_id,
                    "current_skill": skill.id,
                    "attempt": attempt_idx + 1,
                    "item_text": item.text[:200],
                    "item_skills": item.associated_skills,
                    "turns": turns,
                    "solution_found": solution_found,
                    "newly_learned_skills": list(newly_learned),
                    "mastered_skills_after": list(mastered_after),
                    "max_turns_reached": conv_metrics.get("max_turns_reached", False),
                    "tokens_used": conv_metrics.get(
                        "tokens_used",
                        {"input_tokens": 0, "output_tokens": 0},
                    ),
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                }
                climb_records.append(climb_record)

                self.logger.info(
                    f"  → turns={turns}, solution={solution_found}, "
                    f"learned={list(newly_learned)}, "
                    f"mastered_count={len(mastered_after)}",
                )

                if skill.id in learner.mastered_skills:
                    skill_mastered_this_round = True
                    break  # move to next skill in the path

            # Check if target is mastered
            if target_skill_id in learner.mastered_skills:
                target_mastered = True
                self.logger.info(
                    f"Target skill {target_skill_id} mastered after climbing.",
                )
                break

            if not skill_mastered_this_round:
                self.logger.info(
                    f"Skill {skill.id} not mastered after {self.max_climb_items_per_skill} attempts, "
                    f"continuing to next skill in path.",
                )

        # 4. Consolidation phase (only if target was mastered)
        consolidation_records: list[dict] = []
        consolidation_solutions_found = 0

        if target_mastered:
            self.logger.info(
                f"\n--- Consolidation phase: {self.consolidation_runs} conversations on {target_skill_id} ---",
            )

            # Use oversampled pool for consolidation items
            consol_items = self._get_items_for_skill(
                target_skill_id,
                self.oversampled_item_pool,
                exclude_texts=used_item_texts,
            )

            if len(consol_items) < self.consolidation_runs:
                self.logger.warning(
                    f"Only {len(consol_items)} unused items available for "
                    f"{self.consolidation_runs} consolidation runs on {target_skill_id}.",
                )

            for consol_idx in range(min(self.consolidation_runs, len(consol_items))):
                item = consol_items[consol_idx]
                used_item_texts.add(item.text)

                current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_id = f"{uuid.uuid4()}_{current_time_str}"
                db_folder = (
                    self.output_dir / self.test_run_id / tier / target_skill_id / "consolidation" / current_time_str
                )
                db_folder.mkdir(parents=True, exist_ok=True)

                self.logger.debug(
                    f"[Consolidation] run={consol_idx + 1}/{self.consolidation_runs}, item={item.text[:60]}...",
                )

                conv_metrics = self._run_single_conversation(
                    practice_item=item,
                    learner=learner,
                    session_id=session_id,
                    db_folder=db_folder,
                    learning_enabled=True,
                )

                turns = conv_metrics.get("turns_to_solution", 0)
                solution_found = conv_metrics.get("solution_found", False)
                if solution_found:
                    consolidation_solutions_found += 1

                consol_record = {
                    "phase": "consolidation",
                    "tier": tier,
                    "target_skill": target_skill_id,
                    "consolidation_run": consol_idx + 1,
                    "item_text": item.text[:200],
                    "item_skills": item.associated_skills,
                    "turns": turns,
                    "solution_found": solution_found,
                    "max_turns_reached": conv_metrics.get("max_turns_reached", False),
                    "tokens_used": conv_metrics.get(
                        "tokens_used",
                        {"input_tokens": 0, "output_tokens": 0},
                    ),
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                }
                consolidation_records.append(consol_record)

                self.logger.info(
                    f"  → turns={turns}, solution={solution_found}",
                )
        else:
            self.logger.info(
                f"Target skill {target_skill_id} was NOT mastered. Skipping consolidation phase.",
            )

        # 5. Compute summary metrics
        total_consolidation_runs = len(consolidation_records)
        consolidation_solution_rate = (
            consolidation_solutions_found / total_consolidation_runs if total_consolidation_runs > 0 else 0.0
        )
        avg_turns_per_skill = (
            total_climb_turns / total_skills_learned_in_climb if total_skills_learned_in_climb > 0 else float("inf")
        )

        result = {
            "tier": tier,
            "target_skill": target_skill_id,
            "skill_climb_order": skill_order_ids,
            "target_mastered": target_mastered,
            "total_climb_conversations": len(climb_records),
            "total_climb_turns": total_climb_turns,
            "total_skills_learned_in_climb": total_skills_learned_in_climb,
            "avg_turns_per_skill": avg_turns_per_skill,
            "consolidation_runs_completed": total_consolidation_runs,
            "consolidation_solutions_found": consolidation_solutions_found,
            "consolidation_solution_rate": consolidation_solution_rate,
            "final_mastered_skills": list(learner.mastered_skills),
            "climb_records": climb_records,
            "consolidation_records": consolidation_records,
            "timestamp": datetime.now().isoformat(),
        }

        self.logger.info(
            f"\n--- Summary for {target_skill_id} ({tier}) ---\n"
            f"  Target mastered         : {target_mastered}\n"
            f"  Climb conversations     : {len(climb_records)}\n"
            f"  Total climb turns       : {total_climb_turns}\n"
            f"  Skills learned (climb)  : {total_skills_learned_in_climb}\n"
            f"  Avg turns/skill         : {avg_turns_per_skill:.2f}\n"
            f"  Consolidation runs      : {total_consolidation_runs}\n"
            f"  Consolidation solutions : {consolidation_solutions_found}/{total_consolidation_runs}\n"
            f"  Consolidation sol. rate : {consolidation_solution_rate:.2%}\n",
        )

        return result

    # ------------------------------------------------------------------
    # Run all evaluations
    # ------------------------------------------------------------------

    def run_all_evaluations(self) -> Path:
        """Run multi-conversation practice evaluations for all target skills.

        Returns
        -------
            Path to the JSON summary file.

        """
        output_file = self.output_dir / f"multi_conv_practice_{self.test_run_id}.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        total_targets = sum(len(v) for v in self.target_skills.values())
        print(
            f"[MultiConversationsPracticeBenchmark] Starting\n"
            f"  Tiers            : {list(self.target_skills.keys())}\n"
            f"  Total targets    : {total_targets}\n"
            f"  Consolidation N  : {self.consolidation_runs}\n"
            f"  Max turns/conv   : {self.max_conversation_turns}\n"
            f"  Max items/skill  : {self.max_climb_items_per_skill}\n"
            f"  Run ID           : {self.test_run_id}",
        )

        all_results: list[dict] = []

        with TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)

            completed = 0
            for tier, skill_ids in self.target_skills.items():
                for target_skill_id in skill_ids:
                    completed += 1
                    print(
                        f"[MultiConversationsPracticeBenchmark] "
                        f"({completed}/{total_targets}) tier={tier}, "
                        f"target={target_skill_id}",
                    )

                    result = self.run_evaluation_for_target_skill(
                        tier=tier,
                        target_skill_id=target_skill_id,
                        tmp_dir=tmp_dir,
                    )
                    all_results.append(result)

        # Compute aggregate metrics
        total_skills_learned = sum(r["total_skills_learned_in_climb"] for r in all_results)
        total_turns = sum(r["total_climb_turns"] for r in all_results)
        overall_avg_turns_per_skill = total_turns / total_skills_learned if total_skills_learned > 0 else float("inf")

        total_consol_runs = sum(r["consolidation_runs_completed"] for r in all_results)
        total_consol_solutions = sum(r["consolidation_solutions_found"] for r in all_results)
        overall_consolidation_rate = total_consol_solutions / total_consol_runs if total_consol_runs > 0 else 0.0

        summary = {
            "run_id": self.test_run_id,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "target_skills": self.target_skills,
                "consolidation_runs": self.consolidation_runs,
                "max_conversation_turns": self.max_conversation_turns,
                "max_climb_items_per_skill": self.max_climb_items_per_skill,
            },
            "aggregate_metrics": {
                "overall_avg_turns_per_skill": overall_avg_turns_per_skill,
                "overall_consolidation_solution_rate": overall_consolidation_rate,
                "total_targets": total_targets,
                "targets_mastered": sum(1 for r in all_results if r["target_mastered"]),
                "total_climb_conversations": sum(r["total_climb_conversations"] for r in all_results),
                "total_climb_turns": total_turns,
                "total_skills_learned": total_skills_learned,
                "total_consolidation_runs": total_consol_runs,
                "total_consolidation_solutions": total_consol_solutions,
            },
            "per_target_results": all_results,
        }

        # Write JSON summary
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        # Print final summary
        print(
            f"\n{'=' * 64}\n"
            f"  Multi-Conversation Practice Benchmark — Final Summary\n"
            f"{'=' * 64}\n"
            f"  Run ID                          : {self.test_run_id}\n"
            f"  Targets evaluated               : {total_targets}\n"
            f"  Targets mastered                : {summary['aggregate_metrics']['targets_mastered']}\n"
            f"  Overall avg turns/skill         : {overall_avg_turns_per_skill:.2f}\n"
            f"  Overall consolidation sol. rate : {overall_consolidation_rate:.2%}\n"
            f"  Total climb conversations       : {summary['aggregate_metrics']['total_climb_conversations']}\n"
            f"  Total consolidation runs        : {total_consol_runs}\n"
            f"  Results saved to                : {output_file}\n"
            f"{'=' * 64}\n",
        )

        self.logger.info("Results saved to: %s", output_file)
        return output_file
