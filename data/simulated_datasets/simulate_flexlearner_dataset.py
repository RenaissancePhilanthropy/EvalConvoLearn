"""Simulate a dataset of practice conversations using FlexLearner SDK.

Each learner is initialized with a random sample of skills (plus their prerequisites),
then runs N conversations with a default LLM tutor on randomly sampled practice items.
Session IDs encode the learner ID and the sequential conversation index so that
the temporal order of interactions can be reconstructed during analysis.

Usage
-----
    python simulate_flexlearner_dataset.py [--nb-learners N] [--max-init-skills K]
                                           [--nb-conversations C] [--seed S]
                                           [--pool-id POOL_ID]

Defaults
--------
    nb_learners      = 5
    max_init_skills  = 3
    nb_conversations = 3
    seed             = 42
    pool_id          = "simulated_dataset"
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import pandas as pd
from simulation_utils import generate_learning_sequence_summary

from evalconvolearn import EvalConvoLearn
from evalconvolearn.models.tutor import Tutor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to the repo root so the script works from any cwd)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SKILL_SPACE_CSV = REPO_ROOT / "data" / "florida-doe" / "skill-space.csv"
PRACTICE_ITEMS_CSV = (
    REPO_ROOT / "data" / "florida-doe" / "tagged-practice-items-with-responses.csv"
)


# ---------------------------------------------------------------------------
# Misconception helpers
# ---------------------------------------------------------------------------


def load_misconceptions_from_csv(csv_path: str | Path) -> dict[str, str]:
    """Load misconceptions from the skill-space CSV.

    Returns a mapping from skill_id to misconception text.
    Skills without misconceptions are omitted.
    """
    df = pd.read_csv(csv_path)
    misconceptions: dict[str, str] = {}
    for _, row in df.iterrows():
        skill_id = row["skill_id"].strip().strip('"').strip("'")
        misconception = row.get("misconceptions", "")
        if pd.notna(misconception) and str(misconception).strip():
            misconceptions[skill_id] = str(misconception).strip()
    return misconceptions


def sample_learner_misconceptions(
    all_misconceptions: dict[str, str],
    mastered_skill_ids: list[str],
    all_skill_ids: list[str],
    probability: float = 0.5,
) -> dict[str, str]:
    """Sample which misconceptions a learner actively holds.

    For every skill that the learner has **not** mastered and that has a
    misconception entry, include it with the given *probability*.
    """
    active: dict[str, str] = {}
    mastered_set = set(mastered_skill_ids)
    for skill_id in all_skill_ids:
        if skill_id in mastered_set:
            continue
        if skill_id in all_misconceptions and random.random() < probability:
            active[skill_id] = all_misconceptions[skill_id]
    return active


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def build_initial_skill_set(
    sampled_skill_ids: list[str],
    skill_space,
) -> list[str]:
    """Return deduplicated list of skill IDs including all prerequisites.

    For every skill in *sampled_skill_ids* we gather all transitive prerequisites
    using ``SkillSpace.get_all_prerequisites`` and union them together with the
    original sample so the learner starts with a consistent knowledge state.
    """
    all_skills: set[str] = set(sampled_skill_ids)
    for skill_id in sampled_skill_ids:
        prereqs = skill_space.get_all_prerequisites(skill_id, return_as_ids=True)
        all_skills.update(prereqs)
    return list(all_skills)


def run_simulation(
    nb_learners: int = 5,
    max_init_skills: int = 3,
    nb_conversations: int = 7,
    seed: int = 42,
    pool_id: str = "simulated_dataset",
) -> None:
    """Run the full simulation and persist all sessions to disk."""
    random.seed(seed)

    # ------------------------------------------------------------------
    # 1. SDK + data loading
    # ------------------------------------------------------------------
    sdk = EvalConvoLearn()

    logger.info("Loading skill space from %s", SKILL_SPACE_CSV)
    skill_space = sdk.load_skill_space(SKILL_SPACE_CSV)
    logger.info("Skill space loaded: %d skills", len(skill_space.skills))

    logger.info("Loading practice items from %s", PRACTICE_ITEMS_CSV)
    item_pool = sdk.load_practice_items(PRACTICE_ITEMS_CSV, skill_space)
    logger.info("Practice item pool loaded: %d items", len(item_pool.items))

    all_skill_ids = [skill.id for skill in skill_space.skills]

    # ------------------------------------------------------------------
    # Load misconceptions from the skill-space CSV
    # ------------------------------------------------------------------
    all_misconceptions = load_misconceptions_from_csv(SKILL_SPACE_CSV)
    logger.info(
        "Loaded misconceptions for %d / %d skills.",
        len(all_misconceptions),
        len(all_skill_ids),
    )

    # ------------------------------------------------------------------
    # 2. Learner pool
    # ------------------------------------------------------------------
    pool = sdk.create_learner_pool(pool_id, skill_space)
    logger.info("Created learner pool '%s' at %s", pool.id, pool.directory_file)

    # ------------------------------------------------------------------
    # 3. Default LLM tutors (helpful/unhelpful return_only so no HTTP needed)
    # ------------------------------------------------------------------
    helpful_tutor = Tutor(
        id="default_llm_tutor",
        tutor_type="llm",
        tutor_characteristics={"helpfulness": True},
        practice_item_pool=item_pool,
        response_interaction_mode="return_only",
    )
    helpful_tutor.initialize_strategy()
    logger.info("Default LLM tutor initialized.")

    unhelpful_tutor = Tutor(
        id="unhelpful_tutor",
        tutor_type="llm",
        tutor_characteristics={"helpfulness": False},
        practice_item_pool=item_pool,
        response_interaction_mode="return_only",
    )
    unhelpful_tutor.initialize_strategy()
    logger.info("Unhelpful LLM tutor initialized.")

    # ------------------------------------------------------------------
    # 4. Create learners with sampled + prerequisite-closed skills
    # ------------------------------------------------------------------
    learners = []
    for i in range(nb_learners):
        learner_id = f"learner_{i:03d}"

        # Sample between 1 and max_init_skills skills at random
        n_sample = random.randint(1, max(1, min(max_init_skills, len(all_skill_ids))))
        sampled = random.sample(all_skill_ids, n_sample)

        # Expand with all prerequisites so the skill set is valid
        init_skills = build_initial_skill_set(sampled, skill_space)

        logger.info(
            "Learner %s: sampled %d skill(s), expanded to %d with prerequisites: %s",
            learner_id,
            len(sampled),
            len(init_skills),
            init_skills,
        )

        # Sample a persona for this learner
        response_length = random.choice(["wordy", "concise", ""])
        persona = {"response_length": response_length}

        logger.info(
            "Learner %s: persona=%s",
            learner_id,
            persona,
        )

        # Sample misconceptions for skills the learner has not mastered
        active_misconceptions = sample_learner_misconceptions(
            all_misconceptions,
            mastered_skill_ids=init_skills,
            all_skill_ids=all_skill_ids,
            probability=0.5,
        )
        logger.info(
            "Learner %s: %d active misconception(s) sampled out of %d unmastered skills.",
            learner_id,
            len(active_misconceptions),
            len(all_skill_ids) - len(init_skills),
        )

        learner = pool.create_learner(
            learner_id=learner_id,
            mastered_skills=init_skills,
            skill_space=skill_space,
            persona=persona,
            active_misconceptions=active_misconceptions,
        )
        learners.append(learner)

    logger.info("Created %d learners.", len(learners))

    # ------------------------------------------------------------------
    # 5. Run conversations
    # ------------------------------------------------------------------
    for learner in learners:
        # Sample nb_conversations practice items (with replacement if pool is small)
        replace = nb_conversations > len(item_pool.items)
        if replace:
            practice_items = random.choices(item_pool.items, k=nb_conversations)
        else:
            practice_items = random.sample(item_pool.items, nb_conversations)

        logger.info(
            "=== Learner %s | %d conversations ===",
            learner.id,
            nb_conversations,
        )

        for conv_idx, practice_item in enumerate(practice_items):
            # Session ID encodes learner and sequential conversation index
            # e.g. "learner_002__conv_001"  →  learner_002, 2nd conversation
            session_id = f"{learner.id}__conv_{conv_idx:03d}"

            logger.info(
                "  [%s] conversation with item number %d/%d | item: %.60s…",
                session_id,
                conv_idx + 1,
                nb_conversations,
                practice_item.text,
            )

            try:
                session = sdk.create_session(pool, learner, session_id=session_id)

                # sample helpful tutor 90% of the time, unhelpful tutor 10% of the time
                tutor = helpful_tutor if random.random() < 0.9 else unhelpful_tutor

                for message in session.conversation(practice_item, tutor):
                    role = message["role"].capitalize()
                    content = message["content"]
                    logger.debug("    %s: %s", role, content[:100])

                logger.info(
                    "  [%s] finished — %d messages in history.",
                    session_id,
                    len(session.dialogue_history),
                )

            except Exception:
                logger.exception(
                    "  [%s] conversation failed — skipping.",
                    session_id,
                )

    logger.info(
        "Simulation complete. Data stored under: %s",
        pool.directory_file,
    )

    # ------------------------------------------------------------------
    # 6. Post-simulation: generate learning sequence summary
    # ------------------------------------------------------------------
    conversations_file = pool.practice_conversations_file
    if conversations_file and Path(conversations_file).exists():
        logger.info("Generating learning sequence summary from %s", conversations_file)
        summary_path = generate_learning_sequence_summary(conversations_file)
        logger.info("Learning sequence summary saved to %s", summary_path)
    else:
        logger.warning(
            "Could not find conversations file at %s, skipping summary.",
            conversations_file,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a FlexLearner conversation dataset.",
    )
    parser.add_argument(
        "--nb-learners",
        type=int,
        default=5,
        help="Number of learners to simulate (default: 5).",
    )
    parser.add_argument(
        "--max-init-skills",
        type=int,
        default=3,
        help="Maximum number of skills to sample per learner at initialization (default: 3).",
    )
    parser.add_argument(
        "--nb-conversations",
        type=int,
        default=3,
        help="Number of practice conversations per learner (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--pool-id",
        type=str,
        default="simulated_dataset",
        help="Identifier for the student pool (default: 'simulated_dataset').",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simulation(
        nb_learners=args.nb_learners,
        max_init_skills=args.max_init_skills,
        nb_conversations=args.nb_conversations,
        seed=args.seed,
        pool_id=args.pool_id,
    )
