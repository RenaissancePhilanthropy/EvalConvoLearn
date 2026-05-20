"""Example demonstrating creating, saving, and loading learner pools with multiple sessions."""

from pathlib import Path

from evalconvolearn import EvalConvoLearn
from evalconvolearn.models.tutor import Tutor


def run_session(sdk, pool, learner, item, tutor, session_id):
    """Helper function to run a single conversation session."""
    print(f"\n{'='*60}")
    print(f"SESSION: {session_id}")
    print(f"{'='*60}")
    print(f"Practice Item: {item.text}")

    session = sdk.create_session(pool, learner, session_id=session_id)

    for message in session.conversation(item, tutor):
        role = message["role"].capitalize()
        content = message["content"]
        print(f"{role}: {content}")

    print(f"\nSession complete: {len(session.dialogue_history)} messages")

    return session


def main():
    # Initialize SDK
    sdk = EvalConvoLearn()

    # Load skill space and practice items
    skill_space = sdk.load_skill_space(
        Path("data") / "florida-doe" / "skill-space.csv",
    )

    items = sdk.load_practice_items(
        Path("data") / "florida-doe" / "tagged-practice-items-with-responses.csv",
        skill_space,
    )

    print("=" * 60)
    print("CREATING NEW LEARNER POOL")
    print("=" * 60)

    # Create a new learner pool (will have timestamp appended)
    pool = sdk.create_learner_pool("demo_pool", skill_space)
    print(f"Created pool: {pool.id}")
    print(f"Pool directory: {pool.directory_file}")

    # Add multiple learners with different mastered skills
    learner1 = pool.create_learner(
        learner_id="student_alice",
        mastered_skills=["MA.6.NSO.2.1", "MA.6.NSO.2.2"],
        skill_space=skill_space,
    )

    learner2 = pool.create_learner(
        learner_id="student_bob",
        mastered_skills=["MA.6.NSO.2.1"],
        skill_space=skill_space,
    )

    print(f"Added learners: {learner1.id}, {learner2.id}")

    # Create LLM-based helpful tutor
    print("\n" + "=" * 60)
    print("INITIALIZING LLM TUTOR")
    print("=" * 60)

    tutor = Tutor(
        id="helpful_tutor_001",
        tutor_type="llm",
        tutor_characteristics={"helpfulness": True},
        practice_item_pool=items,
        response_interaction_mode="return_only",
    )
    tutor.initialize_strategy()
    print("Tutor initialized with helpful characteristics")

    # Run multiple sessions with different learners and items
    print("\n" + "=" * 60)
    print("RUNNING MULTIPLE SESSIONS")
    print("=" * 60)

    session1 = run_session(
        sdk,
        pool,
        learner1,
        items.items[0],
        tutor,
        "session_alice_1",
    )
    # session2 = run_session(sdk, pool, learner2, items.items[1], tutor, "session_bob_1")
    # session3 = run_session(sdk, pool, learner1, items.items[2], tutor, "session_alice_2")

    print(f"\n{'='*60}")
    print("POOL STATE AFTER SESSIONS")
    print(f"{'='*60}")
    print(f"Total learners in pool: {len(pool.learners)}")
    print("Sessions run: 1")
    print(f"Pool saved at: {pool.directory_file}")

    # Now demonstrate loading the pool back
    print(f"\n{'='*60}")
    print("LOADING EXISTING POOL")
    print(f"{'='*60}")

    # Load the most recent pool with this ID
    loaded_pool = sdk.load_student_pool_most_recent("demo_pool", skill_space)
    print(f"Loaded pool: {loaded_pool.id}")
    print(f"Loaded from: {loaded_pool.directory_file}")
    print(f"Number of learners: {len(loaded_pool.learners)}")
    print(f"Learner IDs: {[l.id for l in loaded_pool.learners]}")

    # Access learners from loaded pool
    loaded_learner = loaded_pool.get_learner("student_alice")
    print(f"\nAccessed learner: {loaded_learner.id}")
    print(f"Mastered skills: {len(loaded_learner.mastered_skills)}")

    # Run another session with the loaded pool
    print(f"\n{'='*60}")
    print("RUNNING SESSION WITH LOADED POOL")
    print(f"{'='*60}")

    session4 = run_session(
        sdk,
        loaded_pool,
        loaded_learner,
        items.items[3],
        tutor,
        "session_alice_3",
    )

    print(f"\n{'='*60}")
    print("DEMONSTRATION COMPLETE")
    print(f"{'='*60}")
    print(f"Pool ID: {loaded_pool.id}")
    print("Total sessions run: 4")
    print(f"All data saved to: {loaded_pool.directory_file}")


if __name__ == "__main__":
    main()
