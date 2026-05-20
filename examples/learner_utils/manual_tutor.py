"""Interactive command-line tutoring session using a FlexLearner."""

from pathlib import Path

from evalconvolearn import EvalConvoLearn
from evalconvolearn.core.base_tutor import BaseTutor


class ManualTutor(BaseTutor):
    """A tutor that prompts the user for responses on the command line."""

    def generate_response(self, dialogue_history: list[dict]) -> str:
        return input("\n[YOUR RESPONSE as tutor]: ").strip()


def main():
    sdk = EvalConvoLearn()

    skill_space = sdk.load_skill_space(Path("data") / "florida-doe" / "skill-space.csv")
    items = sdk.load_practice_items(
        Path("data") / "florida-doe" / "tagged-practice-items-with-responses.csv",
        skill_space,
    )

    # Option 1: Create new pool (timestamp appended automatically)
    pool = sdk.create_learner_pool("command_line_test_pool", skill_space)

    # Option 2: Load the most recent existing pool instead:
    # pool = sdk.load_student_pool_most_recent("command_line_test_pool", skill_space)

    learner = pool.create_learner(
        learner_id="student_1",
        mastered_skills=["MA.6.NSO.2.1"],
        skill_space=skill_space,
    )
    practice_item = items.items[0]
    tutor = ManualTutor()

    session_id = "manual_session_001"
    conversation_session = sdk.create_session(pool, learner, session_id=session_id)

    print("\n" + "=" * 60)
    print("MANUAL TUTORING SESSION")
    print("=" * 60)
    print(f"Practice Item: {practice_item.text}")
    print(f"Session ID: {session_id}")
    print(f"Learner: {learner.id}")
    print("\nType your responses as the tutor. The conversation will continue")
    print("until the learner says they're done or max turns is reached.")
    print("=" * 60)

    try:
        for message in conversation_session.conversation(practice_item, tutor):
            role = message["role"].upper()
            content = message["content"]
            if role == "LEARNER":
                print(f"\n[{role}]: {content}")

        print("\n" + "=" * 60)
        print("CONVERSATION ENDED")
        print("=" * 60)

    except KeyboardInterrupt:
        conversation_session._auto_save()
        print("\n\nSession interrupted. Progress has been saved.")

    print(f"\nTotal messages exchanged: {len(conversation_session.dialogue_history)}")
    print(f"Pool ID: {pool.id}")
    print(f"Pool directory: {pool.directory_file}")


if __name__ == "__main__":
    main()
