"""Minimal example of running a FlexLearner conversation with a custom tutor."""

from pathlib import Path

from evalconvolearn import EvalConvoLearn
from evalconvolearn.core.base_tutor import BaseTutor


class MyCustomTutor(BaseTutor):
    """Simple rule-based tutor."""

    def generate_response(self, dialogue_history: list[dict]) -> str:
        last_message = dialogue_history[-1]["content"] if dialogue_history else ""
        if "?" in last_message:
            return "Let's think about this step by step. What do you already know?"
        return "Good thinking! Can you explain your reasoning?"


def main():
    sdk = EvalConvoLearn()

    skill_space = sdk.load_skill_space(Path("data") / "florida-doe" / "skill-space.csv")
    items = sdk.load_practice_items(
        Path("data") / "florida-doe" / "tagged-practice-items-with-responses.csv",
        skill_space,
    )

    pool = sdk.create_learner_pool("my_test_pool", skill_space)
    learner = pool.create_learner(
        learner_id="learner_1",
        mastered_skills=["MA.6.NSO.2.1"],
        skill_space=skill_space,
    )

    my_tutor = MyCustomTutor()
    session = sdk.create_session(pool, learner)

    print("Starting conversation...")
    for message in session.conversation(items.items[0], my_tutor):
        print(f"{message['role'].capitalize()}: {message['content']}\n")

    print("Conversation complete!")
    print(f"Final dialogue history: {len(session.dialogue_history)} messages")


if __name__ == "__main__":
    main()
