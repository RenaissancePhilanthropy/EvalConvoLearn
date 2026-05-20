import uuid

import pytest

from evalconvolearn.models.flexlearner_conversation import ConversationGraph


@pytest.fixture()
def conversation(learner, practice_item_pool, skill_space):
    """A ConversationGraph for unit testing using the first practice item in the pool."""
    practice_item = practice_item_pool.items[0]
    return ConversationGraph(
        id=str(uuid.uuid4()),
        practice_item=practice_item,
        skill_space=skill_space,
        learner=learner,
        resolve_confusion_style="",
    )
