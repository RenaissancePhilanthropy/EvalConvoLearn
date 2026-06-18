import uuid

import pytest

from evalconvolearn.models.binary_skills_flexlearner import BinarySkillsFlexLearner
from evalconvolearn.models.flexlearner_conversation import ConversationGraph
from evalconvolearn.models.practice_item import PracticeItemPool
from evalconvolearn.models.skill import SkillSpace


@pytest.fixture()
def conversation(
    learner: BinarySkillsFlexLearner, practice_item_pool: PracticeItemPool, skill_space: SkillSpace
) -> ConversationGraph:
    """A ConversationGraph for unit testing using the first practice item in the pool."""
    practice_item = practice_item_pool.items[0]
    return ConversationGraph(
        id=str(uuid.uuid4()),
        practice_item=practice_item,
        skill_space=skill_space,
        learner=learner,
        resolve_confusion_style="",
    )
