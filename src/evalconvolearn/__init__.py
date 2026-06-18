"""evalconvolearn: Learner simulation framework for evaluating conversational AI tutors."""

from importlib.metadata import version

from .core.base_learner import BaseLearner
from .core.base_tutor import BaseTutor, load_effective_conversations
from .core.config import EvalConvoLearnConfig
from .core.flexlearner import FlexLearner
from .core.sdk import (
    BenchmarkRunSummary,
    EvalConvoLearn,
    EvalSetResults,
    EvaluationResults,
)
from .models.binary_skills_flexlearner import BinarySkillsFlexLearner, StudentPool
from .models.evaluation import EvaluationConfig, LearnerEvalConfig
from .models.practice_item import PracticeItem, PracticeItemPool
from .models.skill import Skill, SkillSpace

__version__ = version("evalconvolearn")

__all__ = [
    "BaseLearner",
    "BaseTutor",
    "BinarySkillsFlexLearner",
    "BenchmarkRunSummary",
    "EvalConvoLearn",
    "EvaluationConfig",
    "EvaluationResults",
    "EvalSetResults",
    "FlexLearner",
    "EvalConvoLearnConfig",
    "LearnerEvalConfig",
    "load_effective_conversations",
    "PracticeItem",
    "PracticeItemPool",
    "Skill",
    "SkillSpace",
    "StudentPool",
]
