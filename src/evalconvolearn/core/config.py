"""Configuration for EvalConvoLearn SDK."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class EvalConvoLearnConfig(BaseSettings):
    """Configuration for EvalConvoLearn library."""

    # Data directories
    data_dir: Path = Field(default=Path("./data"))
    student_pools_dir: Path | None = None

    # Conversation settings
    max_conversation_turns: int = Field(default=6)
    learning_enabled: bool = Field(default=True)

    # evaluation settings
    evaluations_dir: str = Field(default="./data/evaluations/learner_evaluations")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set defaults based on data_dir
        if self.student_pools_dir is None:
            self.student_pools_dir = self.data_dir / "student_pools"
