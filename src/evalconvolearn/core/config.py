"""Configuration for EvalConvoLearn SDK."""

from pathlib import Path
from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class EvalConvoLearnConfig(BaseSettings):
    """Configuration for EvalConvoLearn library."""

    # Data directories
    data_dir: Path = Field(default=Path("./data"))
    student_pools_dir: Path = Field(default=Path("./data/student_pools"))

    # Dataset paths (loaded from env vars)
    skill_space_path: Path | None = Field(default=None)
    tagged_practice_items_with_responses_csv: Path | None = Field(default=None)
    oversampled_items_csv: Path | None = Field(default=None)

    # Conversation settings
    max_conversation_turns: int = Field(default=6)
    learning_enabled: bool = Field(default=True)

    # evaluation settings
    evaluations_dir: str = Field(default="./outputs/")

    @model_validator(mode="before")
    @classmethod
    def set_student_pools_dir_default(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("student_pools_dir") is None:
            data_dir = data.get("data_dir", Path("./data"))
            data["student_pools_dir"] = Path(data_dir) / "student_pools"
        return data
