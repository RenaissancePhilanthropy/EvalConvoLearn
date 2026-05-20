"""Abstract storage interfaces."""

from abc import ABC, abstractmethod
from pathlib import Path

from ..models.binary_skills_flexlearner import StudentPool
from ..models.skill import SkillSpace


class StudentPoolStorage(ABC):
    """Abstract interface for student pool persistence."""

    @abstractmethod
    def save_pool(self, pool: StudentPool, path: Path) -> None:
        """Save student pool to storage."""

    @abstractmethod
    def load_pool(self, path: Path, skill_space: SkillSpace) -> StudentPool:
        """Load student pool from storage."""

    @abstractmethod
    def pool_exists(self, path: Path) -> bool:
        """Check if pool exists."""


class SessionStorage(ABC):
    """Abstract interface for session persistence."""

    @abstractmethod
    def save_session_state(self, session_id: str, data: dict, path: Path) -> None:
        """Save session state."""

    @abstractmethod
    def load_session_state(self, session_id: str, path: Path) -> dict:
        """Load session state."""

    @abstractmethod
    def session_exists(self, session_id: str, path: Path) -> bool:
        """Check if session exists."""
