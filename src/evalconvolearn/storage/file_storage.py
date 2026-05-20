"""File-based storage implementation."""

import json
from pathlib import Path

from ..models.binary_skills_flexlearner import StudentPool
from ..models.skill import SkillSpace
from .base import SessionStorage, StudentPoolStorage


class FileStudentPoolStorage(StudentPoolStorage):
    """CSV/JSON-based student pool storage."""

    def save_pool(self, pool: StudentPool, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pool.save_student_pool_practice_history_to_csv(path)

    def load_pool(self, path: Path, skill_space: SkillSpace) -> StudentPool:
        if not path.exists():
            raise FileNotFoundError(f"Student pool file {path} not found")

        dir_name = path.parent.name
        pool_id = "_".join(dir_name.split("_")[:-1]) if "_" in dir_name else dir_name

        pool = StudentPool(
            id=pool_id,
            skill_space=skill_space,
            directory_file=path.parent,
        )
        pool.load_student_pool_from_csv(path, skill_space)
        return pool

    def pool_exists(self, path: Path) -> bool:
        return path.exists()


class FileSessionStorage(SessionStorage):
    """File-based session storage."""

    def save_session_state(self, session_id: str, data: dict, path: Path) -> None:
        session_file = path / f"{session_id}.json"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        with open(session_file, "w") as f:
            json.dump(data, f, indent=2)

    def load_session_state(self, session_id: str, path: Path) -> dict:
        session_file = path / f"{session_id}.json"
        if not session_file.exists():
            return {}

        with open(session_file) as f:
            return json.load(f)

    def session_exists(self, session_id: str, path: Path) -> bool:
        return (path / f"{session_id}.json").exists()
