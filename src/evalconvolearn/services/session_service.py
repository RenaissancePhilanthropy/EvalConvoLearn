"""Session management service."""

import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

from ..core.base_learner import BaseLearner
from ..core.config import EvalConvoLearnConfig
from ..core.flexlearner import FlexLearner
from ..models.base_learner_conversation import (
    BaseConversationResult,
    run_base_learner_conversation,
)
from ..models.binary_skills_flexlearner import StudentPool
from ..models.practice_item import PracticeItem
from ..models.tutor import Tutor
from ..storage.file_storage import FileSessionStorage, FileStudentPoolStorage


class ConversationSession:
    """Manages a conversation session between a tutor and a flexlearner."""

    def __init__(
        self,
        session_id: str,
        learner: FlexLearner,
        student_pool: StudentPool,
        config: EvalConvoLearnConfig,
        conversation_history: list | None = None,
        session_service: "SessionService | None" = None,
    ) -> None:
        self.session_id = session_id
        self.learner = learner
        self.student_pool = student_pool
        self.config = config
        self._conversation_history = conversation_history or []
        self._session_service = session_service

    def conversation(self, practice_item: PracticeItem, tutor: Tutor) -> Generator[dict, None, None]:
        """Run a conversation on a practice item with a custom tutor.

        Args:
        ----
            practice_item: PracticeItem to practice
            tutor: Object implementing generate_response(dialogue_history) -> str

        Yields:
        ------
            dict with keys "role" ("tutor" or "learner") and "content" (the message)

        """
        from ..models.flexlearner_conversation import ConversationGraph

        pool_directory = self.student_pool.directory_file or (self.config.student_pools_dir / self.student_pool.id)
        graph_memory_path = Path(pool_directory) / self.session_id / "graph_memory.db"
        graph_memory_path.parent.mkdir(parents=True, exist_ok=True)

        conversation = ConversationGraph(
            id=str(uuid.uuid4()),
            practice_item=practice_item,
            skill_space=self.learner.skill_space,
            learner=self.learner,
            graph_memory_db_path=str(graph_memory_path),
            learning_enabled=self.config.learning_enabled,
            max_turns=self.config.max_conversation_turns,
        )

        learner_response = ""
        for chunk in conversation.run_conversation(
            session_id=self.session_id,
            start_or_resume_conversation="start",
        ):
            learner_response += chunk

        self._conversation_history.append(
            {
                "role": "user",
                "content": learner_response,
            },
        )

        self._auto_save(tutor=tutor)

        yield {"role": "learner", "content": learner_response}

        turn_count = 0
        while turn_count < self.config.max_conversation_turns:
            tutor_response = tutor.generate_response(
                self._conversation_history,
                student_pool_id=self.student_pool.id,
                learner_id=self.learner.id,
                session_id=self.session_id,
            )

            if hasattr(tutor_response, "message"):
                tutor_msg = tutor_response.message
            else:
                tutor_msg = str(tutor_response)

            self._conversation_history.append(
                {
                    "role": "assistant",
                    "content": tutor_msg,
                },
            )

            self._auto_save(tutor=tutor)

            yield {"role": "tutor", "content": tutor_msg}

            learner_response = ""
            for chunk in conversation.run_conversation(
                session_id=self.session_id,
                start_or_resume_conversation="resume",
                tutor_message=tutor_msg,
            ):
                learner_response += chunk

            self._conversation_history.append(
                {
                    "role": "user",
                    "content": learner_response,
                },
            )

            self._auto_save(tutor=tutor)

            yield {"role": "learner", "content": learner_response}

            if "Conversation ended" in learner_response:
                break

            turn_count += 1

        self._save_student_pool_practice()

    def _auto_save(
        self,
        tutor: Tutor | None = None,
    ) -> None:
        """Auto-save session state if session service is available."""
        if self._session_service:
            self._session_service.save_session(self, tutor=tutor)

    def _save_student_pool_practice(self) -> None:
        """Save practice history to student pool storage."""
        if self._session_service:
            self._session_service.save_session_pool(self.student_pool)

    @property
    def dialogue_history(self) -> list:
        """Get the conversation history."""
        return self._conversation_history

    def to_dict(self) -> dict:
        """Serialize session state to dictionary."""
        return {
            "session_id": self.session_id,
            "learner_id": self.learner.id,
            "student_pool_id": self.student_pool.id,
            "conversation_history": self._conversation_history,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        learner: FlexLearner,
        student_pool: StudentPool,
        config: EvalConvoLearnConfig,
    ) -> "ConversationSession":
        """Deserialize session from dictionary."""
        return cls(
            session_id=data["session_id"],
            learner=learner,
            student_pool=student_pool,
            config=config,
            conversation_history=data.get("conversation_history", []),
        )


class SessionService:
    """Service for managing sessions."""

    def __init__(self, config: EvalConvoLearnConfig) -> None:
        self.config = config
        self._pool_storage = FileStudentPoolStorage()
        self._session_storage = FileSessionStorage()

    def _get_session_storage_path(self, student_pool: StudentPool) -> Path:
        pool_directory = student_pool.directory_file or (self.config.student_pools_dir / student_pool.id)
        return Path(pool_directory) / "sessions"

    def create_session(
        self,
        student_pool: StudentPool,
        learner: FlexLearner,
        session_id: str | None = None,
    ) -> ConversationSession:
        """Create a new conversation session."""
        session_id = session_id or str(uuid.uuid4())

        session = ConversationSession(
            session_id=session_id,
            learner=learner,
            student_pool=student_pool,
            config=self.config,
            session_service=self,
        )

        if self._session_storage.session_exists(
            session_id,
            self._get_session_storage_path(student_pool),
        ):
            raise ValueError(
                f"Session with ID {session_id} already exists for pool {student_pool.id}",
            )
        self.save_session(session)
        return session

    def save_session(
        self,
        session: ConversationSession,
        tutor: Tutor | None = None,
    ) -> None:
        """Save session state to storage."""
        storage_path = self._get_session_storage_path(session.student_pool)
        if tutor:
            session_data = session.to_dict()
            session_data["tutor_details"] = (
                tutor.tutor_characteristics if hasattr(tutor, "tutor_characteristics") else {}
            )
        else:
            session_data = session.to_dict()
        self._session_storage.save_session_state(
            session_id=session.session_id,
            data=session_data,
            path=storage_path,
        )

    def save_session_pool(self, student_pool: StudentPool) -> None:
        """Save student pool practice history to storage."""
        self._pool_storage.save_pool(
            pool=student_pool,
            path=Path(student_pool.directory_file) / "practice.csv",
        )

    def load_session(
        self,
        session_id: str,
        student_pool: StudentPool,
        learner: FlexLearner,
    ) -> ConversationSession | None:
        """Load an existing session from storage."""
        storage_path = self._get_session_storage_path(student_pool)

        if not self._session_storage.session_exists(session_id, storage_path):
            return None

        data = self._session_storage.load_session_state(session_id, storage_path)
        session = ConversationSession.from_dict(
            data,
            learner,
            student_pool,
            self.config,
        )
        session._session_service = self
        return session

    def session_exists(self, session_id: str, student_pool: StudentPool) -> bool:
        """Check if a session exists."""
        storage_path = self._get_session_storage_path(student_pool)
        return self._session_storage.session_exists(session_id, storage_path)


class BaseConversationSession:
    """Conversation session for `BaseLearner` instances.

    Unlike `ConversationSession` (which uses the full
    `ConversationGraph`), this drives interactions through the
    `BaseLearner.start_conversation_with_problem` /
    `BaseLearner.continue_conversation` /
    `BaseLearner.end_conversation` surface.
    """

    def __init__(
        self,
        learner: BaseLearner,
        session_id: str | None = None,
        max_turns: int = 6,
    ) -> None:
        self.learner = learner
        self.session_id = session_id or str(uuid.uuid4())
        self.max_turns = max_turns
        self._history: list[dict[str, str]] = []
        self._started = False

    def start(
        self,
        practice_item: PracticeItem | str,
        initial_tutor_message: str | None = None,
        **kwargs: Any,
    ) -> dict[str, str]:
        """Start a conversation with the given problem.

        The tutor opens with *initial_tutor_message* (or a default problem
        presentation) which is added to the history before the learner's
        first call to ``start_or_continue_conversation``.
        """
        problem_text = practice_item.text if isinstance(practice_item, PracticeItem) else practice_item

        # Build the opening tutor message and seed the history
        opening = initial_tutor_message or f"Let's work on the following problem together: {problem_text}"
        self._history.append({"role": "assistant", "content": opening})

        result = self.learner.start_or_continue_conversation(
            conversation_history=self._history,
        )
        self._history.append({"role": "user", "content": result.get("response", "")})
        self._started = True
        return result

    def send_tutor_message(self, tutor_message: str, **kwargs: Any) -> dict[str, str]:
        """Send a tutor message and return the learner's reply."""
        if not self._started:
            raise RuntimeError("Call .start() before sending tutor messages.")
        self._history.append({"role": "assistant", "content": tutor_message})

        result = self.learner.start_or_continue_conversation(
            conversation_history=self._history,
        )
        self._history.append({"role": "user", "content": result.get("response", "")})
        return result

    def end(self) -> list[dict[str, str]]:
        """End the conversation and return the full history."""
        self.learner.end_conversation(conversation_history=self._history)
        return self._history

    def run_full_conversation(
        self,
        practice_item: PracticeItem | str,
        tutor: Tutor | None = None,
        tutor_responses: list[str] | None = None,
    ) -> BaseConversationResult:
        """Convenience: run a full multi-turn conversation."""
        return run_base_learner_conversation(
            learner=self.learner,
            practice_item=practice_item,
            tutor=tutor,
            max_turns=self.max_turns,
            session_id=self.session_id,
            tutor_responses=tutor_responses,
        )
