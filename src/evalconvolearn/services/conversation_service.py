"""Conversation orchestration service."""

from ..core.config import EvalConvoLearnConfig


class ConversationService:
    """Handles conversation logic."""

    def __init__(self, config: EvalConvoLearnConfig):
        self.config = config
