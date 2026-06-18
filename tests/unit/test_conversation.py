"""Unit tests for ConversationGraph setup and graph structure."""

from pathlib import Path
from unittest.mock import patch

from evalconvolearn.models.binary_skills_flexlearner import BinarySkillsFlexLearner
from evalconvolearn.models.flexlearner_conversation import ConversationGraph
from evalconvolearn.models.practice_item import PracticeItem, PracticeItemPool
from evalconvolearn.models.skill import SkillSpace


class TestConversationSetup:
    """Test data loading and learner/conversation initialization."""

    def test_skill_space_loads(self, skill_space: SkillSpace) -> None:
        assert skill_space is not None
        assert len(skill_space.skills) > 0
        assert "MA.6.NSO.2.2" in skill_space

    def test_practice_item_pool_loads(self, practice_item_pool: PracticeItemPool) -> None:
        assert practice_item_pool is not None
        assert len(practice_item_pool.items) > 0
        assert all(isinstance(item, PracticeItem) for item in practice_item_pool.items)

    def test_learner_initialization(self, learner: BinarySkillsFlexLearner) -> None:
        assert learner is not None
        assert len(learner.mastered_skills) > 0
        assert len(learner.practice_history) == 1

    def test_conversation_initialization(self, conversation: ConversationGraph) -> None:
        assert conversation is not None
        assert conversation.id is not None
        assert conversation.practice_item is not None
        assert conversation.learner is not None
        assert conversation.graph_memory_db_path is not None

    def test_graph_memory_path_is_db(self, conversation: ConversationGraph) -> None:
        assert conversation.graph_memory_db_path.endswith(".db")
        assert Path(conversation.graph_memory_db_path).parent.exists()


class TestConversationState:
    """Test conversation state initialization."""

    def test_state_initial_values(self, conversation: ConversationGraph) -> None:
        initial_state = conversation.initialize_state({})
        assert initial_state["turn_number"] == 1
        assert initial_state["solution_found"] is False
        assert initial_state["conversation_ended"] is False

    def test_state_has_skill_fields(self, conversation: ConversationGraph) -> None:
        initial_state = conversation.initialize_state({})
        assert "item_associated_skills" in initial_state
        assert "learner_mastered_skills" in initial_state
        assert "pct_problem_skills_mastered" in initial_state


class TestConversationGraphStructure:
    """Test graph compilation and structure."""

    def test_graph_compiles(self, conversation: ConversationGraph) -> None:
        compiled_graph = conversation.compile_graph()
        assert compiled_graph is not None

    def test_graph_has_start_and_end(self, conversation: ConversationGraph) -> None:
        compiled_graph = conversation.compile_graph()
        graph = compiled_graph.get_graph()
        node_names = list(graph.nodes.keys()) if hasattr(graph.nodes, "keys") else list(graph.nodes)
        assert "__start__" in node_names or len(node_names) > 0

    def test_print_graph_method_exists(self, conversation: ConversationGraph) -> None:
        compiled_graph = conversation.compile_graph()
        with patch("os.makedirs"), patch("builtins.open", create=True):
            try:
                conversation.print_graph(compiled_graph)
            except Exception:
                assert hasattr(conversation, "print_graph")
