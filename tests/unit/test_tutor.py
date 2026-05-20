"""Unit tests for Tutor model."""


class TestTutorInitialization:
    """Test suite for tutor initialization."""

    def test_helpful_tutor_initialization(self, helpful_tutor):
        """Test that helpful tutor initializes correctly."""
        assert helpful_tutor is not None
        assert helpful_tutor.id is not None
        assert helpful_tutor.tutor_type == "llm"
        assert helpful_tutor.tutor_characteristics.get("helpfulness") is True

    def test_unhelpful_tutor_initialization(self, unhelpful_tutor):
        """Test that unhelpful tutor initializes correctly."""
        assert unhelpful_tutor is not None
        assert unhelpful_tutor.id is not None
        assert unhelpful_tutor.tutor_type == "llm"
        assert unhelpful_tutor.tutor_characteristics.get("helpfulness") is False

    def test_tutor_has_practice_item_pool(self, helpful_tutor, practice_item_pool):
        """Test that tutor has access to practice item pool."""
        assert helpful_tutor.practice_item_pool is not None
        assert helpful_tutor.practice_item_pool == practice_item_pool


class TestTutorBehavior:
    """Test suite for tutor behavior."""

    def test_tutor_response_mode(self, helpful_tutor):
        """Test that tutor has correct response interaction mode."""
        assert helpful_tutor.response_interaction_mode == "return_only"

    def test_tutor_characteristics_stored(self, helpful_tutor, unhelpful_tutor):
        """Test that tutor characteristics are properly stored."""
        assert isinstance(helpful_tutor.tutor_characteristics, dict)
        assert isinstance(unhelpful_tutor.tutor_characteristics, dict)
        # Characteristics should be different between helpful and unhelpful
        assert (
            helpful_tutor.tutor_characteristics != unhelpful_tutor.tutor_characteristics
        )
