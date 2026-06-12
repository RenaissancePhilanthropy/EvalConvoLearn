import abc
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from evalconvolearn.utils.llm_client import make_client

from ..core.base_tutor import BaseTutor
from ..models.practice_item import PracticeItem, PracticeItemPool


class TutorResponse(BaseModel):
    """Standard response from tutor."""

    message: str
    metadata: dict


class LLMTutorResponse(BaseModel):
    """Response from LLM tutor strategy."""

    message: str


class LLMTutorResponseWithEndCheck(LLMTutorResponse):
    """Response from LLM tutor strategy that includes conversation end check."""

    should_end_reasoning: str
    should_conversation_end: bool


class BaseTutorStrategy(abc.ABC):
    """Abstract base class for tutor strategies."""

    @abc.abstractmethod
    def generate_strategy_response(
        self,
        dialogue_history: list[dict],
        **kwargs,
    ) -> TutorResponse:
        """Generate tutor's response based on conversation state."""

    @abc.abstractmethod
    async def start_conversation_with_practice_item(
        self,
        practice_item: PracticeItem,
        learner_context: dict | None = None,
    ) -> TutorResponse:
        """Generate tutor's response based on conversation state."""

    @abc.abstractmethod
    def initialize(self) -> None:
        """Initialize the tutor strategy (e.g., load models, establish connections)."""

    @abc.abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources when done."""


class LLMTutorStrategy(BaseTutorStrategy):
    """LLM-based tutor."""

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        tutor_characteristics: dict[str, Any] = {},
        system_prompt_template: str | None = None,
        include_answer_in_context: bool = False,
        **llm_kwargs,
    ):
        self.model = model
        self.tutor_characteristics = tutor_characteristics
        self.llm_kwargs = llm_kwargs
        self.include_answer_in_context = include_answer_in_context
        self.system_prompt_template = (
            system_prompt_template or self._default_system_prompt()
        )

    def _default_system_prompt(self) -> str:
        return (
            "You are a math tutor teaching a middle school learner with example practice problems."
            "{tutor_characteristics}"
        )

    def initialize(self) -> None:
        """Initialize LLM client."""
        # from openai import AsyncOpenAI

        load_dotenv()
        self.client = make_client(self.model)

    def generate_strategy_response(
        self,
        dialogue_history: list[dict],
        **kwargs,
    ) -> TutorResponse:
        """Generate response using LLM."""
        if not self.client:
            self.initialize()

        response_model = LLMTutorResponse
        # tutor characteristics formatting
        tutor_charact = ""
        if self.tutor_characteristics.get("helpfulness", True):
            tutor_charact += "\nAnswer to any of the learner's questions and guide them through the problem."
        else:
            tutor_charact += "\nBe unhelpful and vague in your responses to the learner and make math mistakes when answering the learner's questions."

        if self.tutor_characteristics.get("length", "short") == "long":
            tutor_charact += "\nProvide long explanations."
        else:  # limit answer length
            tutor_charact += "\nLimit your response to 2 sentences even if you only cover part of the answer."

        if self.tutor_characteristics.get("end_with_question", True):
            tutor_charact += "\nEnd your response by asking the learner to now show their solution to the problem."

        if kwargs.get("should_check_conversation_end", False):
            tutor_charact += (
                "\nEnd the conversation only if the learner has completely understood and solved all aspects of the problem."
                "\nIf the learner solved the initial problem but asks "
            )
            response_model = LLMTutorResponseWithEndCheck

        # Build few-shot block from tutor_generation_metadata if provided
        tutor_generation_metadata = kwargs.get("tutor_generation_metadata") or {}

        # if no few_shot_conversations are provided, we do not use context.
        # context may be example conversations from the same tutor OR conversations where learning happens.
        few_shot_convs: list[dict] = tutor_generation_metadata.get(
            "few_shot_conversations",
            [],
        )
        few_shot_block = ""
        if few_shot_convs:
            blocks = []
            for conv in few_shot_convs:
                item_text = conv.get("practice_item_text", "")
                dialogue = conv.get("dialogue_history", "")
                if isinstance(dialogue, list):
                    formatted_dialogue = "\n".join(
                        f"{m.get('role', 'unknown').capitalize()}: {m.get('content', '')}"
                        for m in dialogue
                        if isinstance(m, dict)
                    )
                else:
                    formatted_dialogue = str(dialogue)
                blocks.append(
                    f"### Practice item: {item_text}\n"
                    f"### Dialogue:\n{formatted_dialogue}",
                )
            few_shot_block = (
                "\n\nBelow are examples of your past tutoring conversations to help ground your response style. "
                "Use them as inspiration — do NOT copy them verbatim.\n\n"
                + "\n\n---\n\n".join(blocks)
                + "\n\n---\n\nNow continue the current conversation below."
            )

        system_content = (
            self.system_prompt_template.format(
                tutor_characteristics=tutor_charact,
            )
            + few_shot_block
        )

        # Build messages from dialogue history
        messages = [
            {
                "role": "system",
                "content": system_content,
            },
        ]

        # Add dialogue history - assistant for tutor and user for learner - should have the first message from assistant asking about the math problem and user responses
        for turn in dialogue_history:
            messages.append(
                {"role": turn.get("role", "user"), "content": turn.get("content", "")},
            )

        # Call LLM
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=response_model,
            **self.llm_kwargs,
        )

        parsed = response.choices[0].message.parsed
        choice = response.choices[0].message
        message_content = parsed.message if parsed else choice.content
        if message_content is None:
            refusal = getattr(choice, "refusal", None)
            raise ValueError(
                "LLM returned None content for tutor response"
                + (f" (refusal: {refusal})" if refusal else ""),
            )
        return TutorResponse(
            message=message_content,
            metadata={
                "model": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "should_conversation_end": (
                    parsed.should_conversation_end
                    if parsed and hasattr(parsed, "should_conversation_end")
                    else False
                ),
                "should_end_reasoning": (
                    parsed.should_end_reasoning
                    if parsed and hasattr(parsed, "should_end_reasoning")
                    else ""
                ),
            },
        )

    async def start_conversation_with_practice_item(
        self,
        practice_item: PracticeItem,
        learner_context: dict | None = None,
    ) -> TutorResponse:
        """Start conversation with practice item."""
        # Add initial message about the practice item
        # Can use the learner context to adapt the message if needed
        message = f"Let's work on the following problem together: {practice_item.text}"

        metadata = {}
        if self.include_answer_in_context:
            metadata["correct_answer"] = practice_item.get_answer()

        return TutorResponse(
            message=message,
            metadata=metadata,
        )

    async def cleanup(self) -> None:
        """Cleanup OpenAI client."""
        if self.client:
            await self.client.close()


class HumanInterfaceTutorStrategy(BaseTutorStrategy):
    """Tutor strategy for human interface. For now, only returns the provided human message."""

    def __init__(self, include_answer_in_context: bool = False, **strategy_kwargs):
        self.include_answer_in_context = include_answer_in_context
        self.strategy_kwargs = strategy_kwargs

    def initialize(self) -> None:
        """Initialize human interface tutor strategy."""

    def generate_strategy_response(
        self,
        dialogue_history: list[dict],
        **kwargs,
    ) -> TutorResponse:
        """Generate response for human interface - just return last message."""
        return TutorResponse(
            message="Human interface is not implemented yet.",
            metadata={},
        )

    async def start_conversation_with_practice_item(
        self,
        practice_item: PracticeItem,
        learner_context: dict | None = None,
    ) -> TutorResponse:
        """Start conversation with practice item for human interface."""
        message = (
            f"Let's work together on the following math problem: {practice_item.text}"
        )

        metadata = {}
        if self.include_answer_in_context:
            metadata["correct_answer"] = practice_item.get_answer()

        return TutorResponse(
            message=message,
            metadata=metadata,
        )

    async def cleanup(self) -> None:
        """Cleanup resources if any."""


class Tutor(BaseModel, BaseTutor):
    """Tutor implementations:
    - an LLM simulating a tutor with characteristics (helpful or not)
    - an interface for a human tutor or external tutoring system

    Only supports response_interaction_mode='return_only' in this version.
    HTTP interaction mode is reserved for future use.
    """

    id: str  # unique identifier for the tutor
    tutor_type: Literal[
        "llm",
        "human_interface",
        "external_API",
    ]  # type of tutor implementation
    tutor_characteristics: dict = Field(
        default_factory=dict,
    )  # characteristics of the tutor (e.g., helpfulness level, style)
    practice_item_pool: PracticeItemPool | None = (
        None  # practice item pool used by the tutor
    )
    response_interaction_mode: Literal[
        "http",
        "return_only",
    ] = "return_only"  # reserved: only 'return_only' is implemented
    include_answer_in_context: bool = (
        False  # whether to include the practice item answer in tutor context/metadata
    )
    # This is a private attribute, not part of the model schema (model_dump)
    _strategy: BaseTutorStrategy | None = (
        None  # internal strategy instance derived from tutor_type
    )

    def __init__(self, **data):
        """Initialize Tutor."""
        super().__init__(**data)
        if self.response_interaction_mode == "http":
            msg = (
                "HTTP interaction mode is not implemented in this version. "
                "Use response_interaction_mode='return_only'."
            )
            raise NotImplementedError(msg)
        self._strategy = None
        self.initialize_tutor_characteristics()

    def initialize_tutor_characteristics(self):
        """Initialize tutor characteristics with defaults if not provided."""
        defaults = {
            "helpfulness": True,
        }
        for key, value in defaults.items():
            # set default characteristics if not already provided
            self.tutor_characteristics.setdefault(key, value)

    # tutor loading information
    def load_practice_item_pool(self, item_pool: PracticeItemPool):
        """Load a practice item pool for the tutor to use."""
        # extra validation as needed
        self.practice_item_pool = item_pool

    def initialize_strategy(
        self,
        model: str = "gpt-4.1-mini",
        **strategy_kwargs: object,
    ):
        """Initialize strategy based on tutor_type, an async operation because can be an external API call."""
        if self.tutor_type == "llm":
            self._strategy = LLMTutorStrategy(
                model=model,
                tutor_characteristics=self.tutor_characteristics,
                include_answer_in_context=self.include_answer_in_context,
                **strategy_kwargs,
            )
        elif self.tutor_type == "human_interface":
            self._strategy = HumanInterfaceTutorStrategy(
                include_answer_in_context=self.include_answer_in_context,
                **strategy_kwargs,
            )
        elif self.tutor_type == "external_API":
            # use **strategy_kwargs to pass external API details
            raise NotImplementedError(
                "External API tutor strategy not implemented yet.",
            )
        else:
            raise ValueError(f"Unknown tutor_type: {self.tutor_type}")

        self._strategy.initialize()

    async def get_teaching_first_message(
        self,
        practice_item: PracticeItem | None = None,
        learner_context: dict | None = None,
    ) -> TutorResponse:
        """Get the first message from the tutor to start teaching a learner."""
        if not self._strategy:
            raise ValueError(
                "Tutor strategy not initialized. Call initialize_strategy() first.",
            )
        if not practice_item:
            if (not self.practice_item_pool) or (len(self.practice_item_pool) == 0):
                raise ValueError(
                    "No practice item provided and no practice item pool loaded.",
                )
            # practice item pool length > 0
            practice_item = self.practice_item_pool.items[
                0
            ]  # for now, just take the first item
        # Initial conversation with empty dialogue history
        return await self._strategy.start_conversation_with_practice_item(
            practice_item=practice_item,
            learner_context=learner_context,
        )

    def get_teacher_followup_message(
        self,
        dialogue_history: list[dict] | None = None,
        **kwargs,
    ) -> TutorResponse:
        """Get the follow up message from the tutor."""
        if not self._strategy:
            raise ValueError(
                "Tutor strategy not initialized. Call initialize_strategy() first.",
            )
        if not dialogue_history:
            raise ValueError(
                "No dialogue history passed to generate teacher followup message.",
            )
        return self._strategy.generate_strategy_response(
            dialogue_history=dialogue_history,
            **kwargs,
        )

    def generate_response(
        self,
        dialogue_history: list[dict],
        **kwargs,
    ) -> TutorResponse:
        """Generate a tutor response based on dialogue history."""
        if self.tutor_type == "human_interface":
            msg = "generate_response does not work with human interface tutor."
            raise ValueError(msg)
        return self.get_teacher_followup_message(
            dialogue_history=dialogue_history,
            **kwargs,
        )
