import abc
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from ..models.practice_item import PracticeItem, PracticeItemPool


class LearnerResponse(BaseModel):
    """Response schema returned by the learner HTTP endpoint."""

    dialogue_history: Any = ""
    conversation_ended: Any = ""
    session_id: str = ""


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
        from openai import OpenAI

        load_dotenv()
        self.client = OpenAI()

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
            tutor_charact += "\nEnd the conversation only if the learner has completely understood and solved all aspects of the problem."
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


class Tutor(BaseModel):
    """Possible tutor implementations:
    - an LLM simulating a tutor with characteristics (helpful or not)
    - an interface for a human tutor or external tutoring system

    Allows to interact with the learner over HTTP or only return tutor responses for an external client or a platform running a learner instance directly.

    Provides a teach_learner method that runs a full conversation with a learner on a practice item.
    Returns the conversation metadata.
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
    ] = "http"  # whether the tutor makes API calls to the learner or just returns responses
    include_answer_in_context: bool = (
        False  # whether to include the practice item answer in tutor context/metadata
    )
    # This is a private attribute, not part of the model schema (model_dump)
    _strategy: BaseTutorStrategy | None = (
        None  # internal strategy instance derived from tutor_type
    )
    _learner_http_details: dict | None = (
        None  # internal details of the response (e.g., API call info)
    )

    def __init__(self, **data):
        """Initialize Tutor."""
        super().__init__(**data)
        self._strategy = None
        self.initialize_tutor_characteristics()
        if self.response_interaction_mode == "http":
            self.initialize_http_details()

    def initialize_tutor_characteristics(self):
        """Initialize tutor characteristics with defaults if not provided."""
        defaults = {
            "helpfulness": True,
        }
        for key, value in defaults.items():
            # set default characteristics if not already provided
            self.tutor_characteristics.setdefault(key, value)

    def initialize_http_details(self):
        """Initialize response details."""
        # handle headers authorization here if needed for communication with learner API

        self._learner_http_details = {
            "base_url": "http://localhost:8000/student_pool/",
            "start_teaching_endpoint": "start_teaching_learner",
            "continue_teaching_endpoint": "teaching_response",
            "start_assessing_endpoint": "start_assessing_learner",
            "continue_assessing_endpoint": "assessing_response",
            "http_client": httpx.AsyncClient(
                timeout=30.0,
                headers={"Content-Type": "application/json"},
            ),
        }

    # tutor loading information
    def load_practice_item_pool(self, item_pool: PracticeItemPool):
        """Load a practice item pool for the tutor to use."""
        # extra validation as needed
        self.practice_item_pool = item_pool

    def initialize_strategy(self, **strategy_kwargs):
        """Initialize strategy based on tutor_type, an async operation because can be an external API call."""
        if self.tutor_type == "llm":
            self._strategy = LLMTutorStrategy(
                model="gpt-4.1-mini",
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

    # TODO - understand how we need to handle get_teaching_first_message
    # this method is not used in all tutor instances, most of times only
    # the get_teacher_followup_message will be used to generate a response
    # and the conversation starts directly from the conversation graph
    # right now this is not being used because we removed student_pool.session
    # --> this should be adapted to support clean api access for tutors
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

    def get_learner_response(
        self,
        api_endpoint: str,
        json_payload: dict,
    ) -> LearnerResponse:
        """Make an API call to the learner endpoint and get the response."""
        if not self._learner_http_details:
            raise ValueError(
                "Learner HTTP details not initialized. Ensure response_interaction_mode is 'http'.",
            )

        url = self._learner_http_details["base_url"] + api_endpoint

        with self._learner_http_details["http_client"] as client:
            response = client.post(url, json=json_payload)
            response.raise_for_status()
            data = response.json()

        # TODO Can validate model directly by passing the response json to model
        return LearnerResponse(
            dialogue_history=data.get("dialogue_history", ""),
            conversation_ended=data.get("conversation_ended", ""),
            session_id=data.get("session_id", ""),
        )

    # external method to start teaching a learner
    async def start_teaching(
        self,
        student_pool_id: str,  # required
        learner_id: str,  # required - the learner to be taught
        session_id: (
            str | None
        ) = None,  # not required - session id can be passed or not to learner
        practice_item: (
            PracticeItem | None
        ) = None,  # Would sample a practice item if not provided
        initial_message: (
            str | None
        ) = None,  # optional initial message to start the conversation
        learner_context: (
            dict | None
        ) = None,  # tutor implementation may use the learner context to adapt teaching
    ) -> TutorResponse | LearnerResponse:
        """Start a teaching conversation with a learner on a practice item.
        Either:
        -Send the 'initial message' provided to the learner and return the learner response.
        -Return the tutor response if response_interaction_mode is 'return_only'.
        -Get and send the tutor's response to the learner and return the learner response.
        """
        if (
            self.response_interaction_mode == "return_only"
            and self.tutor_type == "human_interface"
        ):
            raise ValueError("Human interface tutor cannot use Return Only mode.")

        if self.tutor_type == "human_interface":
            # human interface: require an initial message to start the conversation
            # Should send the initial message to the learner
            assert (
                initial_message
            ), "Human interface tutor requires an initial message to start the conversation."
            assert self._learner_http_details, "Learner HTTP details not initialized."
        else:
            # Other types of tutors: generate the first message
            tutor_response = await self.get_teaching_first_message(
                practice_item=practice_item,
                learner_context=learner_context,
            )
            initial_message = tutor_response.message
            if self.response_interaction_mode == "return_only":
                # just return the tutor response - will be used in the interface to display to human tutor and query the learner.
                return tutor_response
        # else, send the tutor response to the learner and get the learner response
        assert self._learner_http_details, "Learner HTTP details not initialized."
        payload = {
            "student_pool_id": student_pool_id,
            "learner_id": learner_id,
            "session_id": session_id,
            "initial_message": initial_message,
            "practice_item": practice_item.text if practice_item else "",
            "skills": practice_item.associated_skills if practice_item else [],
            "practice_item_answer": practice_item.get_answer() if practice_item else "",
        }
        return await self.get_learner_response(
            api_endpoint=self._learner_http_details["start_teaching_endpoint"],
            json_payload=payload,
        )

    def continue_teaching(
        self,
        student_pool_id: str,  # required
        learner_id: str,
        session_id: str,  # required for continued session teaching
        provided_response: str | None = None,
        dialogue_history: list[dict] | None = None,
        **kwargs,
    ) -> TutorResponse | LearnerResponse:
        """Continue a teaching conversation with a learner using tutor's response."""
        if (
            self.response_interaction_mode == "return_only"
            and self.tutor_type == "human_interface"
        ):
            raise ValueError("Human interface tutor cannot use Return Only mode.")

        if self.tutor_type == "human_interface":
            assert (
                provided_response
            ), "Human interface tutor requires a provided response message to continue the conversation."
        else:
            # Other types of tutors: generate the follow up tutor message
            tutor_response = self.get_teacher_followup_message(
                dialogue_history=dialogue_history,
                **kwargs,
            )
            provided_response = tutor_response.message
            # even if the tutor determines that the conversation should end, we do nothing here for now.
            if self.response_interaction_mode == "return_only":
                return tutor_response

        new_dialogue_history = dialogue_history or []
        new_dialogue_history.append({"role": "assistant", "content": provided_response})

        assert self._learner_http_details, "Learner HTTP details not initialized."
        payload = {
            "student_pool_id": student_pool_id,
            "learner_id": learner_id,
            "session_id": session_id,
            "dialogue_history": new_dialogue_history,
        }
        return self.get_learner_response(
            api_endpoint=self._learner_http_details["continue_teaching_endpoint"],
            json_payload=payload,
        )

    async def teach_learner(
        self,
        student_pool_id: str,
        learner_id: str,
        session_id: str | None = None,
        practice_item: PracticeItem | None = None,
        learner_context: dict | None = None,
    ):
        """Run a full teaching conversation with a learner on a practice item.
        This should happen only if:
        -the tutor_type is not 'human_interface'
        -the response_interaction_mode is set to 'http' and not 'return_only'

        Should return the dialogue history and learner response metadata.
        """
        assert (
            self.response_interaction_mode == "http"
        ), "teach_learner only works in HTTP interaction mode."
        assert (
            self.tutor_type != "human_interface"
        ), "teach_learner does not work with human interface tutor. Use the methods: start_teaching and continue_teaching"

        # Start teaching
        learner_response = await self.start_teaching(
            student_pool_id=student_pool_id,
            learner_id=learner_id,
            session_id=session_id,
            practice_item=practice_item,
            learner_context=learner_context,
        )
        # Should retrieve the session id and the dialogue history from the learner response metadata
        # To be changed based on the implementation of the learner response schema
        dialogue_history = learner_response.get("dialogue_history", [])

        # session id can come either from the initial session id or from learner session response
        session_id = session_id or learner_response.get("session_id", None)

        assert session_id, "No session id provided for continued session teaching."
        assert (
            len(dialogue_history) > 0
        ), "Dialogue history not found or empty in learner response metadata."

        # continue teaching until done (get conversation_ended from learner response metadata)
        done = False
        while not done:
            learner_response = await self.continue_teaching(
                student_pool_id=student_pool_id,
                learner_id=learner_id,
                session_id=session_id,
                dialogue_history=dialogue_history,
            )
            # Update dialogue history
            dialogue_history = learner_response.get(
                "dialogue_history",
                dialogue_history,
            )
            done = learner_response.get("conversation_ended", False)

        # save or return the conversation information
        return {
            "student_pool_id": student_pool_id,
            "learner_id": learner_id,
            "session_id": session_id,
            "dialogue_history": dialogue_history,
            "learner_response": learner_response,
        }

    def generate_response(
        self,
        dialogue_history: list[dict],
        **kwargs,
    ) -> TutorResponse:
        """Generate a tutor response based on dialogue history."""
        student_pool_id = kwargs.get("student_pool_id", "")
        learner_id = kwargs.get("learner_id", "")
        session_id = kwargs.get("session_id")

        assert (
            student_pool_id or self.response_interaction_mode == "return_only"
        ), "student_pool_id is required to generate tutor response when not using return_only mode."
        assert (
            learner_id or self.response_interaction_mode == "return_only"
        ), "learner_id is required to generate tutor response when not using return_only mode."
        assert (
            session_id or self.response_interaction_mode == "return_only"
        ), "session_id is required to generate tutor response when not using return_only mode."

        tutor_response = self.continue_teaching(
            student_pool_id=student_pool_id,
            learner_id=learner_id,
            session_id=session_id,
            dialogue_history=dialogue_history,
            should_check_conversation_end=kwargs.get(
                "should_check_conversation_end",
                False,
            ),
        )
        return tutor_response
