import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from openai import OpenAI
from pydantic import BaseModel, Field, PrivateAttr

from ..core.flexlearner import FlexLearner
from ..utils.data_loaders import render_conversation_messages
from .practice_item import PracticeItem
from .skill import Skill, SkillSpace

# Configure module logger
logger = logging.getLogger(__name__)


class SolutionReadinessResponse(BaseModel):
    reasoning: str
    is_learner_ready: bool


class SolutionProposalResponse(BaseModel):
    reasoning: str
    response: str


class ConversationEndingResponse(BaseModel):
    reasoning: str
    conversation_should_end: bool


class StartConfusionResponse(BaseModel):
    reasoning: str
    current_confusion: str
    response: str


class ContinueConfusionResponse(BaseModel):
    reasoning: str
    response: str


class ResolvedConfusionResponse(BaseModel):
    reasoning: str
    resolved_confusion: bool


class ConversationState(TypedDict):
    """State object for conversation graph."""

    turn_number: int
    solution_found: bool
    learning_enabled: bool
    conversation_ended: bool
    conversation_metadata: dict
    practice_item: str
    messages: Annotated[list, add_messages]
    pct_problem_skills_mastered: float
    item_associated_skills: list[
        Skill
    ]  # skills with id and descriptions - inferred real time for the item.
    learner_mastered_skills: list[Skill]
    mastered_problem_skills: list[Skill]
    learner_skill_paths: list[list[Skill]]
    learned_skills_ids: tuple[str]
    current_confusion: str
    tokens_used: dict[str, int]  # Track input and output tokens


class ConversationGraph(BaseModel):
    """A conversation on a practice item, between a learner and a tutor, teaching or assessing.
    Handles item-specific and context-specific logic.
    Logs dialogue, state, result, timestamps.
    """

    id: str
    skill_space: SkillSpace
    learner: FlexLearner
    base_model: str = "gpt-4.1-mini"
    initial_tutor_metadata: dict = Field(
        default_factory=dict,
    )  # like tutor helpfulness, tutor type etc.
    resolve_confusion_style: str | None = None
    practice_item: str | PracticeItem | None = None
    learning_enabled: bool = True
    max_turns: int = 10  # A turn is a learner-tutor exchange

    # path to graph memory sqlite db
    graph_memory_db_path: str | Path | None = None

    # Private attribute for OpenAI client (not included in model serialization)
    _client: OpenAI = PrivateAttr()

    # Allow arbitrary types for the OpenAI client
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        """Initialize ConversationGraph."""
        super().__init__(**data)

        # Load environment and initialize OpenAI client as private attribute
        load_dotenv()
        self._client = OpenAI()
        # graph sqlite memory
        self.get_memory_path()
        self.initialize_confusion_style()
        # instance logger
        self._logger = logging.LoggerAdapter(logger, {"conversation_id": self.id})

    def get_memory_path(self):
        """Get the path to the SQLite database for graph memory."""
        # build the sqlite db
        DB_CHECKPOINTS = "data/checkpoints"

        if not self.graph_memory_db_path:
            DB_CHECKPOINTS_path = Path(DB_CHECKPOINTS)
            DB_CHECKPOINTS_path.mkdir(parents=True, exist_ok=True)
            # use time to get timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.graph_memory_db_path = str(
                DB_CHECKPOINTS_path
                / f"{timestamp}_conversation_{self.id}_graph_memory.db",
            )

    def initialize_confusion_style(self):
        """Initialize the confusion resolution style based on tutor metadata."""
        if not self.resolve_confusion_style:
            self.resolve_confusion_style = """Continue the conversation by choosing one of the following talk moves:
        - Asking for more information: Confused, need help, request more information
        - Making a claim: make a math claim, factual or lists a step in your answer
        - Providing evidence or reasoning: Explain your thinking, provide evidence, talk about your reasoning
        """

    def print_graph(self, compiled_graph):
        output_dir = "docs/conversations/"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "graph.png")
        graph_png = compiled_graph.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(graph_png)

    def get_final_state(self, session_id: str) -> dict:
        """Retrieve the final state from the conversation checkpoint."""
        try:
            with SqliteSaver.from_conn_string(str(self.graph_memory_db_path)) as memory:
                compiled_graph = self.compile_graph(memory=memory)
                config = {"configurable": {"thread_id": session_id}}
                state_snapshot = compiled_graph.get_state(config)
                return state_snapshot.values
        except Exception:
            self._logger.exception("Error retrieving final state.")
            return {}

    def compile_graph(self, memory: Any = None) -> StateGraph:
        """Compile langgraph for the conversation."""
        graph_builder = StateGraph(ConversationState)

        ##################### NODES
        graph_builder.add_node("initialize_state", self.initialize_state)

        # compute skill gap between learner and problem, also return the shortest path to problem skills from mastered skills
        # graph_builder.add_node("compute_problem_skills_mastery", self.compute_problem_skills_mastery)
        # propose problem solution
        graph_builder.add_node(
            "propose_problem_solution",
            self.propose_problem_solution,
        )

        # teacher message node handling input from the tutor
        graph_builder.add_node(
            "tutor_problem_solution_response",
            self.tutor_problem_solution_response,
        )
        graph_builder.add_node("tutor_practice_response", self.tutor_practice_response)

        # Contextual Learner Follow-Up phase with persisting struggle until tutor resolves it
        graph_builder.add_node("practice_conversation", self.practice_conversation)

        # parallel with the problem solving inner vs outer loops: within exercise and exercise selection
        # Check whether tutor resolved confusion, this is customizable learner dialogue features, confusion resolution
        # # if yes, check if learner is ready for solution
        # # if not, continue conversation
        graph_builder.add_node(
            "check_tutor_resolved_confusion",
            self.check_tutor_resolved_confusion,
        )

        graph_builder.add_node(
            "learning_from_conversation",
            self.learning_from_conversation,
        )

        # conversation moderator checks check whether learner has enough knowledge to propose a solution (from prior knowledge or after practice)
        graph_builder.add_node(
            "check_learner_practiced_all_skills",
            self.check_learner_practiced_all_skills,
        )

        # final moderator, check whether the problem is solved
        graph_builder.add_node(
            "check_conversation_ended",
            self.check_conversation_ended,
        )

        # we stop the conversation here, the learning step happens at the learner level
        # using the conversation as content, plus the skills gap for possible skills to master.
        graph_builder.add_node("conversation_ended", self.conversation_ended)

        #################### EDGES
        graph_builder.add_edge(START, "initialize_state")

        # based on learner's problem skills mastery, decide next step:
        graph_builder.add_edge(
            "initialize_state",
            "check_learner_practiced_all_skills",
        )

        ### Direct problem solving path
        graph_builder.add_edge(
            "propose_problem_solution",
            "tutor_problem_solution_response",
        )
        graph_builder.add_edge(
            "tutor_problem_solution_response",
            "check_conversation_ended",
        )

        ### Further practice required
        graph_builder.add_edge("practice_conversation", "tutor_practice_response")

        # check tutor resolved confusion --> sends to check_learner_practiced_all_skills or to practice_conversation for further practice
        graph_builder.add_edge(
            "tutor_practice_response",
            "check_tutor_resolved_confusion",
        )

        graph_builder.add_edge("conversation_ended", END)

        compiled_graph = graph_builder.compile(checkpointer=memory)
        return compiled_graph

    def initialize_state(self, state: ConversationState) -> dict:
        """Initialize the conversation state with empty values for a new conversation."""
        try:
            assert (
                self.practice_item is not None
            ), "Practice item must be provided at the start of the conversation."

            # get the associated skills for the problem text
            if isinstance(self.practice_item, str):
                problem_skills = self.skill_space.choose_skills_for_item(
                    item_text=self.practice_item,
                    mode="single",
                )
                practice_item_text = self.practice_item
            elif isinstance(self.practice_item, PracticeItem):
                problem_skills = [
                    self.skill_space[sk_id]
                    for sk_id in self.practice_item.associated_skills
                ]
                practice_item_text = self.practice_item.text
            else:
                raise ValueError(
                    "practice_item must be either a string or a PracticeItem instance.",
                )

            mastered_skills = [
                self.skill_space[sk_id] for sk_id in self.learner.mastered_skills
            ]

            mastered_problem_skills = []
            # get skills intersection
            for ps in problem_skills:
                if ps in mastered_skills:
                    mastered_problem_skills.append(ps)
            pct_mastered = len(mastered_problem_skills) / len(problem_skills)

            learner_skill_paths = []
            if (
                pct_mastered < 1
            ):  # will need the learner's skill paths to mastery, for all problem skills which are not yet mastered
                for psk in problem_skills:
                    if psk not in mastered_skills:
                        # add a new skill path Mastered --> Skill 1 --> Skill ... --> Problem Skill
                        learner_skill_paths.append(
                            self.skill_space.get_prerequisite_path_to_skill_from_skill_group(
                                target_skill=psk,
                                skill_group=mastered_skills,
                            ),
                        )
            self._logger.debug(
                f"Initialized conversation state with {pct_mastered*100:.2f}% problem skills mastered."
                f"Item associated skills: {problem_skills},"
                f"Mastered skills: {mastered_skills},"
                f"Mastered problem skills: {mastered_problem_skills},"
                f"Learning paths to unmastered problem skills: {learner_skill_paths}",
            )

            return {
                "turn_number": 1,
                "solution_found": False,
                "learning_enabled": self.learning_enabled,
                "conversation_ended": False,
                "practice_item": practice_item_text,
                "conversation_metadata": {
                    "initial_tutor_metadata": self.initial_tutor_metadata,
                    "learning_enabled": self.learning_enabled,
                    "max_turns": self.max_turns,
                },
                "pct_problem_skills_mastered": pct_mastered,
                "item_associated_skills": problem_skills,
                "learner_mastered_skills": mastered_skills,
                "mastered_problem_skills": mastered_problem_skills,
                "learner_skill_paths": learner_skill_paths,
                "learned_skills_ids": (),
                "current_confusion": "",
                "tokens_used": {"input_tokens": 0, "output_tokens": 0},
            }
        except Exception:
            logger.exception("Error initializing state.")

    def check_learner_practiced_all_skills(
        self,
        state: ConversationState,
    ) -> Command[Literal["learning_from_conversation", "practice_conversation"]]:
        """Check whether the learner practiced all problem's skills, resolved all possible confusions about the problem, can proceed to providing a solution."""
        try:
            # if at the beginning of the conversation, learner masters all problem skills - propose solution
            if state.get("pct_problem_skills_mastered", 0) == 1:
                self._logger.debug(
                    "Learner has already mastered all problem skills at conversation start. Redirecting to learning_from_conversation.",
                )
                return Command(goto="learning_from_conversation")

            # if not, there exist problem skills unmastered
            # pass the mastered skills, the skill paths, and the conversation history to a judge
            # evaluate whether the learner has all the skills to proceed to solution
            mastered_problem_skills = state.get("mastered_problem_skills", [])
            learner_skill_paths = state.get("learner_skill_paths", [])
            # get rendered skill paths lines
            rendered_paths = self._render_skill_paths(learner_skill_paths)

            conversation_history = render_conversation_messages(
                state.get("messages", []),
                roles_names={"user": "Learner", "assistant": "Tutor"},
            )
            solution_readiness_prompt = f"""
            You are an experienced tutor teaching a novice math learner.
            Considering the mastered and unmastered learner skills, and its conversation with a tutor on a math problem, decide whether the learner has practiced the problem's skills enough to propose a meaningful solution to the problem.

            Math problem:
            <{state.get("practice_item", "")}>

            Skills related to the problem that the learner masters:
            <{mastered_problem_skills}>

            Remaining path to unmastered problem skills from mastered skills:
            <{rendered_paths}>

            Current conversation:
            <{conversation_history}>

            Important: Ensure the Learner has practiced the skills required in its responses and not only relied on the tutor's knowledge.
            Return your reasoning and your decision.
            """
            self._logger.debug(
                "Checking if learner is ready to propose solution, with prompt: %s",
                solution_readiness_prompt,
            )
            completion = self._client.beta.chat.completions.parse(
                model=self.base_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an experience math teacher who evaluates students' skill mastery levels.",
                    },
                    {"role": "user", "content": solution_readiness_prompt},
                ],
                response_format=SolutionReadinessResponse,
            )
            # Track tokens
            usage = completion.usage
            current_tokens = state.get(
                "tokens_used",
                {"input_tokens": 0, "output_tokens": 0},
            )
            self._logger.debug(
                f"[DEBUG - check_learner_practiced_all_skills] Current tokens before learner response: {current_tokens} - new tokens: {usage}",
            )
            updated_tokens = {
                "input_tokens": current_tokens["input_tokens"] + usage.prompt_tokens,
                "output_tokens": current_tokens["output_tokens"]
                + usage.completion_tokens,
            }

            # subtract 1 to the returned ids to match original learnable skills
            is_learner_ready = completion.choices[0].message.parsed.is_learner_ready
            reasoning = completion.choices[0].message.parsed.reasoning
            self._logger.debug(
                f"Check learner readiness: {reasoning}",
            )
            if is_learner_ready:
                return Command(
                    update={"tokens_used": updated_tokens},
                    goto="learning_from_conversation",
                )
            return Command(
                update={"tokens_used": updated_tokens},
                goto="practice_conversation",
            )
        except Exception:
            logger.exception("Error checking learner practicing all skills.")

    def _render_skill_paths(self, skill_paths: list[list[Skill]]) -> str:
        """Render skill paths for prompts."""
        rendered_paths = ""
        for sp in skill_paths:
            rendered_paths += (
                self.skill_space.render_skill_path(
                    skill_path=sp,
                    type="first_mastered_only",
                )
                + "\n"
            )
        return rendered_paths

    def learning_from_conversation(
        self,
        state: ConversationState,
        config: RunnableConfig,
    ) -> Command[Literal["propose_problem_solution", "conversation_ended"]]:
        """The learner should reach this node when ready to propose a final solution.

        - If the learner is ready to propose a solution (coming from check_learner_practiced_all_skills),
        then we assume the learner should master all learnable item skills with a probability P <= 1 (=1 for testing.)
        --> redirect to 'propose_problem_solution'

        - If the conversation reached max turns, mastery update is not required but possible, if the learner practiced part of
        the item's skills.
        --> redirect to 'conversation_ended'
        """
        try:
            should_respond_to_problem = (
                state.get("turn_number", 0) < self.max_turns
            )  # conversation is not ended, so learner is responding to problem
            if not state.get("learning_enabled", True):
                self._logger.debug(
                    f"[DEBUG - learning_from_conversation] Learning is disabled for this conversation. should respond to problem: {should_respond_to_problem}",
                )
                return Command(
                    goto=(
                        "propose_problem_solution"
                        if should_respond_to_problem
                        else "conversation_ended"
                    ),
                )
            # get current session id from config
            session_id = config["configurable"]["thread_id"]

            # Get correct answer from practice item if available
            correct_answer = ""
            if isinstance(self.practice_item, PracticeItem):
                correct_answer = self.practice_item.answer

            # save current learner mastered skills before learning:
            # mastered_skills is a list[str] of skill IDs on FlexLearner
            mastered_skills_before_learning = [
                sk.id if hasattr(sk, "id") else sk
                for sk in self.learner.mastered_skills
            ]

            # Learner learns from the conversation if needed:
            dialogue_history = render_conversation_messages(
                state.get("messages", []),
                roles_names={"user": "Learner", "assistant": "Tutor"},
            )
            # item_associated_skills may be Skill objects or dicts (after LangGraph state serialization)
            item_associated_skills = state.get("item_associated_skills", [])
            learned_skills = self.learner.learns_from_conversation(
                dialogue_history=dialogue_history,
                item_skills=item_associated_skills,
                llm_client=self._client,
                use_past_conversations=False,
                solved_problem=should_respond_to_problem,
                correct_answer=correct_answer,
            )
            # save learner practice history
            # learned_skills may be None if learns_from_conversation hit an internal error
            learned_skills = learned_skills or []
            mastered_skills_ids = [sk.id for sk in learned_skills]
            self.learner.log_new_practice(
                session_mastered_skills={
                    "session_id": session_id,
                    "mastered_skills_list": mastered_skills_ids,
                },
            )
            ### Do not save the learner's practice information here, done at the session level because not required for learning.
            # Here, only log the new practice to the learner's information.
            # Collect unique prerequisites across all item-associated skills
            _item_skill_prerequisites: list[str] = []
            _seen_prereq_ids: set[str] = set()
            for _sk in item_associated_skills:
                if hasattr(_sk, "prerequisites"):
                    _prereqs = _sk.prerequisites
                elif isinstance(_sk, dict):
                    _prereqs = _sk.get("prerequisites", [])
                else:
                    _prereqs = []
                for _pid in _prereqs:
                    if _pid not in _seen_prereq_ids:
                        _seen_prereq_ids.add(_pid)
                        _item_skill_prerequisites.append(_pid)

            # get learner details such as persona:
            learner_details = self.learner.persona

            conversation_record = {
                "session_id": session_id,
                "learner_details": learner_details,
                "practice_item_text": state.get("practice_item", ""),
                "dialogue_history": dialogue_history,
                "item_skills": [
                    (
                        sk.id
                        if hasattr(sk, "id")
                        else sk.get("id", "") if isinstance(sk, dict) else str(sk)
                    )
                    for sk in item_associated_skills
                ],
                "item_skill_prerequisites": _item_skill_prerequisites,
                # add previously mastered skills before the conversation and learned skills:
                "mastered_skills_before_conversation": mastered_skills_before_learning,
                "mastered_skills_from_conversation": mastered_skills_ids,
            }

            if correct_answer:
                conversation_record["correct_answer"] = correct_answer

            # save practice conversation:
            self.learner.save_practice_conversation(
                conversation_record=conversation_record,
            )

            learned_skills_ids_tuple = tuple([sk.id for sk in learned_skills])

            if should_respond_to_problem:
                return Command(
                    update={"learned_skills_ids": learned_skills_ids_tuple},
                    goto="propose_problem_solution",
                )
            return Command(
                update={"learned_skills_ids": learned_skills_ids_tuple},
                goto="conversation_ended",
            )
        except Exception:
            logger.exception("Error in learning from conversation.")

    def propose_problem_solution(self, state: ConversationState):
        """The learner should reach this node when ready to propose a final solution.
        Use the practice item problem and the learner's mastery state to formulate an answer.
        """
        try:
            # mastered_problem_skills = state.get("mastered_problem_skills", [])
            learner_skill_paths = state.get("learner_skill_paths", [])
            rendered_paths = self._render_skill_paths(learner_skill_paths)

            conversation_history = render_conversation_messages(
                state.get("messages", []),
                roles_names={"user": "You", "assistant": "Tutor"},
            )

            # ensure item skills can be set invisible to the final solution prompt for custom knowledge configurations
            item_skills = state.get("item_associated_skills", [])
            practice_item_text = state.get("practice_item", "")

            # Delegate prompt generation to the learner
            prompts = self.learner.get_solution_prompt(
                practice_item_text=practice_item_text,
                item_skills=item_skills,
                conversation_history=conversation_history,
                knowledge_attrs={"skill_paths_rendered": rendered_paths},
            )

            completion = self._client.beta.chat.completions.parse(
                model=self.base_model,
                messages=[
                    {
                        "role": "system",
                        "content": prompts["system"],
                    },
                    {"role": "user", "content": prompts["user"]},
                ],
                response_format=SolutionProposalResponse,
            )
            # Track tokens
            usage = completion.usage
            current_tokens = state.get(
                "tokens_used",
                {"input_tokens": 0, "output_tokens": 0},
            )
            self._logger.debug(
                f"[DEBUG - propose_problem_solution] Current tokens before learner response: {current_tokens} - new tokens: {usage}",
            )
            updated_tokens = {
                "input_tokens": current_tokens["input_tokens"] + usage.prompt_tokens,
                "output_tokens": current_tokens["output_tokens"]
                + usage.completion_tokens,
            }

            # subtract 1 to the returned ids to match original learnable skills
            learner_response = completion.choices[0].message.parsed.response
            # stream learner response
            writer = get_stream_writer()
            writer({"learner": learner_response})
            return {
                "messages": [{"role": "user", "content": learner_response}],
                "tokens_used": updated_tokens,
            }
        except Exception:
            logger.exception("Error when learner proposed a solution.")

    def tutor_problem_solution_response(self, state: ConversationState):
        """Wait for tutor's response during problem solution and log it."""
        # use Command when running a conversation to stop the graph execution here and resume with tutor input
        teacher_message = interrupt("placeholder_tutor_input")
        tn = state.get("turn_number")
        tn += 1

        self._logger.debug(
            f"[DEBUG] In tutor_problem_solution_response. Turn number: {tn}. Tutor message: {teacher_message}",
        )

        return {
            "messages": [{"role": "assistant", "content": teacher_message}],
            "turn_number": tn,
        }

    def tutor_practice_response(self, state: ConversationState):
        """Wait for tutor's response during practice and log it."""
        self._logger.info("Waiting for tutor's response.")

        # use Command when running a conversation to stop the graph execution here and resume with tutor input
        teacher_message = interrupt("placeholder_tutor_input")
        tn = state.get("turn_number")
        tn += 1

        self._logger.debug(
            f"[DEBUG] In tutor practice response. Turn number: {tn}. Tutor message: {teacher_message}",
        )

        return {
            "messages": [{"role": "assistant", "content": teacher_message}],
            "turn_number": tn,
        }

    def practice_conversation(self, state: ConversationState):
        """Depending on its current mastery level, the learner should struggle and follow-up on the tutor's response.
        Start or continue the confusion step until the tutor resolved it.

        Delegate prompt generation to the learner's get_practice_prompt method
        """
        try:
            self._logger.debug("[DEBUG] Starting practice conversation step.")
            # mastered_problem_skills = state.get("mastered_problem_skills", [])
            learner_skill_paths = state.get("learner_skill_paths", [])
            rendered_paths = self._render_skill_paths(learner_skill_paths)

            conversation_history = render_conversation_messages(
                state.get("messages", []),
                roles_names={"user": "You", "assistant": "Tutor"},
            )
            # Get confusion state and response mode
            current_confusion = state.get("current_confusion", "")
            current_tokens = state.get(
                "tokens_used",
                {"input_tokens": 0, "output_tokens": 0},
            )

            # ensure item skills can be set invisible to the practice prompt for custom knowledge configurations
            item_skills = state.get("item_associated_skills", [])
            practice_item_text = state.get("practice_item", "")

            # do we really need to pass item skills here?

            # Delegate prompt generation to the learner
            prompts = self.learner.get_practice_prompt(
                practice_item_text=practice_item_text,
                item_skills=item_skills,
                conversation_history=conversation_history,
                current_confusion=current_confusion,
                knowledge_attrs={"skill_paths_rendered": rendered_paths},
            )

            if current_confusion == "":
                self._logger.debug(
                    "Starting new confusion - no current confusion state",
                )
                completion = self._client.beta.chat.completions.parse(
                    model=self.base_model,
                    messages=[
                        {
                            "role": "system",
                            "content": prompts["system"],
                        },
                        {"role": "user", "content": prompts["user"]},
                    ],
                    response_format=StartConfusionResponse,
                )
                # Track tokens
                usage = completion.usage
                self._logger.debug(
                    f"[DEBUG - practice_conversation] Current tokens before learner response: {current_tokens} - new tokens: {usage}",
                )
                updated_tokens = {
                    "input_tokens": current_tokens["input_tokens"]
                    + usage.prompt_tokens,
                    "output_tokens": current_tokens["output_tokens"]
                    + usage.completion_tokens,
                }

                # subtract 1 to the returned ids to match original learnable skills
                learner_current_confusion = completion.choices[
                    0
                ].message.parsed.current_confusion
                learner_response = completion.choices[0].message.parsed.response

                self._logger.debug(
                    f"Started new confusion: {learner_current_confusion}",
                )

                # stream learner response only if in a runnable context
                try:
                    writer = get_stream_writer()
                    writer({"learner": learner_response})
                except Exception as e:
                    self._logger.debug(
                        f"No stream writer available in the current context: {e}. Ignore error if running only this node.",
                    )

                return {
                    "messages": [{"role": "user", "content": learner_response}],
                    "current_confusion": learner_current_confusion,
                    "tokens_used": updated_tokens,
                }

            # continue confusion step
            self._logger.debug(
                f"Continuing existing confusion: {current_confusion}",
            )
            completion = self._client.beta.chat.completions.parse(
                model=self.base_model,
                messages=[
                    {
                        "role": "system",
                        "content": prompts["system"],
                    },
                    {"role": "user", "content": prompts["user"]},
                ],
                response_format=ContinueConfusionResponse,
            )
            # Track tokens
            usage = completion.usage
            updated_tokens = {
                "input_tokens": current_tokens["input_tokens"] + usage.prompt_tokens,
                "output_tokens": current_tokens["output_tokens"]
                + usage.completion_tokens,
            }

            # subtract 1 to the returned ids to match original learnable skills
            learner_response = completion.choices[0].message.parsed.response

            # stream learner response
            writer = get_stream_writer()
            writer({"learner": learner_response})
            return {
                "messages": [{"role": "user", "content": learner_response}],
                "tokens_used": updated_tokens,
            }
        except Exception:
            logger.exception("Error when practicing conversation.")

    def check_tutor_resolved_confusion(
        self,
        state: ConversationState,
    ) -> Command[
        Literal[
            "check_learner_practiced_all_skills",
            "practice_conversation",
            "learning_from_conversation",
        ]
    ]:
        """Check whether the tutor's set of responses resolved the learner's confusion.
        If yes, send to check_learner_practiced_all_skills to move on to next confusion.
        If not yet, continue practice conversation.
        """
        self._logger.info(
            "[DEBUG] Checking if tutor resolved learner's confusion: turn number %d",
            state.get("turn_number", 0),
        )
        if state.get("turn_number", 0) >= self.max_turns:
            self._logger.info(
                "Conversation reached maximum number of turns. Ending conversation.",
            )
            return Command(
                goto="learning_from_conversation",
            )
        current_confusion = state.get("current_confusion", "")
        if current_confusion == "":
            # no confusion to check
            self._logger.info("No current confusion to check for resolution.")
            return Command(
                goto="check_learner_practiced_all_skills",
            )
        conversation_history = render_conversation_messages(
            state.get("messages", []),
            roles_names={"user": "Learner", "assistant": "Tutor"},
        )
        evaluate_resolved_confusion_prompt = f"""
        You are an expert math teacher evaluating another math tutor teaching a novice math learner with problem solving.
        Given the conversation, the math problem and the learner's confusion, evaluate whether the tutor resolved the learner's confusion with its last messages.
        Math problem:
        {state.get("practice_item", "")}

        Learner confusion:
        {current_confusion}.

        Practice conversation:
        {conversation_history}

        Return your reasoning and your evaluation yes/no: the tutor resolved the learner's confusion.
        """
        self._logger.debug(
            "Checking if tutor resolved learner's confusion, with prompt: %s",
            evaluate_resolved_confusion_prompt,
        )
        completion = self._client.beta.chat.completions.parse(
            model=self.base_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert math tutor evaluating another tutor's teaching effectiveness.",
                },
                {"role": "user", "content": evaluate_resolved_confusion_prompt},
            ],
            response_format=ResolvedConfusionResponse,
        )
        # Track tokens
        usage = completion.usage
        current_tokens = state.get(
            "tokens_used",
            {"input_tokens": 0, "output_tokens": 0},
        )
        self._logger.debug(
            f"[DEBUG - check_resolved_confusion] Current tokens before learner response: {current_tokens} - new tokens: {usage}",
        )
        updated_tokens = {
            "input_tokens": current_tokens["input_tokens"] + usage.prompt_tokens,
            "output_tokens": current_tokens["output_tokens"] + usage.completion_tokens,
        }

        # subtract 1 to the returned ids to match original learnable skills
        resolved_confusion = completion.choices[0].message.parsed.resolved_confusion
        self._logger.info(f"Tutor resolved confusion: {resolved_confusion}")
        if resolved_confusion:
            # current confusion is set back to empty
            return Command(
                update={"current_confusion": "", "tokens_used": updated_tokens},
                goto="check_learner_practiced_all_skills",
            )
        # Use Command to send to either update the state or send to next check
        return Command(
            update={"tokens_used": updated_tokens},
            goto="practice_conversation",
        )

    def check_conversation_ended(
        self,
        state: ConversationState,
    ) -> Command[Literal["conversation_ended", "propose_problem_solution"]]:
        """Check whether the conversation should end.
        It can end because a solution was found or because the conversation reached a natural end.
        """
        try:
            conversation_history = render_conversation_messages(
                state.get("messages", []),
                roles_names={"user": "Learner", "assistant": "Tutor"},
            )
            conversation_should_end_prompt = f"""
            You are a math tutor with pedagogical knowledge in math learning.
            Evaluate whether the conversation below should end.
            It should end if the learner in the conversation has found a solution to the math problem. Only focus on the learner's responses to decide this (and ignore tutor's responses).

            Math problem:
            {state.get("practice_item", "")}

            Practice conversation:
            {conversation_history}

            Return your reasoning and your decision.
            """
            completion = self._client.beta.chat.completions.parse(
                model=self.base_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert math tutor evaluating student learning in conversations.",
                    },
                    {"role": "user", "content": conversation_should_end_prompt},
                ],
                response_format=ConversationEndingResponse,
            )
            # Track tokens
            usage = completion.usage
            current_tokens = state.get(
                "tokens_used",
                {"input_tokens": 0, "output_tokens": 0},
            )
            self._logger.debug(
                f"[DEBUG - check_conversation_ended] Current tokens before learner response: {current_tokens} - new tokens: {usage}",
            )
            updated_tokens = {
                "input_tokens": current_tokens["input_tokens"] + usage.prompt_tokens,
                "output_tokens": current_tokens["output_tokens"]
                + usage.completion_tokens,
            }

            conversation_should_end = completion.choices[
                0
            ].message.parsed.conversation_should_end
            if conversation_should_end or (
                state.get("turn_number", 0) >= self.max_turns + 5
            ):  # allowing 5 turns to propose problem solution after max turns
                return Command(
                    update={"tokens_used": updated_tokens},
                    goto="conversation_ended",
                )
            self._logger.debug(
                f"[DEBUG] In check_conversation_ended. Conversation should continue. Turn number: {state.get('turn_number', 0)}",
            )
            return Command(
                update={"tokens_used": updated_tokens},
                goto="propose_problem_solution",
            )
        except Exception:
            logger.exception("Error checking conversation end.")

    def conversation_ended(self, state: ConversationState):
        """Handle conversation end"""
        system_message = f"Conversation ended at turn {state['turn_number']}."

        learned_skills_ids = state.get("learned_skills_ids", ())

        if self.learning_enabled:
            system_message += (
                f" Learner mastered new skills: {','.join(learned_skills_ids)}"
            )
        else:
            system_message += " Learning disabled for this conversation."

        writer = get_stream_writer()
        writer({"system": system_message})

        end_message = {
            "role": "system",
            "content": system_message,
        }

        return {"messages": [end_message], "conversation_ended": True}

    # run a conversation endpoint
    # loads the session's graph from memory, and either start or resume a conversation from the session.
    # The session handling happens before run_conversation
    def run_conversation(
        self,
        session_id: str,
        start_or_resume_conversation: Literal["start", "resume"],
        tutor_message: str = "",
        **kwargs,
    ):
        """Run an example conversation with sqlite memory."""
        try:
            with SqliteSaver.from_conn_string(str(self.graph_memory_db_path)) as memory:
                compiled_graph = self.compile_graph(memory=memory)
                # LangGraph required config for resuming conversations
                config = {
                    "configurable": {"thread_id": session_id},
                    "recursion_limit": 150,
                }
                if start_or_resume_conversation == "start":
                    command = {}  # initial state modifications if needed
                else:  # resume a conversation
                    self._logger.info(
                        f"Resuming conversation for session {session_id} with tutor message: {tutor_message}",
                    )
                    command = Command(resume=tutor_message)

                # stream conversation with command (either start or resume)
                # TODO what happens if tries to resume a non-existing session?
                for chunk in compiled_graph.stream(
                    command,
                    config,
                    stream_mode="custom",
                ):
                    # get learner messages output
                    if chunk.get("learner", None):  # receive learner message
                        yield chunk["learner"]
                    if chunk.get("system", None):
                        system_info = chunk["system"]
                        yield system_info
                        if "Conversation ended at turn" in system_info:
                            return

        except Exception:
            logger.exception("Error running conversation.")
            raise
