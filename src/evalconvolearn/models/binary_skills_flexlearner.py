import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Self

import pandas as pd
from pydantic import BaseModel, model_validator

from ..core.flexlearner import FlexLearner
from .practice_item import PracticeItem
from .skill import Skill, SkillSpace

logger = logging.getLogger(__name__)


class BinarySkillsFlexLearner(FlexLearner):
    """Default skill-binary learner implementation.

    Knowledge state is represented by a list of mastered skill IDs.
    This is the original learner that uses binary skill mastery.
    """

    def __init__(self, **data: Any) -> None:
        """Initialize learner data."""
        super().__init__(**data)

    # ------------------------------------------------------------------ #
    #  FlexLearner abstract method implementations
    # ------------------------------------------------------------------ #

    def get_knowledge_description(self) -> str:
        """Return mastered skill descriptions as the knowledge representation."""
        if not self.mastered_skills:
            return "No skills mastered yet."
        descriptions = []
        for sk_id in self.mastered_skills:
            skill = self.skill_space[sk_id]
            descriptions.append(f"- {skill.id}: {skill.description}")
        return "\n".join(descriptions)

    def get_knowledge_for_problem(
        self,
        practice_item: str | PracticeItem,
        item_skills: list[Skill],
        knowledge_attrs: dict | None = None,
    ) -> str:
        """Return mastered skills relevant to the problem."""
        mastered_problem_skills = [sk for sk in item_skills if sk.id in self.mastered_skills]
        if not mastered_problem_skills:
            return "You have not mastered any of the skills required for this problem."
        lines = [f"- {sk.id}: {sk.description}" for sk in mastered_problem_skills]

        skill_paths_rendered = knowledge_attrs.get("skill_paths_rendered", "") if knowledge_attrs else ""
        knowledge_gaps = (
            f"Knowledge gaps (skills you still need to learn):\n{skill_paths_rendered}"
            if skill_paths_rendered
            else "Knowledge gaps (skills you still need to learn): None identified yet."
        )

        return "Skills you have mastered that are relevant:\n" + "\n".join(lines) + "\n" + knowledge_gaps

    def get_required_knowledge_to_answer_practice_item(
        self,
        practice_item: str | PracticeItem,
        practice_item_skills: list[Skill],
        knowledge_attrs: dict | None = None,
    ) -> str:
        """Return the required skills to answer the problem correctly."""
        mastered_skills_ids = (
            "\n".join(
                [f"- {skill_id}" for skill_id in self.mastered_skills],
            )
            if self.mastered_skills
            else "None"
        )
        associated_skills_ids = (
            "\n".join(
                [f"- {skill.id}" for skill in practice_item_skills],
            )
            if practice_item_skills
            else "None provided"
        )
        required_knowledge_text = f"""
        Your ONLY mastered skills (by assigned ID) are:
        {mastered_skills_ids}

        You must have mastered the following skills (by assigned ID) to answer this question:
        {associated_skills_ids}
        """
        return required_knowledge_text

    def update_knowledge_from_conversation(
        self,
        dialogue_history: str,
    ) -> None:
        """For the default skill-binary learner, knowledge update is just
        mastering skills (already done before this call). No-op here.
        """

    def initialize_learner_knowledge(self, *args: Any, **kwargs: Any) -> None:
        """No need to extend this for the default skill-binary learner,
        but this can be overridden in custom learners.
        """

    def initialize_from_skills(self, mastered_skill_ids: list[str], **kwargs: Any) -> None:
        """Initialize the learner's knowledge state from a list of skill IDs."""
        raise NotImplementedError(
            "This method is not implemented for the Learner, because it already "
            "handles skill initialization through the mastered_skills attribute.",
        )

    # ------------------------------------------------------------------ #
    #  Pydantic validators
    # ------------------------------------------------------------------ #

    @model_validator(mode="after")
    def validate_unique_skills_and_skills_are_in_space(self) -> Self:
        if not self.practice_conversations_file:
            raise ValueError(
                "practice_conversations_file must be provided for Learner.",
            )

        if not Path(self.practice_conversations_file).exists():
            try:
                Path(self.practice_conversations_file).parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                Path(self.practice_conversations_file).touch()
            except Exception as e:
                raise ValueError(
                    f"Could not create practice_conversations_file at {self.practice_conversations_file}: {e}",
                )

        if len(self.mastered_skills) != len(set(self.mastered_skills)):
            for sid in self.mastered_skills:
                if self.mastered_skills.count(sid) > 1:
                    raise ValueError(
                        f"One or more duplicate skill IDs found in Learner skills, e.g: {sid}",
                    )

        for sid in self.mastered_skills:
            assert sid in self.skill_space, (
                f"Skill with id {sid} in Learner skills is not part of the defined SkillSpace."
            )

        if (len(self.mastered_skills) > 0) and (len(self.practice_history) == 0):
            self.log_new_practice(
                {
                    "session_id": "initialization",
                    "mastered_skills_list": self.mastered_skills.copy(),
                },
            )

        practiced_skills = set()
        for session in self.practice_history:
            practiced_skills.update(session.get("mastered_skills_list", []))
        if set(self.mastered_skills) != practiced_skills:
            raise ValueError(
                f"Mismatch between mastered skills and practiced skills in history for Learner {self.id}.",
            )

        return self


class StudentPool(BaseModel):
    """A student_pool is a set of unique learners.
    Learners can be added, retrieved and deleted.
    A student_pool may have multiple SkillSpaces to choose from for its learners.
    """

    id: str
    learner_class: type = BinarySkillsFlexLearner  # Default learner class for this student pool
    learners: list[FlexLearner] = []  # Accepts any FlexLearner subclass
    skill_spaces: list[SkillSpace] = []
    unique_skills: list[Skill] = []
    base_directory: str | Path = "data/student_pools/"
    directory_file: str | Path = ""
    practice_conversations_file: str | Path = ""

    def __init__(self, **data: Any) -> None:
        """Initialize student_pool data."""
        super().__init__(**data)

        if not self.directory_file:
            current_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.directory_file = Path(self.base_directory) / f"{self.id}_{current_timestamp}"
            self.directory_file.mkdir(parents=True, exist_ok=True)
        self.directory_file = Path(self.directory_file)

        if not self.practice_conversations_file:
            self.practice_conversations_file = self.directory_file / "all_conversations.jsonl"
            self.practice_conversations_file.parent.mkdir(parents=True, exist_ok=True)
            if not self.practice_conversations_file.exists():
                self.practice_conversations_file.touch()
        self.practice_conversations_file = Path(self.practice_conversations_file)

        self.skill_spaces = []
        for learner in self.learners:
            if learner.skill_space not in self.skill_spaces:
                self.skill_spaces.append(learner.skill_space)

        self.unique_skills = []
        for skill_space in self.skill_spaces:
            for skill in skill_space:
                if skill not in self.unique_skills:
                    self.unique_skills.append(skill)

    def __len__(self) -> int:
        return len(self.learners)

    def get_learner(self, learner_id: str) -> FlexLearner | None:
        for learner in self.learners:
            if learner.id == learner_id:
                return learner
        return None

    def __getitem__(self, learner_id: str) -> FlexLearner:
        learner = self.get_learner(learner_id)
        if not learner:
            raise KeyError(f"Learner with id {learner_id} not found.")
        return learner

    def add_learner(self, learner: FlexLearner) -> None:
        if any(learner.id == lea.id for lea in self.learners):
            raise ValueError(f"Learner with id {learner.id} already exists.")
        self.learners.append(learner)

    def create_learner(
        self,
        learner_id: str,
        mastered_skills: list[str],
        skill_space: SkillSpace,
        **kwargs: Any,
    ) -> FlexLearner:
        if any(learner_id == lea.id for lea in self.learners):
            raise ValueError(f"Learner with id {learner_id} already exists.")

        new_learner = self.learner_class(
            id=learner_id,
            skill_space=skill_space,
            mastered_skills=mastered_skills,
            practice_history=[],
            practice_conversations_file=self.practice_conversations_file,
            **kwargs,
        )
        # initialize custom extra knowledge configuration.
        # For now, this function should consider the current learner's mastered skills
        # and initialize the learner knowledge accordingly.
        # This allows the benchmarks to define default skills and the custom learner adapts to them
        new_learner.initialize_learner_knowledge(**kwargs)
        self.learners.append(new_learner)
        return new_learner

    def remove_learner(self, learner_id: str) -> None:
        self.learners = [learner for learner in self.learners if learner.id != learner_id]

    def _check_unique_learners(self) -> None:
        learner_ids = [learner.id for learner in self.learners]
        if len(learner_ids) != len(set(learner_ids)):
            for lid in learner_ids:
                if learner_ids.count(lid) > 1:
                    raise ValueError(
                        f"One or more duplicate learner IDs found in student_pool, e.g: {lid}",
                    )

    @model_validator(mode="after")
    def validate_unique_learners(self) -> Self:
        self._check_unique_learners()
        return self

    def get_number_of_skillspaces(self) -> int:
        return len(self.skill_spaces)

    def get_number_of_unique_skills(self) -> int:
        return len(self.unique_skills)

    def load_student_pool_from_csv(
        self,
        file_path: str | Path,
        skill_space: SkillSpace,
    ) -> None:
        try:
            if Path(file_path).exists():
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    learner_id = str(row["learner_id"])
                    session_id = row["session_id"]
                    mastered_skills_list = (
                        row["mastered_skills_list"].split(",") if pd.notna(row["mastered_skills_list"]) else []
                    )
                    learner = self.get_learner(learner_id)
                    if not learner:
                        self.create_learner(
                            learner_id=learner_id,
                            mastered_skills=[],
                            skill_space=skill_space,
                        )
                        learner = self.get_learner(learner_id)
                    assert learner is not None, f"Learner {learner_id} not found after creation"
                    for sk in mastered_skills_list:
                        learner.master_new_skill(sk)
                    learner.log_new_practice(
                        {
                            "session_id": session_id,
                            "mastered_skills_list": mastered_skills_list,
                        },
                    )
                self._check_unique_learners()
            else:
                pd.DataFrame(
                    columns=pd.Index(["learner_id", "session_id", "mastered_skills_list"]),
                ).to_csv(file_path, index=False)

        except Exception:
            logger.exception("Error when loading existing student_pool from csv.")
            raise

    def save_student_pool_practice_history_to_csv(self, file_path: str | Path) -> None:
        try:
            records = []
            for learner in self.learners:
                for practice in learner.practice_history:
                    records.append(
                        {
                            "learner_id": learner.id,
                            "session_id": practice["session_id"],
                            "mastered_skills_list": ",".join(
                                practice["mastered_skills_list"],
                            ),
                        },
                    )
            df = pd.DataFrame(records)
            df.to_csv(file_path, index=False)
        except Exception:
            logger.exception("Error when saving student_pool practice history to csv.")
            raise

    def populate_new_student_pool(
        self,
        n_learners: int,
        skill_space: SkillSpace,
        **kwargs: Any,
    ) -> None:
        for learner_id in range(n_learners):
            new_learner = self.learner_class(
                id=learner_id,
                skill_space=skill_space,
                mastered_skills=[],
                practice_history=[],
                practice_conversations_file=self.practice_conversations_file,
                **kwargs,
            )
            space_root_skills = new_learner.learn_root_skills()

            # TODO- should we add a method to set up custom initial knowledge configuration?
            # or do we consider this to be handled manually by the user's custom code?

            # initialize custom extra knowledge configuration.
            new_learner.initialize_learner_knowledge()

            new_learner.log_new_practice(
                {
                    "session_id": "student_pool_init",
                    "mastered_skills_list": [sk.id for sk in space_root_skills],
                },
            )
            self.add_learner(new_learner)
