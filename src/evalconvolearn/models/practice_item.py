import csv
import json
import random
from pathlib import Path
from typing import Self

from pydantic import BaseModel, model_validator

from ..models.skill import Skill, SkillSpace


class PracticeItem(BaseModel):
    """A practice item defines a problem to be practiced by a learner.
    It is associated with one or more skills for teaching or assessment in conversations.
    """

    text: str  # unique text description of the practice item
    associated_skills: list[str] = []  # list of associated skill IDs that this item teaches or assesses
    answer: str = ""  # unique text description of the problem's answer
    incorrect_answers: list[str] = []  # list of incorrect answer choices for multiple choice

    @model_validator(mode="after")
    def validate_unique_associated_skills(self) -> Self:
        if len(self.associated_skills) != len(set(self.associated_skills)):
            # find a duplicate associated skill to inform the error
            for sid in self.associated_skills:
                if self.associated_skills.count(sid) > 1:
                    raise ValueError(
                        f"One or more duplicate associated skill IDs found in PracticeItem '{self.text}', e.g: {sid}",
                    )
        return self

    def __eq__(self, other: object) -> bool:
        # two practice items are equal if they have the same text
        if not isinstance(other, PracticeItem):
            return NotImplemented
        return self.text == other.text

    def add_associated_skill(self, skill_id: str) -> None:
        if skill_id in self.associated_skills:
            raise ValueError(
                f"Associated skill with id {skill_id} already exists for PracticeItem {self.id}.",
            )
        self.associated_skills.append(skill_id)

    def get_answer(self) -> str:
        """Return the answer for this practice item."""
        return self.answer

    def get_all_choices(self, shuffle: bool = True) -> list[str]:
        """Return all answer choices (correct + incorrect).

        Args:
        ----
            shuffle: Whether to randomize the order of choices (default: True)

        Returns:
        -------
            List of all answer choices

        """
        choices = [self.answer] + self.incorrect_answers
        if shuffle:
            random.shuffle(choices)
        return choices

    def get_correct_choice_index(self, choices: list[str]) -> int:
        """Get the index of the correct answer in a list of choices.

        Args:
        ----
            choices: List of answer choices

        Returns:
        -------
            Index of the correct answer, or -1 if not found

        """
        try:
            return choices.index(self.answer)
        except ValueError:
            return -1


class PracticeItemPool(BaseModel):
    """A collection of practice items available in the simulation for teaching and assessment.
    It is associated with a skill space defining the skills used by the items.
    All skill operations are available via the associated skill space, for any practice item.
    """

    items: list[PracticeItem] = []  # items should be unique.
    skill_space: SkillSpace

    def get_item_by_text(self, item_text: str) -> PracticeItem:
        for item in self.items:
            if item.text == item_text:
                return item
        raise ValueError(
            f"PracticeItem with text '{item_text}' not found in PracticeItemPool.",
        )

    def __getitem__(self, item_text: str) -> PracticeItem:
        return self.get_item_by_text(item_text)

    def __contains__(self, item_object: str | PracticeItem) -> bool:
        if isinstance(item_object, str):
            try:
                item_object = self.get_item_by_text(item_object)
            except ValueError:
                return False
        return item_object in self.items

    def __len__(self) -> int:
        return len(self.items)

    def _check_unique_items_and_associated_skills(self) -> None:
        item_texts = [item.text for item in self.items]
        if len(item_texts) != len(set(item_texts)):
            # find a duplicate item to inform the error
            for text in item_texts:
                if item_texts.count(text) > 1:
                    raise ValueError(
                        f"One or more duplicate practice item texts found in PracticeItemPool, e.g: {text}",
                    )

        # validate that all associated skills exist in the skill space
        for item in self.items:
            for sid in item.associated_skills:
                if sid not in self.skill_space:
                    raise ValueError(
                        f"Associated skill ID {sid} in PracticeItem '{item.text}' not found in SkillSpace.",
                    )

    @model_validator(mode="after")
    def validate_unique_items_and_associated_skills(self) -> Self:
        self._check_unique_items_and_associated_skills()
        return self

    # add a new practice item to the pool
    def add_item(self, item: PracticeItem) -> None:
        if item in self.items:
            raise ValueError(
                f"PracticeItem with text '{item.text}' already exists in PracticeItemPool.",
            )
        # check that all associated skills exist in the skill space
        for sid in item.associated_skills:
            if sid not in self.skill_space:
                raise ValueError(
                    f"Associated skill ID {sid} in PracticeItem '{item.text}' not found in SkillSpace.",
                )
        self.items.append(item)

    def _validate_item_object(self, item_object: str | PracticeItem) -> PracticeItem:
        """Validate and return PracticeItem from item_object."""
        if isinstance(item_object, str):
            item = self.get_item_by_text(item_object)
        elif isinstance(item_object, PracticeItem):
            item = item_object
        else:
            raise ValueError(
                "item_object must be either a PracticeItem instance or a string representing the item text.",
            )
        return item

    # delete a practice item from the pool
    def remove_item(self, item_object: str | PracticeItem) -> None:
        item = self._validate_item_object(item_object)
        self.items.remove(item)

    # get a list of items associated with a given skill ID
    def get_items_with_unique_skill(self, skill: str | Skill) -> list[PracticeItem]:
        """Get all practice items associated only with a given skill ID."""
        skill_id = skill if isinstance(skill, str) else skill.id
        associated_items = []
        for item in self.items:
            # matches only items with that unique skill.
            if item.associated_skills == [skill_id]:
                associated_items.append(item)
        return associated_items

    def get_items_having_skill(self, skill: str | Skill) -> list[PracticeItem]:
        """Get all practice items associated with a given skill ID, including those with multiple associated skills."""
        skill_id = skill if isinstance(skill, str) else skill.id
        associated_items = []
        for item in self.items:
            # matches items that have that skill among their associated skills, even if they have other skills as well.
            if skill_id in item.associated_skills:
                associated_items.append(item)
        return associated_items

    ### Practice Item <> Skill operations via SkillSpace ###
    def get_item_associated_skills(
        self,
        item_object: str | PracticeItem,
    ) -> list[Skill]:
        """Get all associated skills for a given practice item."""
        item = self._validate_item_object(item_object)

        associated_skills = []
        for skill_id in item.associated_skills:
            skill = self.skill_space[skill_id]
            if skill:
                associated_skills.append(skill)

        return associated_skills

    # direct prerequisites of the associated skills at the level -1 of the skill graph
    def get_item_direct_prerequisite_skills(
        self,
        item_object: str | PracticeItem,
    ) -> list[Skill]:
        """Get all prerequisite skills for a given practice item."""
        item = self._validate_item_object(item_object)

        prerequisite_skills = set()
        for skill_id in item.associated_skills:
            skill = self.skill_space[skill_id]
            if skill:
                prerequisite_skills.update(
                    self.skill_space.get_prerequisite_skills(skill),
                )

        return list(prerequisite_skills)

    # get all prerequisites recursively for the associated skills
    def get_item_all_prerequisite_skills(
        self,
        item_object: str | PracticeItem,
    ) -> list[str]:
        """Get all prerequisite skills for a given practice item."""
        item = self._validate_item_object(item_object)

        all_prerequisite_skills = set()
        for skill_id in item.associated_skills:
            skill = self.skill_space[skill_id]
            if skill and skill.id:
                all_prerequisite_skills.update(
                    self.skill_space.get_all_prerequisites(
                        skill.id,
                        return_as_ids=True,
                    ),
                )

        return list(all_prerequisite_skills)

    def load_items_from_json(self, file_path: str) -> None:
        """Load practice items from a JSON with format:
        [{
            "problem": "text of the practice item",
            "answer": "text of the practice item answer",
            "incorrect_answers": ["wrong1", "wrong2", "wrong3"],  # Optional
            "skill_id": ["skill_id_1", "skill_id_2", ...]
        }]

        The "incorrect_answers" field is optional. If provided, it should be a list of
        incorrect answer choices for multiple choice questions.
        """
        with open(file_path) as f:
            data = json.load(f)

        for item_data in data:
            # Get incorrect answers if provided
            incorrect_answers = item_data.get("incorrect_answers", [])

            item = PracticeItem(
                text=item_data["problem"],
                answer=item_data.get("answer", ""),
                associated_skills=item_data.get("skill_id", []),
                incorrect_answers=incorrect_answers,
            )
            self.add_item(item)

        self._check_unique_items_and_associated_skills()

    def load_items_from_csv(self, file_path: str | Path) -> None:
        """Load practice items from a CSV with format:
        problem,answer,incorrect_answer_1,incorrect_answer_2,incorrect_answer_3,skill_id
        "text of the practice item","correct answer","wrong1","wrong2","wrong3","skill_id_1, skill_id_2"

        The incorrect_answer_1, incorrect_answer_2, and incorrect_answer_3 columns are optional.
        If present, they will be loaded as multiple choice distractors. Any combination of these
        columns can be present (e.g., just 1 and 2, or all three, or none).
        """
        with open(file_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                skill_ids = [s.strip() for s in row["skill_id"].split(",")]
                # ensure no trailing or leading whitespaces, or quotes of any kind in row problem:
                problem_text = row["problem"].strip().strip('"').strip("'")
                answer_text = (row.get("answer") or "").strip().strip('"').strip("'")

                # Collect incorrect answers from optional columns
                incorrect_answers = []
                for i in range(1, 4):  # Check for incorrect_answer_1, 2, and 3
                    col_name = f"incorrect_answer_{i}"
                    if row.get(col_name):
                        incorrect_ans = row[col_name].strip().strip('"').strip("'")
                        if incorrect_ans:  # Only add non-empty answers
                            incorrect_answers.append(incorrect_ans)

                item = PracticeItem(
                    text=problem_text,
                    answer=answer_text,
                    associated_skills=skill_ids,
                    incorrect_answers=incorrect_answers,
                )
                self.add_item(item)

        self._check_unique_items_and_associated_skills()

    def get_items_for_skill_scenario(
        self,
        mastered_ids: set[str],
        want_mastered: bool,
        max_items: int = 4,
        retrieve_all_learner_skill_prerequisites: bool = True,
        select_items_near_mastery_boundary_first: bool = True,
        item_prerequisites_should_be_mastered: bool = False,
    ) -> list["PracticeItem"]:
        """Select practice items whose skills are all mastered (or not) by the learner.

        Args:
        ----
            mastered_ids: Set of skill IDs the learner has mastered.
            want_mastered: If True, return items where all associated skills are mastered.
                If False, return items where at least one skill is not yet mastered.
            max_items: Maximum number of items to return.
            retrieve_all_learner_skill_prerequisites: If True, expand mastered_ids to
                include all recursive prerequisites before filtering.
            select_items_near_mastery_boundary_first: If True (and want_mastered=False),
                sort items by how close they are to the learner's mastery boundary before
                truncating to max_items.
            item_prerequisites_should_be_mastered: If True (and want_mastered=False),
                only include items whose direct skill prerequisites are all mastered
                (i.e. items the learner is ready to learn next). Skips root skills
                that have no prerequisites.

        Returns:
        -------
            List of matching PracticeItems, up to max_items.

        """
        if retrieve_all_learner_skill_prerequisites:
            all_prereq: set[str] = set(mastered_ids)
            for mid in mastered_ids:
                all_prereq.update(s.id for s in self.skill_space.get_all_prerequisites(mid))
        else:
            all_prereq = set(mastered_ids)

        def _boundary_distance(item: "PracticeItem") -> int:
            item_prereq_ids = set(self.get_item_all_prerequisite_skills(item))
            return max(
                len(item_prereq_ids - all_prereq),
                len(all_prereq - item_prereq_ids),
            )

        out: list[PracticeItem] = []
        for item in self.items:
            all_mastered = all(sk in all_prereq for sk in item.associated_skills)
            if want_mastered and all_mastered:
                out.append(item)
            elif not want_mastered and not all_mastered:
                if not item_prerequisites_should_be_mastered:
                    out.append(item)
                    continue
                item_direct_prerequisites: set[str] = set()
                for sk in item.associated_skills:
                    if sk in self.skill_space:
                        item_direct_prerequisites.update(
                            self.skill_space[sk].prerequisites,
                        )
                if not item_direct_prerequisites:
                    continue
                if item_direct_prerequisites.issubset(all_prereq):
                    out.append(item)

        if not item_prerequisites_should_be_mastered and select_items_near_mastery_boundary_first:
            return sorted(out, key=_boundary_distance)[:max_items]

        random.shuffle(out)
        return out[:max_items]

    def get_random_item(self) -> PracticeItem:
        """Get a random practice item from the pool."""
        if not self.items:
            raise ValueError("PracticeItemPool is empty.")
        return random.choice(self.items)

    def load_incorrect_answers_from_dict(
        self,
        incorrect_answers_dict: dict[str, list[str]],
    ) -> None:
        """Load incorrect answers for practice items from a dictionary.

        Args:
        ----
            incorrect_answers_dict: Dictionary mapping practice item text to list of incorrect answers.
                Format: {"practice_item_text": ["incorrect1", "incorrect2", "incorrect3"]}

        Updates the incorrect_answers attribute of matching PracticeItems in the pool.

        """
        for item in self.items:
            if item.text in incorrect_answers_dict:
                item.incorrect_answers = incorrect_answers_dict[item.text].copy()
