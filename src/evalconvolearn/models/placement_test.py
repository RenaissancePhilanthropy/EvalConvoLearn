from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, model_validator

if TYPE_CHECKING:
    from ..core.flexlearner import FlexLearner
    from .binary_skills_flexlearner import BinarySkillsFlexLearner

from ..models.practice_item import PracticeItem, PracticeItemPool
from ..models.skill import Skill

logger = logging.getLogger(__name__)


class PracticeItemResult(BaseModel):
    """Result of a single practice item in a placement test."""

    practice_item: PracticeItem
    is_correct: bool
    required_skills: list[str]  # All skills (including prerequisites) required to answer correctly
    learner_has_all_required_skills: bool | None = None  # Whether learner has all required skills (if checked)
    learner_answer: str | None = None  # The learner's actual answer (if provided)
    learner_choice_index: int | None = None  # Index of learner's choice in answer_choices (if multiple choice)
    correct_answer: str = ""  # The correct answer from the practice item
    incorrect_answers: list[str] = []  # Incorrect answer choices (for multiple choice)
    answer_choices: list[str] = []  # All answer choices presented (for multiple choice)
    answer_choice_letters: list[str] = []  # Letters assigned to answer choices (e.g., A, B, C)
    correct_choice_index: int = -1  # Index of correct answer in answer_choices
    correct_answer_letter: str | None = None  # Letter of the correct answer choice (e.g., "A", "B")
    prompt_content: str = ""  # The full prompt text sent to the LLM for answer generation

    @model_validator(mode="after")
    def set_correct_answer_and_choices(self) -> Self:
        """Set the correct answer and choices from the practice item."""
        self.correct_answer = self.practice_item.get_answer()
        # Set incorrect answers from the practice item if not already set
        if not self.incorrect_answers and self.practice_item.incorrect_answers:
            self.incorrect_answers = self.practice_item.incorrect_answers.copy()
        # If answer_choices were provided, set the correct index
        if self.answer_choices:
            self.correct_choice_index = self.practice_item.get_correct_choice_index(
                self.answer_choices,
            )
        return self


class PlacementTest(BaseModel):
    """A placement test that evaluates a learner's knowledge using practice items.

    The test assesses whether the learner can correctly answer practice items
    based on their mastered skills. A learner can answer a practice item correctly
    only if they have mastered ALL the associated skills AND all their prerequisites.

    The learner is passed to methods rather than stored in the test instance,
    allowing the same test to be administered to multiple learners.

    This can be used for:
    - Pre-lesson evaluation: assess current knowledge before instruction
    - Post-lesson evaluation: assess knowledge gained after instruction
    """

    practice_item_pool: PracticeItemPool
    test_results: list[PracticeItemResult] = []

    def _get_all_required_skills_for_item(
        self,
        practice_item: PracticeItem,
    ) -> list[Skill]:
        """Get all skills required to correctly answer a practice item.

        This includes:
        - The single associated skill of the practice item
        - All prerequisites (recursively) of that associated skill
        """
        all_required_skills = set()

        # Get the single associated skill (first one in the list)
        if not practice_item.associated_skills:
            return []

        skill_id = practice_item.associated_skills[0]
        skill = self.practice_item_pool.skill_space.get_skill(skill_id)
        all_required_skills.add(skill.id)

        # Get all prerequisites recursively
        prerequisites = self.practice_item_pool.skill_space.get_all_prerequisites(
            skill,
        )
        for prereq in prerequisites:
            all_required_skills.add(prereq.id)

        return [self.practice_item_pool.skill_space.get_skill(skill_id) for skill_id in all_required_skills]

    def _can_learner_answer_correctly(
        self,
        practice_item: PracticeItem,
        learner: FlexLearner | BinarySkillsFlexLearner,
    ) -> bool:
        """Determine if learner can answer a practice item correctly.

        Args:
        ----
            practice_item: The practice item to evaluate
            learner: The learner being assessed

        Returns:
        -------
            True if learner has all required skills, False otherwise

        For ``BinarySkillsFlexLearner`` instances (skill-binary), this checks mastered_skills
        directly. For other ``FlexLearner`` subclasses, this uses an LLM
        call with ``get_required_knowledge_to_answer_practice_item`` to judge
        whether the learner's knowledge is sufficient.

        """
        from .binary_skills_flexlearner import BinarySkillsFlexLearner

        required_skills = self._get_all_required_skills_for_item(practice_item)

        if type(learner) is BinarySkillsFlexLearner:
            # Skill-binary check: learner must have ALL required skills
            required_skills = required_skills or []
            required_skill_ids = {skill.id for skill in required_skills}
            mastered_skill_ids = set(learner.mastered_skills)

            return required_skill_ids.issubset(mastered_skill_ids)

        # For non-Learner FlexLearner subclasses: use LLM-based
        # knowledge sufficiency check via the learner's own knowledge retrieval.
        from ..utils.llm_evaluator import evaluate_knowledge_sufficiency

        knowledge_text = learner.get_required_knowledge_to_answer_practice_item(
            practice_item=practice_item,
            practice_item_skills=required_skills or [],
        )
        verdict = evaluate_knowledge_sufficiency(
            problem_text=practice_item.text,
            learner_knowledge=knowledge_text,
        )
        logger.info(
            "[PlacementTest] Knowledge sufficiency check for non-Learner — can_answer=%s, reasoning='%s'",
            verdict.can_answer_correctly,
            verdict.reasoning[:120],
        )
        return verdict.can_answer_correctly

    def _generate_placement_test_prompt(
        self,
        practice_item_text: str,
        answer_choices: list[str],
        can_answer_correctly: bool | None = None,
    ) -> str:
        """Generate the placement test prompt based on whether learner can answer.

        Args:
        ----
            practice_item_text: The question text
            answer_choices: List of answer choices (may be empty)
            can_answer_correctly: Whether the learner has required skills

        Returns:
        -------
            The complete prompt string

        """
        # Format answer choices
        answer_choices_text = ""
        if answer_choices:
            letters = [chr(ord("A") + i) for i in range(len(answer_choices))]
            answer_choices_text = "\n".join(
                [f"{letter}. {choice}" for letter, choice in zip(letters, answer_choices, strict=False)],
            )
            answer_choices_text = f"Answer choices:\n{answer_choices_text}"
            response_instructions = (
                f"Select {'an' if can_answer_correctly is None else 'the correct' if can_answer_correctly else 'an incorrect'} answer choice if choices are provided.\n"
                "Your final answer MUST be a single letter from the answer choices (e.g., A, B, C).\n"
                "Reply with ONLY the letter, no other text.\n"
                "If no choices are provided, return a specific number or measurement (include units)."
            )
        else:
            # open ended response
            response_instructions = (
                f"Provide your reasoning and your specific {'correct' if can_answer_correctly else 'incorrect'} answer to the question.\n"
                "Your answer should contain a number or measurement (include units if applicable)."
            )

        if can_answer_correctly is True:
            return f"""Question:
{practice_item_text}

{answer_choices_text}

Solve this problem carefully and correctly.
{response_instructions}
"""
        elif can_answer_correctly is False:
            return f"""Question:
{practice_item_text}

{answer_choices_text}

Make a reasonable response attempt that shows an error typical of students who struggle with this problem.
{response_instructions}
"""
        else:
            # can_answer_correctly is None — no knowledge check was performed.
            # Return empty string so the non-Learner path in answer_practice_item
            # builds its own knowledge-grounded prompt.
            return ""

    def administer_item(
        self,
        practice_item: PracticeItem | str,
        learner: FlexLearner,
        learner_answer: str | None = None,
        learner_choice_index: int | None = None,
        use_multiple_choice: bool = False,
        use_llm_for_answer: bool = False,
        check_if_has_knowledge_before_answering: bool = False,
    ) -> PracticeItemResult:
        """Administer a single practice item to a learner.

        Args:
        ----
            practice_item: PracticeItem object or text string identifying the item
            learner: The learner taking the test
            learner_answer: Optional answer provided by the learner (free text)
            learner_choice_index: Index of learner's choice (for multiple choice)
            use_multiple_choice: Whether to present as multiple choice
            use_llm_for_answer: Whether to have the learner generate answer via LLM

        Returns:
        -------
            PracticeItemResult with the outcome and details

        Raises:
        ------
            ValueError: If learner's skill_space doesn't match the practice_item_pool's skill_space

        """
        # Validate learner uses the same skill space
        if learner.skill_space != self.practice_item_pool.skill_space:
            raise ValueError(
                "Learner and PracticeItemPool must use the same SkillSpace.",
            )

        # Validate and get practice item
        if isinstance(practice_item, str):
            item = self.practice_item_pool.get_item_by_text(practice_item)
        elif isinstance(practice_item, PracticeItem):
            item = practice_item
        else:
            raise ValueError(
                "practice_item must be either a PracticeItem instance or a string representing the item text.",
            )

        # Check if item exists in pool
        if item not in self.practice_item_pool:
            raise ValueError(
                f"PracticeItem '{item.text}' not found in the practice item pool.",
            )

        # Determine if learner can answer correctly based on skills
        required_skills = self._get_all_required_skills_for_item(item)
        can_answer = None
        if check_if_has_knowledge_before_answering:
            can_answer = self._can_learner_answer_correctly(item, learner)
            logger.info(
                "[PlacementTest] Skill check for item '%s': required_skills=%s, can_answer_correctly=%s",
                item.text[:60],
                [s.id for s in required_skills],
                can_answer,
            )
        else:
            logger.info(
                "[PlacementTest] No skill check requested for item '%s'.",
                item.text[:60],
            )

        # Generate answer choices if using multiple choice
        answer_choices = []
        answer_choice_letters: list[str] = []
        incorrect_answers = item.incorrect_answers.copy()

        # Only use multiple choice when explicitly requested — never auto-enable it
        # for LLM-generated answers so open-ended validation is always possible.
        effective_use_multiple_choice = use_multiple_choice and bool(incorrect_answers)

        if effective_use_multiple_choice and incorrect_answers:
            answer_choices = item.get_all_choices(shuffle=True)
            answer_choice_letters = [chr(ord("A") + i) for i in range(len(answer_choices))]

        # Get learner's answer via LLM if requested
        prompt_content = ""
        if use_llm_for_answer and learner_answer is None:
            # Generate the prompt in PlacementTest
            # Generate the prompt here only for Learner class.
            prompt_content = self._generate_placement_test_prompt(
                practice_item_text=item.text,
                answer_choices=answer_choices,  # empty when not using multiple choice
                can_answer_correctly=can_answer,
            )
            logger.info(
                "[PlacementTest] Requesting LLM answer for item '%s' "
                "(can_answer_correctly=%s, multiple_choice=%s).\nPrompt:\n%s",
                item.text[:60],
                can_answer,
                effective_use_multiple_choice,
                prompt_content,
            )
            # For non-Learner class, prompt_content is "" (can_answer is None) —
            # answer_practice_item will use the 'get knowledge' path, which requires
            # practice_item_text and practice_item_skill_ids to build the prompt.
            result = learner.answer_practice_item(
                prompt=prompt_content or None,
                practice_item_text=item.text,
                practice_item_skills=required_skills,
                return_prompt=True,
            )
            if isinstance(result, dict):
                learner_answer = result["answer"]
                # prompt_content already set above
            else:
                # if the answer is a letter, keep going
                # if it is not a letter, check correctness with llm call:
                learner_answer = result
            logger.info(
                "[PlacementTest] LLM returned answer: '%s'",
                learner_answer,
            )

            if answer_choices and learner_answer:
                normalized_answer = learner_answer.strip().upper()
                if normalized_answer in answer_choice_letters:
                    learner_choice_index = answer_choice_letters.index(
                        normalized_answer,
                    )
                elif learner_answer in answer_choices:
                    learner_choice_index = answer_choices.index(learner_answer)

        # Determine if learner's response is correct
        actual_is_correct = False
        if answer_choices and learner_choice_index is None and learner_answer:
            normalized_answer = learner_answer.strip().upper()
            if normalized_answer in answer_choice_letters:
                learner_choice_index = answer_choice_letters.index(normalized_answer)
            elif learner_answer in answer_choices:
                learner_choice_index = answer_choices.index(learner_answer)

        if answer_choices and learner_choice_index is not None:
            # Validate the choice matches the correct answer
            if 0 <= learner_choice_index < len(answer_choices):
                selected_answer = answer_choices[learner_choice_index]
                actual_is_correct = selected_answer == item.get_answer()
        elif learner_answer is not None:
            if use_llm_for_answer:
                # Use LLM to evaluate correctness of the answer
                from ..utils.llm_evaluator import evaluate_response_correctness

                verdict = evaluate_response_correctness(
                    problem_text=item.text,
                    learner_response=learner_answer,
                    correct_answer=item.get_answer(),
                )
                actual_is_correct = verdict.is_correct
                logger.info(
                    "[PlacementTest] LLM evaluation — is_correct=%s reasoning=%s",
                    verdict.is_correct,
                    verdict.reasoning[:120],
                )
        else:
            actual_is_correct = False
            logger.info(
                "[PlacementTest] No answer provided by learner. Setting is_correct=False. skill_check=%s",
                can_answer,
            )

        logger.info(
            "[PlacementTest] Result — item='%s', is_correct=%s, "
            "learner_answer='%s', correct_answer='%s', "
            "learner_has_all_required_skills=%s",
            item.text[:60],
            actual_is_correct,
            learner_answer,
            item.get_answer(),
            can_answer,
        )

        # Determine correct answer letter
        correct_answer_letter = None
        if answer_choices and answer_choice_letters:
            correct_choice_idx = item.get_correct_choice_index(answer_choices)
            if 0 <= correct_choice_idx < len(answer_choice_letters):
                correct_answer_letter = answer_choice_letters[correct_choice_idx]

        # Create result
        result = PracticeItemResult(
            practice_item=item,
            is_correct=actual_is_correct,
            required_skills=[skill.id for skill in required_skills],
            learner_has_all_required_skills=can_answer,
            learner_answer=learner_answer,
            learner_choice_index=learner_choice_index,
            incorrect_answers=incorrect_answers,
            answer_choices=answer_choices,
            answer_choice_letters=answer_choice_letters,
            correct_answer_letter=correct_answer_letter,
            prompt_content=prompt_content,
        )

        # Store result
        self.test_results.append(result)
        return result

    def administer_items(
        self,
        practice_items: list[PracticeItem | str],
        learner: FlexLearner,
        use_llm_for_answer: bool = False,
    ) -> list[PracticeItemResult]:
        """Administer multiple practice items to a learner.

        Args:
        ----
            practice_items: List of PracticeItem objects or text strings
            learner: The learner taking the test
            use_llm_for_answer: Whether to have the learner generate answers via LLM
            check_if_has_knowledge_before_answering: Whether to check if the learner has the required knowledge before answering

        Returns:
        -------
            List of PracticeItemResults

        """
        results = []
        for item in practice_items:
            result = self.administer_item(
                item,
                learner,
                use_llm_for_answer=use_llm_for_answer,
            )
            results.append(result)
        return results

    def administer_items_for_skills(
        self,
        skill_ids: list[str],
        learner: FlexLearner,
        items_per_skill: int = 1,
        use_llm_for_answer: bool = False,
    ) -> list[PracticeItemResult]:
        """Administer practice items aligned to specific skills.

        Args:
        ----
            skill_ids: List of skill IDs to test
            learner: The learner taking the test
            items_per_skill: Number of items to test per skill (default: 1)
            use_llm_for_answer: Whether to have the learner generate answers via LLM

        Returns:
        -------
            List of PracticeItemResults

        """
        results = []

        for skill_id in skill_ids:
            # Validate skill exists
            if skill_id not in self.practice_item_pool.skill_space:
                raise ValueError(
                    f"Skill ID {skill_id} not found in the skill space.",
                )

            # Find items associated with this skill
            matching_items = [item for item in self.practice_item_pool.items if skill_id in item.associated_skills]

            if not matching_items:
                raise ValueError(
                    f"No practice items found for skill ID {skill_id}.",
                )

            # Administer up to items_per_skill items
            for item in matching_items[:items_per_skill]:
                result = self.administer_item(
                    item,
                    learner,
                    use_llm_for_answer=use_llm_for_answer,
                )
                results.append(result)

        return results

    def get_test_summary(self) -> dict:
        """Get a summary of the test results.

        Returns
        -------
            Dictionary with test statistics including answer validation

        """
        if not self.test_results:
            return {
                "total_items": 0,
                "correct": 0,
                "incorrect": 0,
                "accuracy": 0.0,
                "items_with_answers": 0,
                "answers_validated": 0,
            }

        correct_count = sum(1 for result in self.test_results if result.is_correct)
        total_count = len(self.test_results)
        items_with_answers = sum(1 for result in self.test_results if result.learner_answer is not None)
        answers_validated = sum(
            1
            for result in self.test_results
            if result.learner_answer is not None and self.validate_answer(result.learner_answer, result.correct_answer)
        )

        expected_correct = [r for r in self.test_results if r.learner_has_all_required_skills is True]
        expected_incorrect = [r for r in self.test_results if r.learner_has_all_required_skills is False]

        expected_correct_count = len(expected_correct)
        expected_incorrect_count = len(expected_incorrect)

        aligned_correct = sum(1 for r in expected_correct if r.is_correct)
        aligned_incorrect = sum(1 for r in expected_incorrect if not r.is_correct)

        return {
            "total_items": total_count,
            "correct": correct_count,
            "incorrect": total_count - correct_count,
            "accuracy": correct_count / total_count if total_count > 0 else 0.0,
            "items_with_answers": items_with_answers,
            "answers_validated": answers_validated,
            "answer_validation_rate": (answers_validated / items_with_answers if items_with_answers > 0 else 0.0),
            "expected_correct_count": expected_correct_count,
            "expected_incorrect_count": expected_incorrect_count,
            "aligned_when_expected_correct": aligned_correct,
            "aligned_when_expected_incorrect": aligned_incorrect,
            "pct_aligned_when_expected_correct": (
                aligned_correct / expected_correct_count if expected_correct_count > 0 else None
            ),
            "pct_aligned_when_expected_incorrect": (
                aligned_incorrect / expected_incorrect_count if expected_incorrect_count > 0 else None
            ),
        }

    def clear_results(self) -> None:
        """Clear all test results to start a new test."""
        self.test_results = []

    def validate_answer(self, learner_answer: str, correct_answer: str) -> bool:
        """Compare learner's answer with the correct answer.

        Args:
        ----
            learner_answer: Answer provided by the learner
            correct_answer: Correct answer from the practice item

        Returns:
        -------
            True if answers match (with flexible numerical comparison), False otherwise

        """
        if not learner_answer or not correct_answer:
            return False

        # Try exact match first (case-insensitive, whitespace-trimmed)
        if learner_answer.strip().lower() == correct_answer.strip().lower():
            return True

        # Try to extract and compare numerical values
        import re

        # Extract all numbers (including decimals and fractions)
        learner_numbers = re.findall(r"\d+\.?\d*", learner_answer)
        correct_numbers = re.findall(r"\d+\.?\d*", correct_answer)

        if learner_numbers and correct_numbers:
            try:
                # Compare the main numerical value (usually the first number)
                learner_value = float(learner_numbers[0])
                correct_value = float(correct_numbers[0])

                # Allow 10% tolerance for numerical answers (LLMs may round or approximate)
                tolerance = abs(correct_value * 0.10) or 0.1
                return abs(learner_value - correct_value) <= tolerance
            except (ValueError, IndexError):
                pass

        return False
