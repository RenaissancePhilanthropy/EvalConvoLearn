"""Metric calculation functions for benchmark evaluations."""


def calculate_placement_test_alignment(
    result,
    expected_correct_skills: set[str],
    expected_incorrect_skills: set[str],
) -> tuple[bool | None, bool]:
    """Calculate alignment for a single placement test result.

    Args:
    ----
        result: PlacementTestResult object
        expected_correct_skills: Set of skill IDs expected to be answered correctly
        expected_incorrect_skills: Set of skill IDs expected to be answered incorrectly

    Returns:
    -------
        tuple of (expected_correct, is_aligned):
            - expected_correct: True if skill should be correct, False if incorrect, None if not in matrix
            - is_aligned: True if answer matches expectation

    """
    skill_id = (
        result.practice_item.associated_skills[0]
        if result.practice_item.associated_skills
        else None
    )

    expected_correct = None
    if skill_id in expected_correct_skills:
        expected_correct = True
    elif skill_id in expected_incorrect_skills:
        expected_correct = False

    effective_is_correct = result.is_correct
    if (
        result.answer_choices
        and result.learner_choice_index is not None
        and 0 <= result.learner_choice_index < len(result.answer_choices)
    ):
        selected_answer = result.answer_choices[result.learner_choice_index]
        effective_is_correct = selected_answer == result.correct_answer

    is_aligned = None
    if expected_correct is not None:
        is_aligned = effective_is_correct == expected_correct

    return (expected_correct, is_aligned)
