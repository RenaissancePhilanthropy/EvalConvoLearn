"""Learner configuration data for benchmarks."""


def get_placement_test_skill_levels() -> dict[str, list[str]]:
    """Get skill levels for placement test learners at different proficiency levels.

    Returns
    -------
        dict: Mapping from learner level to list of mastered skill IDs
            - beginner: 3 foundational skills
            - intermediate: 6 skills (foundational + level 2)
            - expert: 8 skills (all available skills)

    """
    return {
        "beginner": [
            "MA.6.NSO.1.1",
            "MA.6.NSO.2.1",
            "MA.6.NSO.3.1",
        ],
        "intermediate": [
            "MA.6.NSO.1.1",
            "MA.6.NSO.1.2",
            "MA.6.NSO.2.1",
            "MA.6.NSO.2.2",
            "MA.6.NSO.3.1",
            "MA.6.NSO.3.2",
        ],
        "expert": [
            "MA.6.NSO.1.1",
            "MA.6.NSO.1.2",
            "MA.6.NSO.1.3",
            "MA.6.NSO.2.1",
            "MA.6.NSO.2.2",
            "MA.6.NSO.3.1",
            "MA.6.NSO.3.2",
            "MA.6.NSO.3.3",
        ],
    }


def get_beginner_config() -> dict:
    """Get configuration for beginner learner.

    Returns
    -------
        dict with mastered_skills list

    """
    return {"mastered_skills": get_placement_test_skill_levels()["beginner"]}


def get_intermediate_config() -> dict:
    """Get configuration for intermediate learner.

    Returns
    -------
        dict with mastered_skills list

    """
    return {"mastered_skills": get_placement_test_skill_levels()["intermediate"]}


def get_expert_config() -> dict:
    """Get configuration for expert learner.

    Returns
    -------
        dict with mastered_skills list

    """
    return {"mastered_skills": get_placement_test_skill_levels()["expert"]}


def get_blank_config() -> dict:
    """Get configuration for learner with no mastered skills.

    Returns
    -------
        dict with empty mastered_skills list

    """
    return {"mastered_skills": []}
