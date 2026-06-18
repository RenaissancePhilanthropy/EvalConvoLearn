import logging
import os
from collections import deque
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal, Self, cast, overload

import networkx as nx
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)


class SkillSelectionResponse(BaseModel):
    reasoning: str
    answer: list[int]


class Skill(BaseModel):
    """A skill defines a unit of knowledge that a learner can master.
    Has prerequisite skills that need be mastered to enable mastery.
    A skill is identified by an ID or a description.

    It can be extracted from any skill base, the ID can refer to an existing curriculum standards notation.
    """

    id: str  # required - as part of a skill base
    description: str  # required - skill meaning used for practice
    prerequisites: list[str] = []  # list of prerequisite skill IDs - are not validated to be existing skills

    def __eq__(self, other: object) -> bool:
        # two skills are equal if they have the same id or description, regardless of prerequisites
        if not isinstance(other, Skill):
            return NotImplemented
        return (self.id == other.id) or (self.description == other.description)

    # validate unique prerequisite skills
    @model_validator(mode="after")
    def validate_unique_prerequisites(self) -> Self:
        if len(self.prerequisites) != len(set(self.prerequisites)):
            # find a duplicate prerequisite skill to inform the error
            for sid in self.prerequisites:
                if self.prerequisites.count(sid) > 1:
                    raise ValueError(
                        f"One or more duplicate prerequisite skill IDs found in Skill {self.id}, e.g: {sid}",
                    )

        assert "," not in self.id, f"Skill ID {self.id} cannot contain commas."

        return self

    # add a new prerequisite skill to the prerequisites list
    def add_prerequisite(self, skill_id: str) -> None:
        if skill_id in self.prerequisites:
            raise ValueError(
                f"Prerequisite skill with id {skill_id} already exists for Skill {self.id}.",
            )
        self.prerequisites.append(skill_id)


class SkillSpace(BaseModel):
    """A collection of skills available in the simulation.
    Validates the skill prerequisites structure to avoid cycles.
    Validates that all skills are unique in the space.
    """

    skills: list[Skill] = []
    _skill_graph: nx.DiGraph | None = None  # internal graph of skill ids for the prerequisite structure

    def _check_unique_skills_and_prerequisite_structure(self) -> None:
        skill_ids = [skill.id for skill in self.skills]
        if len(skill_ids) != len(set(skill_ids)):
            # find a duplicate skill to inform the error
            for sid in skill_ids:
                if skill_ids.count(sid) > 1:
                    raise ValueError(
                        f"One or more duplicate skill IDs found in SkillSpace, e.g: {sid}",
                    )

        # validate unique descriptions
        skill_descriptions = [skill.description for skill in self.skills]
        if len(skill_descriptions) != len(set(skill_descriptions)):
            # find a duplicate description to inform the error
            for desc in skill_descriptions:
                if skill_descriptions.count(desc) > 1:
                    raise ValueError(
                        f"One or more duplicate skill descriptions found in SkillSpace, e.g: {desc}",
                    )

        # build a nx graph to check for cycles, and validate that all prerequisite skills exist
        graph = nx.DiGraph()
        for skill in self.skills:
            graph.add_node(skill.id)  # add all skills as nodes

        for skill in self.skills:
            for prerequisite_id in skill.prerequisites:
                assert prerequisite_id in skill_ids, (
                    f"Prerequisite skill ID {prerequisite_id} for skill {skill.id} does not exist in SkillSpace."
                )
                graph.add_edge(prerequisite_id, skill.id)
        graph_string = f"Graph Nodes: {graph.nodes()}, Edges: {graph.edges()}"
        assert nx.is_directed_acyclic_graph(
            graph,
        ), f"Prerequisite structure contains cycles:\n{graph_string}"
        self._skill_graph = (
            graph  # cache the graph for future use in methods that need to traverse the prerequisite structure
        )

    @model_validator(mode="after")
    def validate_unique_skills_and_prerequisite_structure(self) -> Self:
        self._check_unique_skills_and_prerequisite_structure()
        return self

    # get a skill by its ID
    def get_skill(self, skill_id: str) -> Skill:
        for skill in self.skills:
            if skill.id == skill_id:
                return skill
        raise ValueError(f"Skill with id {skill_id} not found in SkillSpace.")

    # equality of SkillSpace based on skills
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SkillSpace):
            return NotImplemented
        # Compare by skill IDs since Skill objects are not hashable
        if len(self.skills) != len(other.skills):
            return False
        self_skill_ids = {skill.id for skill in self.skills}
        other_skill_ids = {skill.id for skill in other.skills}
        return self_skill_ids == other_skill_ids

    def __len__(self) -> int:
        return len(self.skills)

    def __getitem__(self, skill_id: str) -> Skill:
        return self.get_skill(skill_id)

    def __iter__(self) -> Iterator[Skill]:  # type: ignore[override]
        return iter(self.skills)

    # SkillSpace contains method
    def __contains__(self, skill_id: str | Skill) -> bool:
        if isinstance(skill_id, Skill):
            skill_id = skill_id.id
        return any(skill.id == skill_id for skill in self.skills)

    def _validates_skill_object(self, skill_object: str | Skill) -> Skill:
        """Validate and return Skill from skill_object."""
        if isinstance(skill_object, str):
            skill = self.get_skill(skill_object)
        elif isinstance(skill_object, Skill):
            skill = skill_object
        else:
            raise ValueError(
                "skill_object must be either a Skill instance or a string representing the skill ID.",
            )
        return skill

    # get all prerequisite skills for a given skill ID
    def get_prerequisite_skills(self, skill_object: str | Skill) -> list[Skill]:
        skill = self._validates_skill_object(skill_object)
        return [self.get_skill(prereq_id) for prereq_id in skill.prerequisites]

    # get all dependent skills for a given skill ID
    def get_dependent_skills(self, skill_object: str | Skill) -> list[Skill]:
        skill = self._validates_skill_object(skill_object)
        dependents = []
        # not performant for now
        for sk in self.skills:
            if skill.id in sk.prerequisites:
                dependents.append(sk)
        return dependents

    # get all root skills (skills with no prerequisites)
    def get_root_skills(self) -> list[Skill]:
        root_skills = []
        for skill in self.skills:
            if not skill.prerequisites:
                root_skills.append(skill)
        return root_skills

    def get_root_skills_for_target(self, target_skill: str | Skill) -> list[Skill]:
        """Return only the root skills (no prerequisites) that are transitive
        prerequisites of *target_skill* (i.e. belong to its subgraph).

        This is useful when initialising a learner so that only the root skills
        relevant to the learning path toward *target_skill* are mastered,
        rather than every root skill in the entire skill space.
        """
        all_prereq_ids = set(
            self.get_all_prerequisites(
                target_skill,
                include_self=True,
                return_as_ids=True,
            ),
        )
        return [skill for skill in self.get_root_skills() if skill.id in all_prereq_ids]

    # get all the prerequisites recursively for a given skill ID
    @overload
    def get_all_prerequisites(
        self,
        skill_object: str | Skill,
        include_self: bool = ...,
        *,
        return_as_ids: Literal[True],  # keyword only with no default
    ) -> list[str]: ...

    @overload
    def get_all_prerequisites(
        self,
        skill_object: str | Skill,
        include_self: bool = ...,
        return_as_ids: Literal[False] = ...,
    ) -> list[Skill]: ...

    def get_all_prerequisites(
        self,
        skill_object: str | Skill,
        include_self: bool = False,
        return_as_ids: bool = False,
    ) -> list[Skill] | list[str]:
        if isinstance(skill_object, str):
            skill = self.get_skill(skill_object)
        elif isinstance(skill_object, Skill):
            skill = skill_object
        else:
            raise ValueError(
                f"skill_object must be a str or Skill object, got {type(skill_object)}",
            )
        all_prereqs = set()

        def _get_prereqs_recursive(sid: str) -> None:
            sk = self.get_skill(sid)
            for prereq_id in sk.prerequisites:
                if prereq_id not in all_prereqs:
                    all_prereqs.add(prereq_id)
                    _get_prereqs_recursive(prereq_id)

        _get_prereqs_recursive(skill.id)
        if include_self:
            all_prereqs.add(skill.id)
        if return_as_ids:
            return list(all_prereqs)
        return [self.get_skill(prereq_id) for prereq_id in all_prereqs]

    def get_prerequisite_path_to_skill_from_skill_group(
        self,
        target_skill: str | Skill,
        skill_group: Sequence[str | Skill],
    ) -> list[Skill]:
        """Returns one of the shortest, if existing,
        ordered list of skills (by prerequisite relationship) from any skill in the prerequisite skill group TO the target skill.
        -If no path exists or the target_skill is in the skill_group, return [target_skill]
        -The first list's element should be an element in the skill_group and the last should be the target skill
        """
        target = self._validates_skill_object(target_skill)

        group_ids = set()
        for skill_obj in skill_group:
            sk = self._validates_skill_object(skill_obj)
            group_ids.add(sk.id)

        if target.id in group_ids:
            return [target]

        # BFS to find shortest path from any skill in group to target
        # We traverse from target backwards through prerequisites to find a skill in the group
        # Then reverse the path
        queue = deque([(target.id, [target.id])])
        visited = {target.id}

        while queue:
            current_id, path = queue.popleft()

            # Get prerequisite skills for current skill
            current_skill = self.get_skill(current_id)
            for prereq_id in current_skill.prerequisites:
                if prereq_id in visited:
                    continue

                new_path = [prereq_id] + path

                # Check if we found a skill in the group
                if prereq_id in group_ids:
                    return [self.get_skill(sid) for sid in new_path]

                visited.add(prereq_id)
                queue.append((prereq_id, new_path))

        # No path found, return just the target skill
        return [target]

    def render_skill_path(self, skill_path: list[Skill], type: str = "first_mastered_only") -> str:
        """Render a skill path as a meaningful string showing the skills' descriptions in a row with tags."""
        final_string = ""
        if not skill_path:
            return "No skill path"
        if type == "first_mastered_only":
            if len(skill_path) == 1:
                return f"[{skill_path[0].id} UNMASTERED but no prerequisites needed] <{skill_path[0].description}>"
            final_string += f"[{skill_path[0].id} MASTERED] <{skill_path[0].description}>"
            for sk in skill_path[1:]:
                final_string += f" --> [{skill_path[0].id} UNMASTERED] <{sk.description}>"
        else:
            raise NotImplementedError(
                "Only skill path rendering using first mastered only type implemented",
            )
        return final_string

    def choose_skills_for_item(
        self,
        item_text: str,
        mode: Literal["single", "multiple"] = "single",
    ) -> list[Skill]:
        """Use LLM to choose relevant skill(s) for a practice item.
        Inspired from SkillTagger in tagging_items_with_skills.py
        Returns a list of Skill objects associated with the item (empty list if none found)
        """
        if not self.skills:
            return []
        if mode not in ["single", "multiple"]:
            raise ValueError(f"mode must be 'single' or 'multiple', got '{mode}'")

        # Initialize OpenAI client
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Format all skills for the prompt
        skills_text = "\n".join(
            [f"{i + 1}. {skill.description}" for i, skill in enumerate(self.skills)],
        )

        # Adjust prompt based on mode
        if mode == "single":
            selection_instruction = "Select the SINGLE most relevant skill required to solve this problem. There may be cases where no skill is relevant; in such cases, respond with an empty list."
            response_format = "Your answer should be a list with a single integer (e.g., [5])."
        else:
            selection_instruction = """Select the HIGHEST-LEVEL skill(s) specifically required to solve this problem.
            There may be cases where no skill is relevant; in such cases, respond with an empty list.

IMPORTANT RULES:
1. Only select the most advanced/highest-level skills directly needed
2. Do NOT select prerequisite skills that are implied by higher-level skills
3. For example: If a problem requires a higher-level skill that has prerequisites, don't include those prerequisite skills"""
            response_format = "Your answer should be a list of integers (e.g., [1, 5, 10])."

        prompt = f"""Given the following practice item/problem:
"{item_text}"

{selection_instruction}

Available skills:
{skills_text}

First, provide your reasoning about which skill(s) are required.
Then provide your answer as a list of integers (1-indexed) corresponding to relevant skill(s).

{response_format}
"""
        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an education expert helping to identify skills required for practice items.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format=SkillSelectionResponse,
            )
            content = response.choices[0].message.parsed
            if content is None:
                return []
            # Convert 1-indexed answers to 0-indexed and validate
            selected_indices = [idx - 1 for idx in content.answer if 0 < idx <= len(self.skills)]
            # Get skill objects for selected indices
            selected_skills = [self.skills[idx] for idx in selected_indices]

            # If mode is single, ensure only one skill is returned
            if mode == "single" and len(selected_skills) > 1:
                selected_skills = [selected_skills[0]]

            return selected_skills

        except Exception as e:
            raise RuntimeError(f"Failed to get skill selection from LLM: {e}") from e

    def load_skills_from_csv(self, file_path: str | Path) -> None:
        """Load skills from a CSV using pandas, using columns:
        skill_id, skill_description, prerequisite_skills (comma-separated).

        The prerequisite_skills column is only the DIRECT, one-hop prerequisites and not all transitive prerequisites.

        Returns a new SkillSpace instance with the loaded skills.
        """
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            prereq_list = (
                [sid.strip() for sid in row["prerequisite_skills"].split(",")]
                if pd.notna(row["prerequisite_skills"])
                else []
            )
            skill = Skill(
                id=row["skill_id"].strip().strip('"').strip("'"),
                description=row["skill_description"].strip().strip('"').strip("'"),
                prerequisites=prereq_list,
            )
            self.skills.append(skill)

        self._check_unique_skills_and_prerequisite_structure()

    def get_unique_separate_subgraphs_for_skills(
        self,
        skills: list[str] | list[Skill],
    ) -> list[list[Skill]]:
        """Get subgraphs which contain any of the input skills.

        Return the descending topologically sorted skills in each subgraph
        meaning the most advanced skills (with more prerequisites) are first
        and the more foundational skills are later in the list.
        """
        skill_ids = set()
        for skill_obj in skills:
            skill = self._validates_skill_object(
                skill_obj,
            )  # validate skill objects and existence in skill space
            skill_ids.add(skill.id)

        if self._skill_graph is None:
            raise ValueError(
                "Internal skill graph not found. Ensure that the SkillSpace is properly initialized and validated.",
            )

        # Get all connected components (subgraphs) of the skill graph
        subgraphs = []
        for component in nx.weakly_connected_components(self._skill_graph):
            if any(sid in skill_ids for sid in component):
                # order the skills by topological sort, descending from the most advanced skills to the more foundational ones
                sorted_component = list(
                    reversed(
                        list(
                            nx.topological_sort(cast(nx.DiGraph, self._skill_graph.subgraph(component))),
                        ),
                    ),
                )
                subgraphs.append([self.get_skill(str(sid)) for sid in sorted_component])
        return subgraphs

    def get_bfs_skill_order(
        self,
        target_skill_id: str,
        filter_out_root_skills: bool = True,
    ) -> list["Skill"]:
        """Return a BFS-ordered list of skills from root skills to *target_skill_id*.

        The ordering ensures that for each skill in the returned list, all of
        its prerequisites appear earlier, making each skill "learnable" when
        its turn comes.  Root skills are excluded (the learner starts with
        them already mastered).

        Parameters
        ----------
        target_skill_id:
            The skill to climb towards.

        Returns
        -------
        list[Skill]
            Topologically sorted skills (roots excluded) ending with the
            target skill.

        """
        if self._skill_graph is None:
            raise ValueError(
                "Internal skill graph not found. Ensure that the SkillSpace is properly initialized and validated.",
            )

        target_skill = self.get_skill(target_skill_id)
        # Get all transitive prerequisites including the target itself
        all_prereqs = self.get_all_prerequisites(
            target_skill,
            include_self=True,
            return_as_ids=True,
        )
        root_skill_ids = {sk.id for sk in self.get_root_skills()}

        subgraph = cast(nx.DiGraph, self._skill_graph.subgraph(all_prereqs).copy())
        topo_order = list(nx.topological_sort(subgraph))

        # Filter out root skills (learner already has them)
        ordered_skills = [
            self.get_skill(str(sid))
            for sid in topo_order
            if not (filter_out_root_skills and str(sid) in root_skill_ids)
        ]
        return ordered_skills

    def get_all_subgraphs_of_skill_prerequisites(
        self,
        skills: list[str] | list[Skill],
    ) -> list[list[Skill]]:
        """Get distinct subgraphs of all prerequisites for the ensemble of input skills.

        First get the set of all prerequisites from all skills until root skills.
        Then get the distinct subgraphs of those prerequisites using weakly connected components.
        """
        skill_ids = set()
        for skill_obj in skills:
            skill = self._validates_skill_object(
                skill_obj,
            )  # validate skill objects and existence in skill space
            skill_ids.add(skill.id)
        logger.debug("Getting all subgraphs of prerequisites for skills: %s", skill_ids)
        all_prereq_ids = set()
        for sid in skill_ids:
            prereqs = self.get_all_prerequisites(sid, include_self=True)
            all_prereq_ids.update([sk.id for sk in prereqs])
        logger.debug("All prerequisite skill IDs: %s", all_prereq_ids)
        if self._skill_graph is None:
            raise ValueError(
                "Internal skill graph not found. Ensure that the SkillSpace is properly initialized and validated.",
            )
        subgraphs = []
        all_prereq_subgraph = cast(nx.DiGraph, self._skill_graph.subgraph(all_prereq_ids))
        logger.debug(
            "Subgraph of all prerequisites - Nodes: %s, Edges: %s",
            all_prereq_subgraph.nodes(),
            all_prereq_subgraph.edges(),
        )
        for component in nx.weakly_connected_components(all_prereq_subgraph):
            sorted_component = list(
                reversed(
                    list(nx.topological_sort(cast(nx.DiGraph, all_prereq_subgraph.subgraph(component)))),
                ),
            )
            logger.debug("Component: %s, Sorted: %s", component, sorted_component)
            subgraphs.append([self.get_skill(str(sid)) for sid in sorted_component])
        return subgraphs
