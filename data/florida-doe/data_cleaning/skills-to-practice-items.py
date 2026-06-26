"""Pivot skill-space.csv into tagged-practice-items.csv.

Each skill row with two example problems becomes two output rows,
one per problem, with skill_id and prerequisites carried over.
"""

import pandas as pd

df = pd.read_csv("data/florida-doe/skill-space.csv", engine="python")

pivoted_data = []
for _, row in df.iterrows():
    skill_id = row["skill_id"]
    prerequisites = row["prerequisite_skills"]

    # find all problem_K columns in row:
    problem_columns = [col for col in df.columns if col.startswith("problem_")]
    for problem_col in problem_columns:
        problem = row[problem_col]
        if pd.notna(problem):
            pivoted_data.append(
                {
                    "skill_id": skill_id,
                    "prerequisite_skills": prerequisites,
                    "problem": problem,
                }
            )

pivoted_df = pd.DataFrame(pivoted_data)
pivoted_df.to_csv("data/florida-doe/tagged-practice-items.csv", index=False)
print("Saved to tagged-practice-items.csv")
