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

    if pd.notna(row["problem_1"]):
        pivoted_data.append(
            {
                "problem": str(row["problem_1"]),
                "skill_id": skill_id,
                "prerequisites": prerequisites,
            },
        )

    if pd.notna(row["problem_2"]):
        pivoted_data.append(
            {
                "problem": str(row["problem_2"]),
                "skill_id": skill_id,
                "prerequisites": prerequisites,
            },
        )

pivoted_df = pd.DataFrame(pivoted_data)
pivoted_df.to_csv("data/florida-doe/tagged-practice-items.csv", index=False)
print("Saved to tagged-practice-items.csv")
