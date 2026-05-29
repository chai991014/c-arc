import json
import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import sparse

INPUT_JSON = "../../../data_factory/datasets/synthetic_dataset.json"
OUTPUT_DIR = "./processed_data"


def engineer_features():
    print("Loading synthetic dataset...")
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    print("Flattening ALL Professional Vectors (Tasks, DWAs, Skills, Tech)...")

    # 1. One-Hot Encode Skills
    mlb_tasks = MultiLabelBinarizer(sparse_output=True)
    mlb_dwas = MultiLabelBinarizer(sparse_output=True)
    mlb_skills = MultiLabelBinarizer(sparse_output=True)
    mlb_tech = MultiLabelBinarizer(sparse_output=True)

    # 2. Transform the data
    tasks_sparse = mlb_tasks.fit_transform(df.get('synthetic_tasks', []))
    dwas_sparse = mlb_dwas.fit_transform(df.get('synthetic_dwas', []))
    skills_sparse = mlb_skills.fit_transform(df.get('synthetic_skills', []))
    tech_sparse = mlb_tech.fit_transform(df.get('synthetic_tech_skills', []))

    # 3. Extract OCEAN traits (Dense data)
    ocean_df = pd.json_normalize(df['ocean_vector'])

    # 4. Combine into a final Feature Matrix
    # The stacking order here MUST match the inference vectorization order
    feature_matrix = sparse.hstack([
        tasks_sparse,
        dwas_sparse,
        skills_sparse,
        tech_sparse,
        ocean_df.values
    ])

    # 5. Save components
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df['soc_code'].to_csv(f"{OUTPUT_DIR}/labels.csv", index=False)
    sparse.save_npz(f"{OUTPUT_DIR}/feature_matrix.npz", feature_matrix)

    # Helper function to save classes cleanly
    def save_classes(mlb, filename):
        # Convert to int if the IDs are numeric, otherwise keep as string
        classes = [int(x) if str(x).isdigit() else str(x) for x in mlb.classes_]
        with open(f"{OUTPUT_DIR}/{filename}", 'w') as f:
            json.dump(classes, f)

    save_classes(mlb_tasks, "task_classes.json")
    save_classes(mlb_dwas, "dwa_classes.json")
    save_classes(mlb_skills, "skill_classes.json")
    save_classes(mlb_tech, "tech_classes.json")

    print(f"✅ Success! Feature matrix saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    engineer_features()
