import json
import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import sparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths (Adjusted for your new directory structure)
INPUT_JSON = "../../../data_factory/datasets/synthetic_dataset.json"
OUTPUT_DIR = "./processed_data"


def engineer_features():
    print("Loading synthetic dataset...")
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    print("Flattening Task and DWA vectors...")

    # 1. One-Hot Encode Skills
    mlb_tasks = MultiLabelBinarizer(sparse_output=True)
    mlb_dwas = MultiLabelBinarizer(sparse_output=True)

    tasks_sparse = mlb_tasks.fit_transform(df['synthetic_tasks'])
    dwas_sparse = mlb_dwas.fit_transform(df['synthetic_dwas'])

    # 2. Extract OCEAN traits (Dense data)
    ocean_df = pd.json_normalize(df['ocean_vector'])

    # 3. Combine into a final Feature Matrix
    feature_matrix = sparse.hstack([tasks_sparse, dwas_sparse, ocean_df.values])

    # 4. Save components
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df['soc_code'].to_csv(f"{OUTPUT_DIR}/labels.csv", index=False)
    sparse.save_npz(f"{OUTPUT_DIR}/feature_matrix.npz", feature_matrix)

    # CONVERT TO NATIVE INT: This fixes the TypeError
    task_classes = [int(x) for x in mlb_tasks.classes_]
    # If DWA IDs are strings, this will work, if they are ints, use the same int(x) logic
    dwa_classes = list(mlb_dwas.classes_)

    with open(f"{OUTPUT_DIR}/task_classes.json", 'w') as f:
        json.dump(task_classes, f)
    with open(f"{OUTPUT_DIR}/dwa_classes.json", 'w') as f:
        json.dump(dwa_classes, f)

    print(f"✅ Success! Feature matrix saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    engineer_features()
