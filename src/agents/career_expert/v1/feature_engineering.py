import json
import pandas as pd
import os
import joblib
import scipy.sparse as sp
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Configuration for Output
EXPORT_DIR = "./processed_data"


def generate_master_feature_blueprint(benchmark_json_path, skill_pool_path, tech_pool_path):
    """
    STEP 1: Scans the ground-truth pools and benchmark datasets to create
    the absolute fixed list of feature column names.
    """
    print("Building Master Feature Blueprint from ground-truth artifacts...")
    master_columns = ['OCEAN_O', 'OCEAN_C', 'OCEAN_E', 'OCEAN_A', 'OCEAN_N']

    with open(skill_pool_path, 'r', encoding='utf-8') as f:
        skills = json.load(f)
        for s in skills:
            master_columns.append(f"SKILL_{s['id']}")

    with open(tech_pool_path, 'r', encoding='utf-8') as f:
        techs = json.load(f)
        for t in techs:
            master_columns.append(f"HOT_{t['id']}")
            master_columns.append(f"BASE_{t['id']}")

    with open(benchmark_json_path, 'r', encoding='utf-8') as f:
        benchmarks = json.load(f)

    path_set = set()
    for job in benchmarks:
        env_hierarchy = job['inputs'].get('environmental_hierarchy', {})
        for wa_id, wa_data in env_hierarchy.items():
            dwas = wa_data.get('execution_dwas', {})
            if not dwas:
                path_set.add(f"path_WA_{wa_id}_orphan")
            else:
                for dwa_id, task_data in dwas.items():
                    core_tasks = task_data.get('core_tasks', [])
                    supp_tasks = task_data.get('supplemental_tasks', [])

                    if not core_tasks and not supp_tasks:
                        path_set.add(f"path_WA_{wa_id}_DWA_{dwa_id}_orphan")
                    else:
                        for task_id in core_tasks:
                            path_set.add(f"path_WA_{wa_id}_DWA_{dwa_id}_core_{task_id}")
                        for task_id in supp_tasks:
                            path_set.add(f"path_WA_{wa_id}_DWA_{dwa_id}_supp_{task_id}")

    master_columns.extend(sorted(list(path_set)))
    # Sanitize names for XGBoost (remove [, ], <)
    master_columns = [str(c).replace('[', '').replace(']', '').replace('<', '') for c in master_columns]

    print(f"Master Blueprint established with exactly {len(master_columns)} fixed columns.")
    return master_columns


def build_xgboost_training_matrix(synthetic_csv_path, master_columns):
    """
    STEP 2: Transforms the synthetic training dataset into the strict sparse matrix,
    enforcing the Master Blueprint.
    """
    print(f"Loading synthetic dataset from {synthetic_csv_path}...")
    df_synth = pd.read_csv(synthetic_csv_path)

    X_features = []
    y_raw_targets = []
    w_scores = []

    print("Mapping synthetic rows to the Master Blueprint...")
    for idx, row in tqdm(df_synth.iterrows(), total=len(df_synth), desc="Parsing CSV Rows"):
        row_features = {}

        y_raw_targets.append(row['target_soc_code'])
        w_scores.append(float(row['label_score']))

        row_features['OCEAN_O'] = float(row['O'])
        row_features['OCEAN_C'] = float(row['C'])
        row_features['OCEAN_E'] = float(row['E'])
        row_features['OCEAN_A'] = float(row['A'])
        row_features['OCEAN_N'] = float(row['N'])

        core_skills = json.loads(row['synthetic_core_skills']) if isinstance(row['synthetic_core_skills'], str) else []
        for skill in core_skills:
            row_features[f"SKILL_{skill}"] = 1.0

        hot_tech = json.loads(row['synthetic_hot_tech']) if isinstance(row['synthetic_hot_tech'], str) else []
        for tech in hot_tech:
            row_features[f"HOT_{tech}"] = 1.0

        base_tech = json.loads(row['synthetic_base_tech']) if isinstance(row['synthetic_base_tech'], str) else []
        for tech in base_tech:
            row_features[f"BASE_{tech}"] = 1.0

        env_hierarchy = json.loads(row['synthetic_env_hierarchy']) if isinstance(row['synthetic_env_hierarchy'],
                                                                                 str) else {}
        for wa_id, wa_data in env_hierarchy.items():
            dwas = wa_data.get('execution_dwas', {})
            if not dwas:
                row_features[f"path_WA_{wa_id}_orphan"] = 1.0
            else:
                for dwa_id, task_data in dwas.items():
                    core_tasks = task_data.get('core_tasks', [])
                    supp_tasks = task_data.get('supplemental_tasks', [])

                    if not core_tasks and not supp_tasks:
                        row_features[f"path_WA_{wa_id}_DWA_{dwa_id}_orphan"] = 1.0
                    else:
                        for task_id in core_tasks:
                            row_features[f"path_WA_{wa_id}_DWA_{dwa_id}_core_{task_id}"] = 1.0
                        for task_id in supp_tasks:
                            row_features[f"path_WA_{wa_id}_DWA_{dwa_id}_supp_{task_id}"] = 1.0

        X_features.append(row_features)

    print("\nConstructing the Sparse Matrix aligned to the Master Columns...")
    col_to_idx = {col: i for i, col in enumerate(master_columns)}

    sparse_rows, sparse_cols, sparse_data = [], [], []
    for r_idx, row_dict in enumerate(tqdm(X_features, desc="Generating Matrix Coordinates")):
        for col_name, val in row_dict.items():
            if col_name in col_to_idx:
                sparse_rows.append(r_idx)
                sparse_cols.append(col_to_idx[col_name])
                sparse_data.append(val)

    X_sparse = sp.csr_matrix(
        (sparse_data, (sparse_rows, sparse_cols)),
        shape=(len(X_features), len(master_columns)),
        dtype='float32'
    )

    print("Wrapping into Pandas Sparse DataFrame (Zero Memory Allocation)...")
    X_matrix = pd.DataFrame.sparse.from_spmatrix(X_sparse, columns=master_columns)

    print("Encoding Target Labels for XGBoost...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw_targets)

    y_vector = pd.Series(y_encoded, name='target_class')
    w_vector = pd.Series(w_scores, name='sample_weight')

    print(f"-> X Matrix Shape: {X_matrix.shape}")
    print(f"-> Target Classes: {len(label_encoder.classes_)} unique jobs encoded")

    return X_matrix, y_vector, w_vector, label_encoder


if __name__ == "__main__":
    BENCHMARK_JSON = '../../../data_factory/datasets/benchmark_cleaned_dataset.json'
    SKILL_POOL = '../../../data_factory/artifacts/skill_pool.json'
    TECH_POOL = '../../../data_factory/artifacts/tech_pool.json'
    SYNTHETIC_CSV = '../../../data_factory/datasets/synthetic_training_dataset.csv'

    os.makedirs(EXPORT_DIR, exist_ok=True)

    # 1. Build Master Features
    fixed_columns = generate_master_feature_blueprint(BENCHMARK_JSON, SKILL_POOL, TECH_POOL)

    # 2. Build the FULL Matrix
    X_full, y_full, w_full, label_encoder = build_xgboost_training_matrix(SYNTHETIC_CSV, fixed_columns)

    # 3. SPLIT DATA HERE
    print(f"\nSplitting Dataset (Total: {X_full.shape[0]} rows) into 80% Train / 20% Test...")
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X_full, y_full, w_full, test_size=0.2, random_state=42
    )

    print(f" -> Training Set: {X_train.shape[0]} profiles")
    print(f" -> Testing Set:  {X_test.shape[0]} profiles")

    print(f"\nSaving generated assets to '{EXPORT_DIR}'...")

    # Save Blueprints
    with open(os.path.join(EXPORT_DIR, 'model_feature_columns.json'), 'w') as f:
        json.dump(fixed_columns, f)
    joblib.dump(label_encoder, os.path.join(EXPORT_DIR, 'label_encoder.joblib'))

    # Save Training Set
    X_train.to_pickle(os.path.join(EXPORT_DIR, 'X_train.pkl'))
    y_train.to_pickle(os.path.join(EXPORT_DIR, 'y_train.pkl'))
    w_train.to_pickle(os.path.join(EXPORT_DIR, 'w_train.pkl'))

    # Save Evaluation/Test Set
    X_test.to_pickle(os.path.join(EXPORT_DIR, 'X_test.pkl'))
    y_test.to_pickle(os.path.join(EXPORT_DIR, 'y_test.pkl'))
    w_test.to_pickle(os.path.join(EXPORT_DIR, 'w_test.pkl'))

    print("All preprocessing complete and train/test splits saved successfully!")
