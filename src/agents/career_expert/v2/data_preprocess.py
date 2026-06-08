import pandas as pd
import numpy as np
import json
import ast
import os
import pickle
import sqlite3
from scipy.sparse import hstack, vstack
from scipy.spatial.distance import euclidean
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split


# --- Configuration ---
DATA_PATH = "../../../data_factory/datasets/synthetic_training_dataset.csv"
DB_PATH = "../../../data_factory/onet.db"
EXPORT_DIR = "./assets"

os.makedirs(EXPORT_DIR, exist_ok=True)


# --- Helper Functions ---
def safe_parse(val):
    """Safely parses stringified JSON lists/dicts back into Python objects."""
    if pd.isna(val) or val is None:
        return []
    if isinstance(val, (list, dict)):
        return val
    try:
        return json.loads(val)
    except Exception:
        try:
            return ast.literal_eval(val)
        except Exception:
            return []


def extract_weighted_features(row, is_user=False, target_soc=None, skill_weights=None):
    """
    Asymmetric extraction:
    Jobs keep continuous weights (from DB & JSON). Users are strictly binary (1.0).
    """
    features = {}
    if skill_weights is None:
        skill_weights = {}

    # 1. Extract Core Skills (Applying DB Weights)
    core = safe_parse(row.get('synthetic_core_skills', '[]'))

    if isinstance(core, list):
        for skill_id in core:
            if is_user:
                features[f"SKILL_{skill_id}"] = 1.0  # User just checks the box
            else:
                # Look up the true weight from the DB! Default to 1.0 if missing.
                weight = skill_weights.get(target_soc, {}).get(str(skill_id), 1.0)
                features[f"SKILL_{skill_id}"] = float(weight)

    # 2. Extract Tech Skills (Default to 1.0)
    hot = safe_parse(row.get('synthetic_hot_tech', '[]'))
    base = safe_parse(row.get('synthetic_base_tech', '[]'))

    if isinstance(hot, list):
        for x in hot: features[f"TECH_HOT_{x}"] = 1.0
    if isinstance(base, list):
        for x in base: features[f"TECH_BASE_{x}"] = 1.0

    # 3. Extract Environmental Hierarchy (Applying JSON Relevance Scores)
    env = safe_parse(row.get('synthetic_env_hierarchy', '{}'))
    if isinstance(env, dict):
        for wa_id, wa_data in env.items():

            if is_user:
                features[f"WA_{wa_id}"] = 1.0
            else:
                features[f"WA_{wa_id}"] = float(wa_data.get('relevance_score', 1.0))

            for dwa_id, dwa_data_dict in wa_data.get('execution_dwas', {}).items():
                features[f"DWA_{dwa_id}"] = 1.0

                for t in dwa_data_dict.get('core_tasks', []):
                    features[f"TASK_CORE_{t}"] = 1.0
                for t in dwa_data_dict.get('supplemental_tasks', []):
                    features[f"TASK_SUPP_{t}"] = 1.0

    return features


# --- Main Preprocessing Pipeline ---
def build_pipeline_assets():
    print("🚀 Starting C-Arc Data Preprocessing Pipeline...")

    # 1. Load the unified synthetic dataset
    print("Loading synthetic dataset...")
    df_syn = pd.read_csv(DATA_PATH)

    # 2. Pre-fetch Ground Truth Weights from onet.db
    print("Pre-fetching core skill weights from onet.db...")
    conn = sqlite3.connect(DB_PATH)

    query = """
        SELECT 
            im.onet_soc_code, 
            im.element_id, 
            im.data_value AS IM, 
            lv.data_value AS LV 
        FROM skills im
        JOIN skills lv ON im.onet_soc_code = lv.onet_soc_code AND im.element_id = lv.element_id
        WHERE im.scale_id = 'IM' AND lv.scale_id = 'LV'
    """
    df_weights = pd.read_sql_query(query, conn)
    conn.close()

    # Calculate exact relevance: sqrt(IM * LV)
    df_weights['relevance'] = np.sqrt(df_weights['IM'] * df_weights['LV'])

    # Build fast lookup dictionary: { soc_code: { skill_id: relevance_score } }
    skill_weights = {}
    for _, row in df_weights.iterrows():
        soc = row['onet_soc_code']
        skill_id = str(row['element_id'])
        if soc not in skill_weights:
            skill_weights[soc] = {}
        skill_weights[soc][skill_id] = float(row['relevance'])

    # 3. Extract Benchmark Profiles (The 'SYN_0000' Blueprints)
    print("Isolating benchmark blueprints (SYN_0000)...")
    df_bench = df_syn[df_syn['profile_id'].str.endswith('SYN_0000')].copy()

    # Apply Asymmetric Extraction
    print("Extracting weighted benchmark taxonomies...")
    df_bench['weighted_features'] = df_bench.apply(
        lambda r: extract_weighted_features(r, is_user=False, target_soc=r['target_soc_code'],
                                            skill_weights=skill_weights),
        axis=1
    )

    print("Extracting binary user taxonomies...")
    df_syn['weighted_features'] = df_syn.apply(
        lambda r: extract_weighted_features(r, is_user=True, target_soc=r['target_soc_code'],
                                            skill_weights=skill_weights),
        axis=1
    )

    # 4. Fit the Universal Feature Space using DictVectorizer
    print("Building universal sparse matrix space...")
    vec = DictVectorizer(sparse=True)
    benchmark_matrix = vec.fit_transform(df_bench['weighted_features'])

    # Save mapping of row index to SOC code for fast retrieval
    soc_index_map = {pos_idx: row['target_soc_code'] for pos_idx, (_, row) in enumerate(df_bench.iterrows())}
    bench_ocean = df_bench[['O', 'C', 'E', 'A', 'N']].values

    # Create a fast lookup dictionary for concatenation
    benchmark_lookup = {}
    for pos_idx, (pd_idx, row) in enumerate(df_bench.iterrows()):
        benchmark_lookup[row['target_soc_code']] = {
            'features': benchmark_matrix[pos_idx],
            'ocean': bench_ocean[pos_idx].reshape(1, -1)
        }

    # 5. Transform Synthetic Profiles (The Training Data)
    print("Vectorizing synthetic user profiles...")
    user_sparse = vec.transform(df_syn['weighted_features'])
    user_ocean = df_syn[['O', 'C', 'E', 'A', 'N']].values

    # 6. Construct the Concatenated Matrix
    print("Assembling Cross-Encoder Training Matrix...")
    target_job_sparse_list = []
    target_job_ocean_list = []

    for soc in df_syn['target_soc_code']:
        bench_data = benchmark_lookup[soc]
        target_job_sparse_list.append(bench_data['features'])
        target_job_ocean_list.append(bench_data['ocean'])

    job_sparse = vstack(target_job_sparse_list)
    job_ocean = np.vstack(target_job_ocean_list)

    # # Concatenate: [User Sparse] + [User OCEAN] + [Job Sparse] + [Job OCEAN]
    # X_full = hstack([
    #     user_sparse,
    #     user_ocean,
    #     job_sparse,
    #     job_ocean
    # ]).tocsr()

    print("Calculating interaction features across full matrix...")
    u_sparse = user_sparse.tocsr()
    j_sparse = job_sparse.tocsr()

    # 1. Vectorized Row-wise Dot Product
    # .multiply() does element-wise math, .sum(axis=1) adds them up per row yielding an (N, 1) vector
    dot_product = np.asarray(u_sparse.multiply(j_sparse).sum(axis=1))

    # 2. Vectorized Cosine Similarity
    u_norm = np.asarray(np.sqrt(u_sparse.power(2).sum(axis=1)))
    j_norm = np.asarray(np.sqrt(j_sparse.power(2).sum(axis=1)))
    u_norm[u_norm == 0] = 1.0  # Prevent division by zero
    j_norm[j_norm == 0] = 1.0
    explicit_cosine_sim = dot_product / (u_norm * j_norm)

    # 3. Vectorized Jaccard Similarity (Intersection over Union)
    u_bool = (u_sparse > 0).astype(int)
    j_bool = (j_sparse > 0).astype(int)
    intersection = np.asarray(u_bool.multiply(j_bool).sum(axis=1))
    union = np.asarray(u_bool.sum(axis=1) + j_bool.sum(axis=1)) - intersection
    union[union == 0] = 1.0
    jaccard = intersection / union

    # 4. Vectorized OCEAN Euclidean Distance
    ocean_distance = np.sqrt(np.sum((user_ocean - job_ocean) ** 2, axis=1, keepdims=True))

    # 5. Create a dense matrix of interaction metrics (Shape: N rows x 4 columns)
    interaction_meta = np.hstack([dot_product, explicit_cosine_sim, jaccard, ocean_distance])

    # 6. Final concatenation for the whole dataset
    # Replace your old hstack[...] with this:
    X_full = hstack([u_sparse, user_ocean, j_sparse, job_ocean, interaction_meta])

    y_full = df_syn['label_score'].values

    print("Splitting dataset into Train (70%), Validation (15%), and Test (15%)...")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_full, y_full, test_size=0.30, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # 7. Export assets
    print("💾 Saving assets to disk...")

    with open(os.path.join(EXPORT_DIR, "benchmark_matrix.pkl"), 'wb') as f:
        pickle.dump(benchmark_matrix, f)
    with open(os.path.join(EXPORT_DIR, "soc_index_map.pkl"), 'wb') as f:
        pickle.dump(soc_index_map, f)
    with open(os.path.join(EXPORT_DIR, "feature_vectorizer.pkl"), 'wb') as f:
        pickle.dump(vec, f)  # Saved the DictVectorizer
    with open(os.path.join(EXPORT_DIR, "X_train.pkl"), 'wb') as f:
        pickle.dump(X_train, f)
    with open(os.path.join(EXPORT_DIR, "y_train.pkl"), 'wb') as f:
        pickle.dump(y_train, f)
    with open(os.path.join(EXPORT_DIR, "X_val.pkl"), 'wb') as f:
        pickle.dump(X_val, f)
    with open(os.path.join(EXPORT_DIR, "y_val.pkl"), 'wb') as f:
        pickle.dump(y_val, f)
    with open(os.path.join(EXPORT_DIR, "X_test.pkl"), 'wb') as f:
        pickle.dump(X_test, f)
    with open(os.path.join(EXPORT_DIR, "y_test.pkl"), 'wb') as f:
        pickle.dump(y_test, f)

    print(f"✅ Preprocessing complete!")
    print(f" -> Training Set:   {X_train.shape[0]} rows")
    print(f" -> Validation Set: {X_val.shape[0]} rows")
    print(f" -> Testing Set:    {X_test.shape[0]} rows")

if __name__ == "__main__":
    build_pipeline_assets()
