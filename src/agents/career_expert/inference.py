import joblib
import pickle
import numpy as np
import pandas as pd
import sqlite3
import torch
from agents.career_expert.dcn import DCNv2
from scipy.sparse import hstack, vstack
from scipy.spatial.distance import euclidean

# ASSETS_DIR = "./assets"
# MODEL_DIR = "./saved_model"
# BENCHMARK_DIR = "../../../data_factory/datasets/benchmark_cleaned_dataset.csv"
ASSETS_DIR = "./agents/career_expert/assets"
MODEL_DIR = "./agents/career_expert/saved_model"
BENCHMARK_DIR = "../data_factory/datasets/benchmark_cleaned_dataset.csv"


class CareerExpert:
    def __init__(self):

        print("Loading C-Arc Inference Engines...")
        # 1. Load Stage 1 (Retrieval)
        self.knn = joblib.load(f"{ASSETS_DIR}/knn_retrieval_index.joblib")
        with open(f"{ASSETS_DIR}/benchmark_matrix.pkl", 'rb') as f:
            self.benchmark_matrix = pickle.load(f)
        with open(f"{ASSETS_DIR}/soc_index_map.pkl", 'rb') as f:
            self.soc_map = pickle.load(f)

        # 2. Load Vectorizer
        with open(f"{ASSETS_DIR}/feature_vectorizer.pkl", 'rb') as f:
            self.vec = pickle.load(f)
        with open(f"{ASSETS_DIR}/tfidf_transformer.pkl", 'rb') as f:
            self.tfidf = pickle.load(f)

        # 3. Load Stage 2 (DCN V2 Scorer)
        print("Initializing Deep & Cross Network V2...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Calculate dynamic input width (15502 * 2 + 14 = 31018)
        input_dim = self.benchmark_matrix.shape[1] * 2 + 14

        self.model = DCNv2(input_dim=input_dim).to(self.device)
        self.model.load_state_dict(torch.load(f"{MODEL_DIR}/career_expert_dcn_v2.pth", map_location=self.device))
        self.model.eval()

        # 4. Build Job Meta Lookup (For Title and Job OCEAN scores)
        # We need the benchmark OCEAN scores to concatenate into the XGBoost row
        df_bench = pd.read_csv(BENCHMARK_DIR)
        self.job_meta = {}
        for _, row in df_bench.iterrows():
            self.job_meta[row['soc_code']] = {
                'title': row['job_title'],
                'ocean': np.array([row['O'], row['C'], row['E'], row['A'], row['N']]).reshape(1, -1)
            }

    def _parse_user_profile(self, master_profile):
        """Translates the fixed flat array input into the asymmetric DictVectorizer format."""
        features = {}

        # Helper to extract the raw ID, whether the frontend sends a string "4.A.1" or a dict {"id": "4.A.1"}
        def extract_id(item):
            return str(item.get("id", item)) if isinstance(item, dict) else str(item)

        # 1. Map Skills
        for skill in master_profile.get("skills", []):
            features[f"SKILL_{extract_id(skill)}"] = 1.0

        # 2. Map Tech Skills (Activating both; DictVectorizer will safely drop the invalid one)
        for tech in master_profile.get("tech_skills", []):
            tech_id = extract_id(tech)

            features[f"TECH_HOT_{tech_id}"] = 1.0
            features[f"TECH_BASE_{tech_id}"] = 1.0

        # 3. Map Environmental Hierarchy
        for wa in master_profile.get("work_activities", []):
            features[f"WA_{extract_id(wa)}"] = 1.0

        for dwa in master_profile.get("dwas", []):
            features[f"DWA_{extract_id(dwa)}"] = 1.0

        # 4. Map Tasks (Activating both; DictVectorizer will safely drop the invalid one)
        for task in master_profile.get("tasks", []):
            task_id = extract_id(task)
            features[f"TASK_CORE_{task_id}"] = 1.0
            features[f"TASK_SUPP_{task_id}"] = 1.0

        return features

    def predict(self, master_profile: dict, ocean_vector: dict):
        # 1. Format User Data
        user_features = self._parse_user_profile(master_profile)
        user_sparse = self.vec.transform([user_features])
        user_sparse_tfidf = self.tfidf.transform(user_sparse)

        print(f"  • Parsed Vector Features : {len(user_features)} active entries mapped to sparse space.")
        if len(user_features) > 0:
            print(f"    ↳ Active features: {list(user_features.keys())[:8]}...")

        user_ocean = np.array([[
            ocean_vector.get('O', 0.5),
            ocean_vector.get('C', 0.5),
            ocean_vector.get('E', 0.5),
            ocean_vector.get('A', 0.5),
            ocean_vector.get('N', 0.5)
        ]])
        print(f"  • Reconciled Vector OCEAN: {user_ocean.tolist()}")

        # 2. Stage 1: Retrieval (Get Top K closest blueprints)
        distances, indices = self.knn.kneighbors(user_sparse_tfidf, n_neighbors=30)
        top_indices = indices[0]
        top_distances = distances[0]

        # 3. Stage 2: Cross-Encoder Construction
        inference_rows = []
        candidate_socs = []

        print("\n🔮 STAGE 1 RETRIEVAL (KNN):")
        for rank, (idx, knn_dist) in enumerate(zip(top_indices, top_distances)):
            soc = self.soc_map[idx]
            cand_job_sparse = self.benchmark_matrix[idx]
            cand_job_ocean = self.job_meta[soc]['ocean']
            job_title = self.job_meta[soc]['title']

            print(f"  Rank {rank + 1:02d} | Cosine Dist: {knn_dist:.4f} | SOC: {soc} | Job: {job_title}")

            u_sparse_arr = user_sparse.tocsr()
            j_sparse_arr = cand_job_sparse.tocsr()

            # Interaction Features
            dot_product = float(u_sparse_arr.dot(j_sparse_arr.T).toarray()[0][0])
            u_norm = np.linalg.norm(u_sparse_arr.data) if len(u_sparse_arr.data) > 0 else 1.0
            j_norm = np.linalg.norm(j_sparse_arr.data) if len(j_sparse_arr.data) > 0 else 1.0
            explicit_cosine_sim = dot_product / (u_norm * j_norm)

            u_set = set(u_sparse_arr.indices)
            j_set = set(j_sparse_arr.indices)
            jaccard = len(u_set.intersection(j_set)) / len(u_set.union(j_set)) if u_set or j_set else 0.0

            ocean_distance = euclidean(user_ocean.flatten(), cand_job_ocean.flatten())

            interaction_meta = np.array([[dot_product, explicit_cosine_sim, jaccard, ocean_distance]])

            # Concatenate
            combined_row = hstack([user_sparse, user_ocean, cand_job_sparse, cand_job_ocean, interaction_meta])
            inference_rows.append(combined_row)
            candidate_socs.append(soc)

        # 4. Predict Absolute Capability Scores using Neural Network
        X_inference_sparse = vstack(inference_rows).tocsr()
        X_tensor = torch.FloatTensor(X_inference_sparse.toarray()).to(self.device)

        with torch.no_grad():
            # Pass through DCN V2 and bring results back to CPU
            predicted_scores = self.model(X_tensor).cpu().numpy().flatten()

        # 5. Format and Sort Results
        results = []
        for i in range(len(candidate_socs)):
            soc = candidate_socs[i]
            score = float(predicted_scores[i])
            score = max(0.0, min(1.0, score))
            results.append({
                "soc_code": soc,
                "job_title": self.job_meta[soc]['title'],
                "match_score": round(score * 100, 2)
            })

        # Sort by the regressor's continuous score in descending order
        results = sorted(results, key=lambda x: x["match_score"], reverse=True)

        print("\n🔮 STAGE 2 RE-RANKED (DCN V2 Scorer):")
        for rank, res in enumerate(results):
            print(
                f"  Rank {rank + 1:02d} | Match Score: {res['match_score']:6.2f}% | SOC: {res['soc_code']} | Job: {res['job_title']}")

        return results

    def get_soc_details(self, soc_code: str, db_path: str = "../data_factory/onet.db") -> dict:
        """Fetches the official job title and description from the O*NET database."""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT title, description FROM occupation_data WHERE onet_soc_code = ? LIMIT 1",
                           (soc_code,))
            row = cursor.fetchone()
            conn.close()

            if row:
                return {"title": row[0], "description": row[1]}
        except sqlite3.Error as e:
            print(f"[X] SQLite query failed for SOC {soc_code}: {e}")

        return {"title": f"O*NET Role {soc_code}", "description": "A highly recommended career path."}
