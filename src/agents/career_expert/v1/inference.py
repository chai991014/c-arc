import xgboost as xgb
import joblib
import json
import numpy as np
import sqlite3
from scipy import sparse


class CareerExpert:
    def __init__(self,
                 model_path="agents/career_expert/saved_model/career_expert_xgboost.json",
                 data_dir="agents/career_expert/processed_data"):

        self.model = xgb.Booster()
        self.model.load_model(model_path)
        self.le = joblib.load(data_dir + "/label_encoder.joblib")

        # Load the unified master feature blueprint
        with open(f"{data_dir}/model_feature_columns.json", 'r') as f:
            self.feature_columns = json.load(f)

    def predict(self, master_profile: dict, ocean_vector: dict):

        def sanitize(val):
            return str(val).replace('[', '').replace(']', '').replace('<', '')

        active_features = {}
        for k, v in ocean_vector.items():
            active_features[sanitize(k)] = v

        # 1. Map flat Skills and Tech
        for skill in master_profile.get("skills", []):
            active_features[f"SKILL_{sanitize(skill)}"] = 1.0
        for tech in master_profile.get("tech_skills", []):
            sanitized_tech = sanitize(tech)
            active_features[f"HOT_{sanitized_tech}"] = 1.0
            active_features[f"BASE_{sanitized_tech}"] = 1.0

        # 2. Reconstruct Environmental Hierarchy Paths from flat arrays
        user_tasks = {sanitize(t) for t in master_profile.get("tasks", [])}
        user_dwas = {sanitize(d) for d in master_profile.get("dwas", [])}
        user_was = {sanitize(w) for w in master_profile.get("work_activities", [])}

        active_was_with_children = set()
        active_dwas_with_children = set()

        # Step A: Map Tasks. Since we lack the job label, we must activate ALL valid
        # core/supp variations of this task found in the blueprint to let XGBoost evaluate both.
        for task in user_tasks:
            for col in self.feature_columns:
                if col.endswith(f"_core_{task}") or col.endswith(f"_supp_{task}"):
                    active_features[col] = 1.0

                    # Track parents to enforce mutual exclusivity (preventing orphan columns)
                    try:
                        parts = col.split("_DWA_")
                        wa_id = parts[0].replace("path_WA_", "")
                        active_was_with_children.add(wa_id)

                        dwa_id = parts[1].split("_core_")[0].split("_supp_")[0]
                        active_dwas_with_children.add(dwa_id)
                    except IndexError:
                        continue

        # Step B: Map DWAs. Only trigger orphan status if no child tasks activated it.
        for dwa in user_dwas:
            for col in self.feature_columns:
                if col.endswith(f"_DWA_{dwa}_orphan"):
                    # If this DWA already had tasks mapped in Step A, it is NOT an orphan.
                    if str(dwa) in active_dwas_with_children:
                        continue

                    active_features[col] = 1.0
                    try:
                        wa_id = col.split("_DWA_")[0].replace("path_WA_", "")
                        active_was_with_children.add(wa_id)
                    except IndexError:
                        continue

        # Step C: Map WAs. Only trigger orphan status if no child DWAs activated it.
        for wa in user_was:
            col_name = f"path_WA_{wa}_orphan"
            if col_name in self.feature_columns:
                if str(wa) in active_was_with_children:
                    continue
                active_features[col_name] = 1.0

        # 3. Convert to Sparse Array exactly matching the Blueprint length
        input_data = np.zeros((1, len(self.feature_columns)), dtype=np.float32)
        for i, col in enumerate(self.feature_columns):
            if col in active_features:
                input_data[0, i] = active_features[col]

        csr_input = sparse.csr_matrix(input_data)
        dmatrix = xgb.DMatrix(csr_input)
        probs = self.model.predict(dmatrix)[0]

        # Get top 3 indices and their scores
        top_3_idx = np.argsort(probs)[-3:][::-1]

        results = []
        for i in top_3_idx:
            results.append({
                "soc_code": self.le.inverse_transform([i])[0],
                "probability": round(float(probs[i]) * 100, 2)
            })

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


# --- Test Execution ---
if __name__ == "__main__":
    expert = CareerExpert(
        model_path="./saved_model/career_expert_xgboost.json",
        data_dir="./processed_data"
    )

    # Example: Cold Start Profile
    cold_start_profile = {"OCEAN_O": 0.5, "OCEAN_C": 0.5, "OCEAN_E": 0.5, "OCEAN_A": 0.5, "OCEAN_N": 0.5}

    # Pass an empty master_profile and the scaled ocean_vector
    recommendations = expert.predict({}, cold_start_profile)

    print("Top 3 Career Recommendations:")
    for rec in recommendations:
        print(f"SOC: {rec['soc_code']} | Confidence: {rec['probability']}%")
