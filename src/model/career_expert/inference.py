import xgboost as xgb
import joblib
import json
import numpy as np
import sqlite3
from scipy import sparse


class CareerExpert:
    def __init__(self,
                 model_path="model/career_expert/saved_model/career_expert_v1.json",
                 encoder_path="model/career_expert/saved_model/label_encoder.pkl",
                 task_classes_path="model/career_expert/processed_data/task_classes.json",
                 dwa_classes_path="model/career_expert/processed_data/dwa_classes.json"):
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        self.le = joblib.load(encoder_path)

        with open(task_classes_path, 'r') as f: self.task_classes = json.load(f)
        with open(dwa_classes_path, 'r') as f: self.dwa_classes = json.load(f)

    def predict(self, user_tasks, user_dwas, ocean_vector):
        # 1. Vectorize input
        task_vec = [1 if t in user_tasks else 0 for t in self.task_classes]
        dwa_vec = [1 if d in user_dwas else 0 for d in self.dwa_classes]
        ocean_vec = [ocean_vector.get(t, 50.0) for t in
                     ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']]

        # 2. Predict probabilities
        input_data = np.array(task_vec + dwa_vec + ocean_vec).reshape(1, -1)
        dmatrix = xgb.DMatrix(input_data)
        probs = self.model.predict(dmatrix)[0]  # Array of 1,016 probabilities

        # 3. Get top 3 indices and their scores
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
        "./saved_model/career_expert_v1.json",
        "./saved_model/label_encoder.pkl",
        "./processed_data/task_classes.json",
        "./processed_data/dwa_classes.json"
    )

    # Example: User is highly Open and Extraverted with no specific technical skills (Cold Start)
    cold_start_profile = {"openness": 50, "conscientiousness": 50, "extraversion": 50, "agreeableness": 50,
                          "neuroticism": 50}

    recommendations = expert.predict([], [], cold_start_profile)

    print("Top 3 Career Recommendations:")
    for rec in recommendations:
        print(f"SOC: {rec['soc_code']} | Confidence: {rec['probability']}%")
