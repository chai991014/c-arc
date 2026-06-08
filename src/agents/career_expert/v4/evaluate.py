import pandas as pd
import numpy as np
import os
import ast
from tqdm import tqdm
from inference import CareerExpert


def safe_parse(val):
    if isinstance(val, list): return val
    try:
        return ast.literal_eval(val)
    except:
        return []


def evaluate_full_space_ranking(
        test_csv_path="./assets/test_dataset.csv",
        sample_limit=None
):
    print("🚀 Booting Production DCNv2 Inference Engine for Ranking Evaluation...")
    expert = CareerExpert()

    # Load the test dataset
    df = pd.read_csv(test_csv_path)
    if sample_limit and len(df) > sample_limit:
        df = df.sample(n=sample_limit, random_state=42)

    hits_at_1, hits_at_3, hits_at_5 = 0, 0, 0
    reciprocal_ranks = []
    catalog_coverage = set()

    print(f"\n📊 Evaluating Ranking Metrics across {len(df)} simulated users...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        true_target_soc = row['target_soc_code']

        # 1. Reconstruct User Profile Dictionary mimicking Frontend
        master_profile = {
            "skills": safe_parse(row.get('synthetic_core_skills', '[]')),
            "tech_skills": safe_parse(row.get('synthetic_hot_tech', '[]')) + safe_parse(
                row.get('synthetic_base_tech', '[]')),
            "work_activities": safe_parse(row.get('synthetic_env_hierarchy', '[]')),
            "tasks": [], "dwas": []
        }
        user_ocean = {
            "O": row.get('O', 0.5), "C": row.get('C', 0.5), "E": row.get('E', 0.5),
            "A": row.get('A', 0.5), "N": row.get('N', 0.5)
        }

        # 2. Run Full-Space Prediction (Scores against all 900+ jobs)
        # Suppress prints from inference to keep tqdm clean
        top_jobs = expert.predict(master_profile, user_ocean, verbose=False)

        # 3. Track Catalog Coverage (Which unique jobs made it into Top 5?)
        for job in top_jobs[:5]:
            catalog_coverage.add(job['soc_code'])

        # 4. Check for Hits and calculate MRR
        ranked_socs = [job['soc_code'] for job in top_jobs]

        if true_target_soc in ranked_socs:
            # +1 because index is 0-based, rank is 1-based
            rank = ranked_socs.index(true_target_soc) + 1

            if rank == 1: hits_at_1 += 1
            if rank <= 3: hits_at_3 += 1
            if rank <= 5: hits_at_5 += 1

            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    # 5. Calculate Final Percentages
    N = len(df)
    hr_1 = (hits_at_1 / N) * 100
    hr_3 = (hits_at_3 / N) * 100
    hr_5 = (hits_at_5 / N) * 100
    mrr = np.mean(reciprocal_ranks)
    coverage_pct = (len(catalog_coverage) / len(expert.soc_map)) * 100

    print("\n============================================================")
    print("🎯 FULL-SPACE RANKING METRICS (DCNv2)")
    print("============================================================")
    print(f"  • Hit Rate @ 1               : {hr_1:.2f}%")
    print(f"  • Hit Rate @ 3               : {hr_3:.2f}%")
    print(f"  • Hit Rate @ 5               : {hr_5:.2f}%")
    print(f"  • Mean Reciprocal Rank (MRR) : {mrr:.4f}")
    print(f"  • Catalog Coverage           : {coverage_pct:.2f}% ({len(catalog_coverage)} unique jobs recommended)")
    print("============================================================\n")

    # 6. Export to CSV
    metrics_df = pd.DataFrame([
        {"Metric": "Hit Rate @ 1 (%)", "Value": round(hr_1, 2)},
        {"Metric": "Hit Rate @ 3 (%)", "Value": round(hr_3, 2)},
        {"Metric": "Hit Rate @ 5 (%)", "Value": round(hr_5, 2)},
        {"Metric": "Mean Reciprocal Rank (MRR)", "Value": round(mrr, 4)},
        {"Metric": "Catalog Coverage (%)", "Value": round(coverage_pct, 2)},
        {"Metric": "Unique Jobs Recommended", "Value": len(catalog_coverage)},
        {"Metric": "Users Evaluated", "Value": N}
    ])

    csv_path = "./saved_model/ranking_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"💾 Ranking metrics exported to CSV: {csv_path}")


if __name__ == "__main__":
    evaluate_full_space_ranking()
