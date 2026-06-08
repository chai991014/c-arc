import os
import pickle
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import hstack, vstack
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_system(
        assets_dir="./assets",
        benchmark_csv="../../../data_factory/datasets/benchmark_cleaned_dataset.csv",
        sample_limit=500,  # Set to None to run on the entire blind test set
        knn_neighbors=15  # Mirror the current production retrieval depth
):
    print("============================================================")
    print("🚀 STARTING FULL SYSTEM END-TO-END EVALUATION PIPELINE")
    print("============================================================")

    # 1. Load Datasets and System Assets
    print("[+] Loading test datasets and trained components...")
    with open(os.path.join(assets_dir, "X_test.pkl"), 'rb') as f:
        X_test = pickle.load(f)
    with open(os.path.join(assets_dir, "y_test.pkl"), 'rb') as f:
        y_test = pickle.load(f)

    knn_retrieval = joblib.load(os.path.join(assets_dir, "knn_retrieval_index.joblib"))
    with open(os.path.join(assets_dir, "benchmark_matrix.pkl"), 'rb') as f:
        benchmark_matrix = pickle.load(f)
    with open(os.path.join(assets_dir, "soc_index_map.pkl"), 'rb') as f:
        soc_map = pickle.load(f)

    print(f"X_test matrix shape : {X_test.shape}")
    print(f"y_test vector shape : {y_test.shape}")

    scorer_model = xgb.XGBRegressor()
    scorer_model.load_model(os.path.join(assets_dir, "career_expert_xgboost_regressor.json"))

    # Load metadata for title resolution
    df_bench = pd.read_csv(benchmark_csv)
    job_meta = {}
    for _, row in df_bench.iterrows():
        job_meta[row['soc_code']] = {
            'title': row['job_title'],
            'ocean': np.array([row['O'], row['C'], row['E'], row['A'], row['N']]).reshape(1, -1)
        }

    # 2. Standalone Stage 2 Scorer Evaluation
    print("\n📊 [Phase 1/2] Evaluating Isolated Stage 2 Scorer Matrix Performance...")
    y_pred_standalone = scorer_model.predict(X_test)
    standalone_mse = mean_squared_error(y_test, y_pred_standalone)
    standalone_mae = mean_absolute_error(y_test, y_pred_standalone)
    standalone_r2 = r2_score(y_test, y_pred_standalone)

    print(f"  • Standalone Scorer R² Score : {standalone_r2:.4f}")
    print(f"  • Standalone Scorer MAE      : {standalone_mae:.4f}")
    print(f"  • Standalone Scorer MSE      : {standalone_mse:.4f}")

    # 3. Dynamic End-to-End System Evaluation
    print("\n🔄 [Phase 2/2] Running Complete Two-Stage Simulation...")

    # Calculate vocabulary size (V) based on the asymmetric hstack layout:
    # [User Sparse (V)] + [User OCEAN (5)] + [Job Sparse (V)] + [Job OCEAN (5)] = 2V + 10
    total_features = X_test.shape[1]
    V = (total_features - 14) // 2
    print(f"  • Context Mapping Verification: Feature space V = {V} variables detected.")

    # Build a reverse lookup index using a localized exact NN model to map the job sub-matrix back to its true SOC index
    job_identifier = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='brute')
    job_identifier.fit(benchmark_matrix)

    # Determine execution sample size
    num_queries = X_test.shape[0] if sample_limit is None else min(sample_limit, X_test.shape[0])
    print(f"  • Simulating {num_queries} production queries from the blind test pool...")

    # Tracking metrics
    stage1_hits = 0
    system_hits_at_1 = 0
    system_hits_at_3 = 0
    system_hits_at_5 = 0
    reciprocal_ranks = []
    pipeline_score_deltas = []

    for i in range(num_queries):
        # Slice out the User profile segments from the dataset row
        user_sparse = X_test[i, 0:V]
        user_ocean = X_test[i, V:V + 5].toarray()  # Ensure dense representation for hstack later

        # Slice out the Job profile segment to find out which job this row was generated against
        job_sparse = X_test[i, V + 5:2 * V + 5]

        # Identify the true SOC code index for this test sample
        _, matched_idx = job_identifier.kneighbors(job_sparse)
        true_job_idx = matched_idx[0][0]
        true_soc_code = soc_map[true_job_idx]
        ground_truth_score = float(y_test[i])

        # --- STEP 1: Simulate Stage 1 Retrieval ---
        distances, indices = knn_retrieval.kneighbors(user_sparse, n_neighbors=knn_neighbors)
        retrieved_indices = indices[0].tolist()
        retrieved_distances = distances[0].tolist()

        is_stage1_hit = true_job_idx in retrieved_indices
        if is_stage1_hit:
            stage1_hits += 1

        # --- STEP 2: Simulate Stage 2 Scoring ---
        inference_rows = []
        candidate_socs = []

        for idx, knn_dist in zip(retrieved_indices, retrieved_distances):
            candidate_soc = soc_map[idx]
            cand_job_sparse = benchmark_matrix[idx]
            cand_job_ocean = job_meta[candidate_soc]['ocean']

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
            candidate_socs.append(candidate_soc)

        # Process scoring if Stage 1 successfully pulled candidate profiles
        if inference_rows:
            X_inference = vstack(inference_rows).tocsr()
            predicted_scores = scorer_model.predict(X_inference)

            # Package and sort recommendations by the continuous capability score
            rankings = []
            for j, soc in enumerate(candidate_socs):
                rankings.append({
                    "soc": soc,
                    "score": float(predicted_scores[j])
                })
            rankings = sorted(rankings, key=lambda x: x["score"], reverse=True)

            # Locate the final rank position of the ground truth job
            final_rank = -1
            predicted_target_score = None
            for rank_idx, item in enumerate(rankings):
                if item["soc"] == true_soc_code:
                    final_rank = rank_idx + 1  # 1-indexed
                    predicted_target_score = item["score"]
                    break

            # Collate ranking metrics if the target survived Stage 1 retrieval
            if final_rank != -1:
                reciprocal_ranks.append(1.0 / final_rank)
                pipeline_score_deltas.append(abs(ground_truth_score - predicted_target_score))

                if final_rank == 1:
                    system_hits_at_1 += 1
                if final_rank <= 3:
                    system_hits_at_3 += 1
                if final_rank <= 5:
                    system_hits_at_5 += 1
            else:
                # Stage 1 Retrieval missed the target entirely; rank is effectively infinity
                reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(0.0)

        # Print a progress indicator
        if (i + 1) % max(1, num_queries // 5) == 0 or (i + 1) == num_queries:
            print(f"    ↳ Evaluated {i + 1}/{num_queries} user profiles...")

    # 4. Compile and Display Report Cards
    print("\n" + "=" * 60)
    print("📈 FINAL PIPELINE SYSTEM REPORT CARD")
    print("=" * 60)

    hr_stage1 = (stage1_hits / num_queries) * 100
    hr_sys_1 = (system_hits_at_1 / num_queries) * 100
    hr_sys_3 = (system_hits_at_3 / num_queries) * 100
    hr_sys_5 = (system_hits_at_5 / num_queries) * 100
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    pipeline_mae = np.mean(pipeline_score_deltas) if pipeline_score_deltas else float('nan')

    print(f"🎯 STAGE 1 RETRIEVAL (KNN Search Space size: {knn_neighbors})")
    print(f"  • Hit Rate @ {knn_neighbors}       : {hr_stage1:.2f}% (Target job captured by retrieval)")
    print(f"  • Retrieval Leakage Margin : {100.0 - hr_stage1:.2f}% (Target job lost by retrieval bottleneck)")

    print(f"\n🔮 STAGE 2 RE-RANKED FULL PIPELINE (KNN + XGBoost Cross-Encoder)")
    print(f"  • Full System Hit Rate @ 1 : {hr_sys_1:.2f}% (Target job ranked as absolute #1 choice)")
    print(f"  • Full System Hit Rate @ 3 : {hr_sys_3:.2f}% (Target job placed inside Top 3 choices)")
    print(f"  • Full System Hit Rate @ 5 : {hr_sys_5:.2f}% (Target job placed inside final Top 5 counselor view)")
    print(f"  • Mean Reciprocal Rank(MRR): {mrr:.4f}")

    print(f"\n📉 PIPELINE ACCURACY")
    print(f"  • End-to-End Target MAE    : {pipeline_mae:.4f} (Prediction accuracy on the true target job)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Point paths to your specific project layout
    evaluate_system(
        assets_dir="./assets",
        benchmark_csv="../../../data_factory/datasets/benchmark_cleaned_dataset.csv",
        sample_limit=None
    )
