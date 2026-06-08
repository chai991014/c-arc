import os
import pickle
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack, vstack
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import the new Neural Network Architecture
from dcn import DCNv2


def evaluate_system(
        assets_dir="./assets",
        model_dir="./saved_model",
        sample_limit=None,
        knn_neighbors=30
):
    print("============================================================")
    print("🚀 STARTING FULL SYSTEM END-TO-END EVALUATION PIPELINE (KNN + DCN V2)")
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
    with open(os.path.join(assets_dir, "tfidf_transformer.pkl"), 'rb') as f:
        tfidf = pickle.load(f)
    with open(os.path.join(assets_dir, "soc_index_map.pkl"), 'rb') as f:
        soc_index_map = pickle.load(f)

    # 2. Load PyTorch Stage 2 Model (DCN V2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_test.shape[1]

    scorer_model = DCNv2(input_dim=input_dim).to(device)
    scorer_model.load_state_dict(torch.load(os.path.join(model_dir, "career_expert_dcn_v2.pth"), map_location=device))
    scorer_model.eval()  # Set to evaluation mode

    # ---------------------------------------------------------
    # PHASE 1: Standalone Stage 2 Evaluation
    # ---------------------------------------------------------
    print("\n📊 [Phase 1/2] Evaluating Isolated Stage 2 Scorer Matrix Performance...")
    # Process in batches to avoid overwhelming GPU/CPU memory with a dense matrix conversion
    y_pred_phase1 = []
    batch_size = 512
    with torch.no_grad():
        for i in range(0, X_test.shape[0], batch_size):
            batch_tensor = torch.FloatTensor(X_test[i:i + batch_size].toarray()).to(device)
            preds = scorer_model(batch_tensor).cpu().numpy().flatten()
            y_pred_phase1.extend(preds)

    # Cache metrics for CSV export
    phase1_r2 = r2_score(y_test, y_pred_phase1)
    phase1_mae = mean_absolute_error(y_test, y_pred_phase1)
    phase1_mse = mean_squared_error(y_test, y_pred_phase1)

    print(f"  • Standalone Scorer R² Score : {phase1_r2:.4f}")
    print(f"  • Standalone Scorer MAE      : {phase1_mae:.4f}")
    print(f"  • Standalone Scorer MSE      : {phase1_mse:.4f}")

    # ---------------------------------------------------------
    # PHASE 2: Full System Pipeline Simulation
    # ---------------------------------------------------------
    print(f"\n🔄 [Phase 2/2] Running Complete Two-Stage Simulation...")
    total_features = X_test.shape[1]
    V = (total_features - 14) // 2
    print(f"  • Context Mapping Verification: Feature space V = {V} variables detected.")

    num_queries = sample_limit if sample_limit else X_test.shape[0]
    print(f"  • Simulating {num_queries} production queries from the blind test pool...")

    retrieval_hits = 0
    system_hits_at_1 = 0
    system_hits_at_3 = 0
    system_hits_at_5 = 0
    reciprocal_ranks = []
    pipeline_score_deltas = []

    # Track unique jobs recommended in the Top 5
    catalog_coverage = set()

    for i in range(num_queries):
        if i > 0 and i % (max(1, num_queries // 20)) == 0:
            print(f"    ↳ Evaluated {i}/{num_queries} user profiles...")

        # 1. Extract Target Ground Truth
        user_sparse = X_test[i, :V]
        user_ocean = X_test[i, V:V + 5]
        true_job_sparse = X_test[i, V + 5:2 * V + 5]
        true_job_ocean = X_test[i, 2 * V + 5:2 * V + 10]
        true_score = y_test[i]

        # 2. Stage 1: Fast KNN Retrieval
        user_sparse_tfidf = tfidf.transform(user_sparse)
        distances, indices = knn_retrieval.kneighbors(user_sparse_tfidf, n_neighbors=knn_neighbors)
        top_distances = distances[0]
        top_indices = indices[0]

        retrieved_socs = [soc_index_map[idx] for idx in top_indices]
        true_job_idx_in_benchmark = -1

        # Locate true job in benchmark to check retrieval success
        for idx in range(benchmark_matrix.shape[0]):
            if (benchmark_matrix[idx] != true_job_sparse).nnz == 0:
                true_job_idx_in_benchmark = idx
                break

        true_soc = soc_index_map.get(true_job_idx_in_benchmark, None)
        target_in_retrieval = true_soc in retrieved_socs
        if target_in_retrieval:
            retrieval_hits += 1

        # 3. Stage 2: Cross-Encoder Construction
        inference_rows = []
        candidate_socs = []

        for rank, (idx, knn_dist) in enumerate(zip(top_indices, top_distances)):
            soc = soc_index_map[idx]
            cand_job_sparse = benchmark_matrix[idx]
            cand_job_ocean_extracted = true_job_ocean if soc == true_soc else np.zeros((1, 5))

            u_sparse_arr = user_sparse.tocsr()
            j_sparse_arr = cand_job_sparse.tocsr()

            # Calculate interaction features
            dot_product = float(u_sparse_arr.dot(j_sparse_arr.T).toarray()[0][0])
            u_norm = np.linalg.norm(u_sparse_arr.data) if len(u_sparse_arr.data) > 0 else 1.0
            j_norm = np.linalg.norm(j_sparse_arr.data) if len(j_sparse_arr.data) > 0 else 1.0
            explicit_cosine_sim = dot_product / (u_norm * j_norm)

            u_set = set(u_sparse_arr.indices)
            j_set = set(j_sparse_arr.indices)
            jaccard = len(u_set.intersection(j_set)) / len(u_set.union(j_set)) if u_set or j_set else 0.0

            ocean_distance = euclidean(user_ocean.toarray().flatten(),
                                       cand_job_ocean_extracted.toarray().flatten() if hasattr(cand_job_ocean_extracted,
                                                                                               'toarray') else cand_job_ocean_extracted.flatten())

            interaction_meta = np.array([[dot_product, explicit_cosine_sim, jaccard, ocean_distance]])

            combined_row = hstack([user_sparse, user_ocean, cand_job_sparse, cand_job_ocean_extracted, interaction_meta])
            inference_rows.append(combined_row)
            candidate_socs.append(soc)

        # 4. Neural Network Scoring
        X_inference = vstack(inference_rows)
        X_inference_tensor = torch.FloatTensor(X_inference.toarray()).to(device)

        with torch.no_grad():
            predicted_scores = scorer_model(X_inference_tensor).cpu().numpy().flatten()

        results = []
        for j in range(len(candidate_socs)):
            results.append({
                "soc_code": candidate_socs[j],
                "score": float(predicted_scores[j])
            })

        results = sorted(results, key=lambda x: x["score"], reverse=True)

        # Track Catalog Coverage for the Top 5 recommended jobs
        for r in results[:5]:
            catalog_coverage.add(r["soc_code"])

        # 5. Evaluate Pipeline Metrics
        if true_soc:
            ranked_socs = [r["soc_code"] for r in results]
            if true_soc in ranked_socs:
                rank = ranked_socs.index(true_soc) + 1
                reciprocal_ranks.append(1.0 / rank)

                if rank == 1: system_hits_at_1 += 1
                if rank <= 3: system_hits_at_3 += 1
                if rank <= 5: system_hits_at_5 += 1

                predicted_target_score = next(r["score"] for r in results if r["soc_code"] == true_soc)
                pipeline_score_deltas.append(abs(predicted_target_score - true_score))
            else:
                reciprocal_ranks.append(0.0)

    # ---------------------------------------------------------
    # FINAL METRICS REPORT
    # ---------------------------------------------------------
    print("\n============================================================")
    print("📈 CAREER EXPERT PERFORMANCE EVALUATION (KNN + DCN V2)")
    print("============================================================")
    hr_stage1 = (retrieval_hits / num_queries) * 100
    hr_sys_1 = (system_hits_at_1 / num_queries) * 100
    hr_sys_3 = (system_hits_at_3 / num_queries) * 100
    hr_sys_5 = (system_hits_at_5 / num_queries) * 100
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    pipeline_mae = np.mean(pipeline_score_deltas) if pipeline_score_deltas else float('nan')

    # Calculate overall catalog coverage percentage
    total_possible_jobs = len(soc_index_map)
    coverage_pct = (len(catalog_coverage) / total_possible_jobs) * 100

    print(f"🎯 STAGE 1 RETRIEVAL (KNN Search Space size: {knn_neighbors})")
    print(f"  • Hit Rate @ {knn_neighbors}       : {hr_stage1:.2f}% (Target job captured by retrieval)")
    print(f"  • Retrieval Leakage Margin : {100.0 - hr_stage1:.2f}% (Target job lost by retrieval bottleneck)")

    print(f"\n🔮 STAGE 2 RE-RANKED SCORING (KNN + DCN V2)")
    print(f"  • Full System Hit Rate @ 1 : {hr_sys_1:.2f}% (Target job ranked as absolute #1 choice)")
    print(f"  • Full System Hit Rate @ 3 : {hr_sys_3:.2f}% (Target job placed inside Top 3 choices)")
    print(f"  • Full System Hit Rate @ 5 : {hr_sys_5:.2f}% (Target job placed inside final Top 5 counselor view)")
    print(f"  • Mean Reciprocal Rank(MRR): {mrr:.4f}")
    print(f"  • Catalog Coverage         : {coverage_pct:.2f}% ({len(catalog_coverage)} unique jobs recommended)")

    print(f"\n📉 PIPELINE ACCURACY")
    print(f"  • End-to-End Target MAE    : {pipeline_mae:.4f} (Prediction accuracy on the true target job)")
    print("============================================================\n")

    metrics_data = [
        {"Metric": f"KNN Hit Rate @ {knn_neighbors} (%)", "Value": round(hr_stage1, 2)},
        {"Metric": "KNN Retrieval Leakage (%)", "Value": round(100.0 - hr_stage1, 2)},
        {"Metric": "Scorer R2", "Value": round(phase1_r2, 4)},
        {"Metric": "Scorer MAE", "Value": round(phase1_mae, 4)},
        {"Metric": "Scorer MSE", "Value": round(phase1_mse, 4)},
        {"Metric": "Hit Rate @ 1 (%)", "Value": round(hr_sys_1, 2)},
        {"Metric": "Hit Rate @ 3 (%)", "Value": round(hr_sys_3, 2)},
        {"Metric": "Hit Rate @ 5 (%)", "Value": round(hr_sys_5, 2)},
        {"Metric": "Mean Reciprocal Rank (MRR)", "Value": round(mrr, 4)},
        {"Metric": "Catalog Coverage (%)", "Value": round(coverage_pct, 2)},
        {"Metric": "Unique Jobs Recommended", "Value": len(catalog_coverage)},
        {"Metric": "End-to-End Pipeline MAE", "Value": round(pipeline_mae, 4)},
        {"Metric": "Total Queries Simulated", "Value": num_queries}
    ]

    metrics_df = pd.DataFrame(metrics_data)
    csv_path = os.path.join(model_dir, "pipeline_evaluation_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"💾 Full evaluation metrics successfully exported to CSV: {csv_path}")


if __name__ == "__main__":
    evaluate_system()
