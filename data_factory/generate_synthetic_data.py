import pandas as pd
import numpy as np
import sqlite3
import random
import json
import ast


def calculate_dynamic_weights(soc_code, conn, hot_count, base_count, global_tech_median):
    """Calculates Method A normalized weights dynamically."""
    # 1. Core Skills Raw Score
    mu_core_df = pd.read_sql_query(
        "SELECT AVG(data_value) FROM skills WHERE onet_soc_code = ? AND scale_id = 'IM'",
        conn, params=[soc_code]
    )
    mu_core = mu_core_df.iloc[0, 0]
    if pd.isna(mu_core): mu_core = 3.0

    # 2. Environmental / Work Activities Raw Score
    mu_env_df = pd.read_sql_query(
        "SELECT AVG(data_value) FROM work_activities WHERE onet_soc_code = ? AND scale_id = 'IM'",
        conn, params=[soc_code]
    )
    mu_env = mu_env_df.iloc[0, 0]
    if pd.isna(mu_env): mu_env = 3.0

    # 3. Calculate Tech Density Scale Factor (theta) using the benchmark's true median
    total_tech_count = hot_count + base_count
    if global_tech_median > 0:
        theta = min(1.0, total_tech_count / global_tech_median)
    else:
        theta = 1.0 if total_tech_count > 0 else 0.0

    # 4. Distribute Core Importance using Relative Prevalence Ratio & Theta
    if total_tech_count > 0:
        mu_hot = mu_core * (hot_count / total_tech_count) * theta
        mu_base = mu_core * (base_count / total_tech_count) * theta
    else:
        mu_hot = 0.0
        mu_base = 0.0

    # Handle absolute floor constraints if a job has tech but evaluated near zero
    if hot_count > 0 and mu_hot == 0: mu_hot = 1.0
    if base_count > 0 and mu_base == 0: mu_base = 1.0

    # 5. Continuous Normalization Matrix
    total_raw = mu_core + mu_env + mu_hot + mu_base
    weights = {
        'core': mu_core / total_raw,
        'env': mu_env / total_raw,
        'hot': mu_hot / total_raw,
        'base': mu_base / total_raw
    }
    return weights


def stochastic_dropout_list(item_list, T):
    """Randomly retains items in a list based on Target Capability (T)."""
    if not isinstance(item_list, list) or len(item_list) == 0:
        return [], 0.0

    retained = [item for item in item_list if random.random() <= T]
    retention_rate = len(retained) / len(item_list) if len(item_list) > 0 else 0.0
    return retained, retention_rate


def stochastic_dropout_dict(env_dict, T):
    """Randomly retains nested dictionary items and calculates weighted retention."""
    if not isinstance(env_dict, dict) or len(env_dict) == 0:
        return {}, 0.0

    retained_dict = {}
    total_original_score = 0.0
    total_retained_score = 0.0

    for activity, payload in env_dict.items():
        activity_score = float(payload['relevance_score'])
        total_original_score += activity_score

        if random.random() <= T:
            retained_payload = payload.copy()
            retained_dwas = {}

            # Traverse and apply dropout to child DWAs
            for dwa_id, dwa_data in payload.get('execution_dwas', {}).items():
                if random.random() <= T:
                    retained_dwa_data = dwa_data.copy()

                    # Traverse and apply dropout to grandchild Tasks
                    core_tasks, _ = stochastic_dropout_list(dwa_data.get('core_tasks', []), T)
                    supp_tasks, _ = stochastic_dropout_list(dwa_data.get('supplemental_tasks', []), T)

                    retained_dwa_data['core_tasks'] = core_tasks
                    retained_dwa_data['supplemental_tasks'] = supp_tasks
                    retained_dwas[dwa_id] = retained_dwa_data

            retained_payload['execution_dwas'] = retained_dwas
            retained_dict[activity] = retained_payload
            total_retained_score += activity_score

    retention_rate = (total_retained_score / total_original_score) if total_original_score > 0 else 0.0
    return retained_dict, retention_rate


def inject_ocean_noise(ocean_scores, sigma=0.1):
    """Mutates OCEAN scores using Gaussian noise and calculates drift."""
    mutated_ocean = {}
    total_drift = 0.0

    for trait, benchmark_val in ocean_scores.items():
        noise = random.gauss(0, sigma)
        synthetic_val = max(0.0, min(1.0, benchmark_val + noise))
        mutated_ocean[trait] = round(synthetic_val, 4)
        total_drift += abs(synthetic_val - benchmark_val)

    avg_drift = total_drift / 5.0
    return mutated_ocean, avg_drift


def safe_parse_json_structures(val):
    """Ensures string lists and structures pass internal python literal parsing safely."""
    if pd.isna(val) or val is None:
        return [] if isinstance(val, list) else {}
    if isinstance(val, (list, dict)):
        return val
    try:
        return json.loads(val)
    except Exception:
        try:
            return ast.literal_eval(val)
        except Exception:
            return val


def generate_synthetic_dataset(benchmark_csv_path, output_csv_path, db_path, N_per_job=100):
    print("Loading benchmark dataset...")
    benchmark_df = pd.read_csv(benchmark_csv_path)

    print("Computing empirical tech density metrics from benchmark profiles...")
    total_tech_counts = []
    for _, row in benchmark_df.iterrows():
        hc = len(safe_parse_json_structures(row.get('hot_tech_skills', '[]')))
        bc = len(safe_parse_json_structures(row.get('baseline_tech_skills', '[]')))
        total_tech_counts.append(hc + bc)

    # Establish dynamic median from your exact cohort
    global_tech_median = float(np.median(total_tech_counts)) if total_tech_counts else 7.0
    print(f"   ↳ Empirical cohort tech median established at: {global_tech_median} tools.")

    print("Connecting to O*NET SQL Database...")
    conn = sqlite3.connect(db_path)

    synthetic_rows = []

    counts = {
        'Expert': int(N_per_job * 0.15),
        'Senior': int(N_per_job * 0.45),
        'Junior': int(N_per_job * 0.30),
        'Unqualified': int(N_per_job * 0.10)
    }
    counts['Senior'] += N_per_job - sum(counts.values())

    print(f"Generating synthetic profiles...")
    for index, row in benchmark_df.iterrows():
        soc_code = row['soc_code']
        job_title = row['job_title']

        core_skills = safe_parse_json_structures(row.get('core_skills', '[]'))
        hot_tech = safe_parse_json_structures(row.get('hot_tech_skills', '[]'))
        base_tech = safe_parse_json_structures(row.get('baseline_tech_skills', '[]'))
        env_hierarchy = safe_parse_json_structures(row.get('environmental_hierarchy', '{}'))

        W = calculate_dynamic_weights(
            soc_code=soc_code,
            conn=conn,
            hot_count=len(hot_tech),
            base_count=len(base_tech),
            global_tech_median=global_tech_median
        )

        ocean_benchmark = {
            'O': float(row.get('O', 0.5)),
            'C': float(row.get('C', 0.5)),
            'E': float(row.get('E', 0.5)),
            'A': float(row.get('A', 0.5)),
            'N': float(row.get('N', 0.5))
        }



        synthetic_rows.append({
            'profile_id': f"{soc_code}_SYN_0000",
            'target_soc_code': soc_code,
            'target_job_title': job_title,
            'implied_seniority': 'Benchmark Match',
            'target_T': 1.0,
            'synthetic_core_skills': json.dumps(core_skills),
            'synthetic_hot_tech': json.dumps(hot_tech),
            'synthetic_base_tech': json.dumps(base_tech),
            'synthetic_env_hierarchy': json.dumps(env_hierarchy),
            'O': ocean_benchmark['O'],
            'C': ocean_benchmark['C'],
            'E': ocean_benchmark['E'],
            'A': ocean_benchmark['A'],
            'N': ocean_benchmark['N'],
            'label_score': 1.00
        })

        profile_counter = 1
        for persona, count in counts.items():
            for _ in range(count):
                if persona == 'Expert':
                    T, sigma = random.uniform(0.90, 1.00), 0.05
                elif persona == 'Senior':
                    T, sigma = random.uniform(0.70, 0.89), 0.10
                elif persona == 'Junior':
                    T, sigma = random.uniform(0.50, 0.69), 0.15
                else:
                    T, sigma = random.uniform(0.30, 0.49), 0.25

                syn_core, r_core = stochastic_dropout_list(core_skills, T)
                syn_hot, r_hot = stochastic_dropout_list(hot_tech, T)
                syn_base, r_base = stochastic_dropout_list(base_tech, T)
                syn_env, r_env = stochastic_dropout_dict(env_hierarchy, T)

                syn_ocean, ocean_drift = inject_ocean_noise(ocean_benchmark, sigma)

                score_capabilities = (W['core'] * r_core) + (W['env'] * r_env) + (W['hot'] * r_hot) + (
                            W['base'] * r_base)
                final_score = max(0.0, min(1.0, score_capabilities - ocean_drift))

                synthetic_rows.append({
                    'profile_id': f"{soc_code}_SYN_{profile_counter:04d}",
                    'target_soc_code': soc_code,
                    'target_job_title': job_title,
                    'implied_seniority': persona,
                    'target_T': round(T, 4),
                    'synthetic_core_skills': json.dumps(syn_core),
                    'synthetic_hot_tech': json.dumps(syn_hot),
                    'synthetic_base_tech': json.dumps(syn_base),
                    'synthetic_env_hierarchy': json.dumps(syn_env),
                    'O': syn_ocean['O'],
                    'C': syn_ocean['C'],
                    'E': syn_ocean['E'],
                    'A': syn_ocean['A'],
                    'N': syn_ocean['N'],
                    'label_score': round(final_score, 4)
                })
                profile_counter += 1

    conn.close()

    output_df = pd.DataFrame(synthetic_rows)
    output_df.to_csv(output_csv_path, index=False)
    print(f"Synthetic dataset generation complete! Saved to {output_csv_path}")

    json_ready_rows = []
    for row in synthetic_rows:
        row_copy = row.copy()
        row_copy['synthetic_core_skills'] = safe_parse_json_structures(row['synthetic_core_skills'])
        row_copy['synthetic_hot_tech'] = safe_parse_json_structures(row['synthetic_hot_tech'])
        row_copy['synthetic_base_tech'] = safe_parse_json_structures(row['synthetic_base_tech'])
        row_copy['synthetic_env_hierarchy'] = safe_parse_json_structures(row['synthetic_env_hierarchy'])
        json_ready_rows.append(row_copy)

    output_json_path = output_csv_path.replace('.csv', '.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_ready_rows, f, indent=4)
    print(f"Structured JSON saved to {output_json_path}")


if __name__ == "__main__":
    BENCHMARK_INPUT = './datasets/benchmark_cleaned_dataset.csv'
    SYNTHETIC_OUTPUT = './datasets/synthetic_training_dataset.csv'
    DB_PATH = './onet.db'
    SAMPLES_PER_JOB = 100

    generate_synthetic_dataset(BENCHMARK_INPUT, SYNTHETIC_OUTPUT, DB_PATH, N_per_job=SAMPLES_PER_JOB)
