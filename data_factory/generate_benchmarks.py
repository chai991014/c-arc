import pandas as pd
import json
import os
import numpy as np
import sqlite3

# Configuration and File Paths
ARTIFACT_DIR = "./artifacts"
OUTPUT_JSON_PATH = "./datasets/benchmark_dataset.json"
OUTPUT_CSV_PATH = "./datasets/benchmark_dataset.csv"
OUTPUT_CLEANED_JSON_PATH = "./datasets/benchmark_cleaned_dataset.json"
OUTPUT_CLEANED_CSV_PATH = "./datasets/benchmark_cleaned_dataset.csv"
DB_PATH = "./onet.db"


def create_intermediate_pools(df_wa, df_dwa_ref, df_tasks, df_skills, df_tech):
    """Generates the semantic search artifact pools."""
    print("Generating Intermediate Semantic Search Pools...")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    wa_pool = df_wa[['Element ID', 'Element Name']].drop_duplicates().rename(
        columns={'Element ID': 'id', 'Element Name': 'name'}).to_dict('records')
    with open(os.path.join(ARTIFACT_DIR, "wa_pool.json"), 'w', encoding='utf-8') as f:
        json.dump(wa_pool, f, indent=4)

    dwa_pool = df_dwa_ref[['DWA ID', 'DWA Title']].drop_duplicates().rename(
        columns={'DWA ID': 'id', 'DWA Title': 'name'}).to_dict('records')
    with open(os.path.join(ARTIFACT_DIR, "dwa_pool.json"), 'w', encoding='utf-8') as f:
        json.dump(dwa_pool, f, indent=4)

    task_pool = df_tasks[['Task ID', 'Task']].drop_duplicates().rename(
        columns={'Task ID': 'id', 'Task': 'name'}).to_dict('records')
    with open(os.path.join(ARTIFACT_DIR, "task_pool.json"), 'w', encoding='utf-8') as f:
        json.dump(task_pool, f, indent=4)

    skill_pool = df_skills[['Element ID', 'Element Name']].drop_duplicates().rename(
        columns={'Element ID': 'id', 'Element Name': 'name'}).to_dict('records')
    with open(os.path.join(ARTIFACT_DIR, "skill_pool.json"), 'w', encoding='utf-8') as f:
        json.dump(skill_pool, f, indent=4)

    tech_pool = df_tech[['Example']].drop_duplicates()
    tech_pool['id'] = "TECH-" + tech_pool['Example']
    tech_pool = tech_pool.rename(columns={'Example': 'name'})[['id', 'name']].to_dict('records')
    with open(os.path.join(ARTIFACT_DIR, "tech_pool.json"), 'w', encoding='utf-8') as f:
        json.dump(tech_pool, f, indent=4)


def calculate_relevance(df):
    """Applies the mathematical Gatekeeper Filter: sqrt(IM * LV) >= 3.5"""
    df_im = df[df['Scale ID'] == 'IM'][['O*NET-SOC Code', 'Element ID', 'Element Name', 'Data Value']]
    df_lv = df[df['Scale ID'] == 'LV'][['O*NET-SOC Code', 'Element ID', 'Data Value']]

    merged = pd.merge(df_im, df_lv, on=['O*NET-SOC Code', 'Element ID'], suffixes=('_IM', '_LV'))
    merged['relevance'] = np.sqrt(merged['Data Value_IM'] * merged['Data Value_LV'])

    return merged[merged['relevance'] >= 3.5]


def compute_ocean_scores(df_styles):
    """Calculates continuous OCEAN percentiles from O*NET Work Styles."""
    crosswalk = {
        'O': ['Innovation', 'Intellectual Curiosity', 'Tolerance for Ambiguity'],
        'C': ['Achievement Orientation', 'Initiative', 'Perseverance', 'Cautiousness', 'Attention to Detail',
              'Dependability', 'Integrity'],
        'E': ['Leadership Orientation', 'Optimism', 'Social Orientation'],
        'A': ['Humility', 'Sincerity', 'Empathy', 'Cooperation'],
        'Stability': ['Adaptability', 'Self-Confidence', 'Stress Tolerance', 'Self-Control']
    }

    df_im = df_styles[df_styles['Scale ID'] == 'WI']
    ocean_dict = {}

    for soc, group in df_im.groupby('O*NET-SOC Code'):
        trait_scores = dict(zip(group['Element Name'], group['Data Value']))

        def get_scaled_avg(traits):
            scores = [trait_scores.get(t) for t in traits if pd.notna(trait_scores.get(t))]
            if not scores: return 0.5
            avg_score = sum(scores) / len(scores)
            return max(0.0, min(1.0, (avg_score + 3) / 6))

        ocean_dict[soc] = {
            "O": round(get_scaled_avg(crosswalk['O']), 4),
            "C": round(get_scaled_avg(crosswalk['C']), 4),
            "E": round(get_scaled_avg(crosswalk['E']), 4),
            "A": round(get_scaled_avg(crosswalk['A']), 4),
            "N": round(1.0 - get_scaled_avg(crosswalk['Stability']), 4)
        }
    return ocean_dict


def build_benchmark_dataset():
    print("Connecting to O*NET SQLite Database...")
    conn = sqlite3.connect(DB_PATH)

    # FIXED: Replaced 'o_net_soc_code' with 'onet_soc_code' to align with the real DB columns
    df_occ = pd.read_sql_query('SELECT onet_soc_code AS "O*NET-SOC Code", title AS "Title" FROM occupation_data', conn)
    df_skills = pd.read_sql_query(
        'SELECT onet_soc_code AS "O*NET-SOC Code", element_id AS "Element ID", element_name AS "Element Name", scale_id AS "Scale ID", data_value AS "Data Value" FROM skills',
        conn)
    df_tech = pd.read_sql_query(
        'SELECT onet_soc_code AS "O*NET-SOC Code", example AS "Example", hot_technology AS "Hot Technology", in_demand AS "In Demand" FROM technology_skills',
        conn)
    df_wa = pd.read_sql_query(
        'SELECT onet_soc_code AS "O*NET-SOC Code", element_id AS "Element ID", element_name AS "Element Name", scale_id AS "Scale ID", data_value AS "Data Value" FROM work_activities',
        conn)
    df_styles = pd.read_sql_query(
        'SELECT onet_soc_code AS "O*NET-SOC Code", element_name AS "Element Name", scale_id AS "Scale ID", data_value AS "Data Value" FROM work_styles',
        conn)
    df_tasks = pd.read_sql_query(
        'SELECT task_id AS "Task ID", task AS "Task", task_type AS "Task Type" FROM task_statements', conn)
    df_t2d = pd.read_sql_query(
        'SELECT onet_soc_code AS "O*NET-SOC Code", dwa_id AS "DWA ID", task_id AS "Task ID" FROM tasks_to_dwas', conn)
    df_dwa_ref = pd.read_sql_query(
        'SELECT dwa_id AS "DWA ID", dwa_title AS "DWA Title", element_id AS "Element ID" FROM dwa_reference', conn)

    conn.close()

    create_intermediate_pools(df_wa, df_dwa_ref, df_tasks, df_skills, df_tech)

    print("Applying Gatekeeper Relevance Filters...")
    relevant_skills = calculate_relevance(df_skills)
    relevant_was = calculate_relevance(df_wa)

    print("Executing Psychometric OCEAN Crosswalk...")
    ocean_benchmarks = compute_ocean_scores(df_styles)

    dwa_to_wa_map = dict(zip(df_dwa_ref['DWA ID'], df_dwa_ref['Element ID']))
    task_type_map = dict(zip(df_tasks['Task ID'], df_tasks['Task Type']))

    benchmark_json = []
    benchmark_csv_rows = []
    benchmark_cleaned_json = []
    benchmark_cleaned_csv_rows = []

    print("Constructing Hierarchical Profiles...")
    for _, occ_row in df_occ.iterrows():
        soc = occ_row['O*NET-SOC Code']
        title = occ_row['Title']

        soc_skills = relevant_skills[relevant_skills['O*NET-SOC Code'] == soc]['Element ID'].tolist()

        soc_tech = df_tech[df_tech['O*NET-SOC Code'] == soc]
        hot_tech = soc_tech[soc_tech['Hot Technology'] == 'Y']['Example'].apply(lambda x: f"TECH-{x}").unique().tolist()
        base_tech = soc_tech[(soc_tech['Hot Technology'] != 'Y') & (soc_tech['In Demand'] == 'Y')]['Example'].apply(
            lambda x: f"TECH-{x}").unique().tolist()

        soc_was = relevant_was[relevant_was['O*NET-SOC Code'] == soc]
        soc_t2d = df_t2d[df_t2d['O*NET-SOC Code'] == soc]

        env_hierarchy = {}
        for _, wa_row in soc_was.iterrows():
            wa_id = wa_row['Element ID']
            env_hierarchy[wa_id] = {
                "activity_name": wa_row['Element Name'],
                "relevance_score": round(wa_row['relevance'], 2),
                "execution_dwas": {}
            }

        for _, t2d_row in soc_t2d.iterrows():
            dwa_id = t2d_row['DWA ID']
            task_id = t2d_row['Task ID']
            wa_id = dwa_to_wa_map.get(dwa_id)

            if wa_id in env_hierarchy:
                if dwa_id not in env_hierarchy[wa_id]["execution_dwas"]:
                    env_hierarchy[wa_id]["execution_dwas"][dwa_id] = {"core_tasks": [], "supplemental_tasks": []}

                t_type = task_type_map.get(task_id, 'Supplemental')
                if t_type == 'Core':
                    if task_id not in env_hierarchy[wa_id]["execution_dwas"][dwa_id]["core_tasks"]:
                        env_hierarchy[wa_id]["execution_dwas"][dwa_id]["core_tasks"].append(task_id)
                else:
                    if task_id not in env_hierarchy[wa_id]["execution_dwas"][dwa_id]["supplemental_tasks"]:
                        env_hierarchy[wa_id]["execution_dwas"][dwa_id]["supplemental_tasks"].append(task_id)

        soc_ocean = ocean_benchmarks.get(soc, {"O": 0.0, "C": 0.0, "E": 0.0, "A": 0.0, "N": 0.0})

        json_node = {
            "inputs": {
                "capabilities": {
                    "core_skills": soc_skills,
                    "technology_skills": {"hot_technologies": hot_tech, "baseline_tools": base_tech}
                },
                "environmental_hierarchy": env_hierarchy,
                "personality_ocean": soc_ocean
            },
            "target_output": {"soc_code": soc, "job_title": title}
        }
        benchmark_json.append(json_node)

        benchmark_csv_rows.append({
            "soc_code": soc,
            "job_title": title,
            "core_skills": json.dumps(soc_skills),
            "hot_tech_skills": json.dumps(hot_tech),
            "baseline_tech_skills": json.dumps(base_tech),
            "environmental_hierarchy": json.dumps(env_hierarchy),
            "O": soc_ocean["O"],
            "C": soc_ocean["C"],
            "E": soc_ocean["E"],
            "A": soc_ocean["A"],
            "N": soc_ocean["N"]
        })

        # Gatekeeper Filter for cleaned datasets
        if sum(soc_ocean.values()) == 0 or (not hot_tech and not base_tech) or not env_hierarchy:
            continue

        benchmark_cleaned_json.append(json_node)
        benchmark_cleaned_csv_rows.append({
            "soc_code": soc,
            "job_title": title,
            "core_skills": json.dumps(soc_skills),
            "hot_tech_skills": json.dumps(hot_tech),
            "baseline_tech_skills": json.dumps(base_tech),
            "environmental_hierarchy": json.dumps(env_hierarchy),
            "O": soc_ocean["O"],
            "C": soc_ocean["C"],
            "E": soc_ocean["E"],
            "A": soc_ocean["A"],
            "N": soc_ocean["N"]
        })

    # Export
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(benchmark_json, f, indent=4)
    with open(OUTPUT_CLEANED_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(benchmark_cleaned_json, f, indent=4)

    pd.DataFrame(benchmark_csv_rows).to_csv(OUTPUT_CSV_PATH, index=False)
    pd.DataFrame(benchmark_cleaned_csv_rows).to_csv(OUTPUT_CLEANED_CSV_PATH, index=False)
    print("Pipeline Execution Complete.")


if __name__ == "__main__":
    build_benchmark_dataset()
