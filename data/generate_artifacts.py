import sqlite3
import pandas as pd
import json
import os

DB_PATH = "./onet.db"
ARTIFACT_DIR = "./artifacts"


def create_skill_pool(conn):
    print("⏳ Extracting Master Skill Pool (The Skeleton)...")

    # 1. Get functional skills WITH their O*NET IDs
    # We use 'element_id' as the unique key for O*NET math
    skills_df = pd.read_sql_query(
        "SELECT DISTINCT element_id as id, element_name as name FROM skills", conn
    )

    # 2. Get technology tools
    # Tech skills often use 'commodity_code', but we'll use the name as the ID
    # for tools since they don't have personality distributions in O*NET.
    tech_df = pd.read_sql_query(
        "SELECT DISTINCT example as name FROM technology_skills", conn
    )
    tech_df['id'] = "TECH-" + tech_df['name']  # Assign a tech-prefix ID

    # 3. Combine into a list of dictionaries
    skills_list = skills_df.to_dict('records')
    tech_list = tech_df.to_dict('records')

    master_pool = skills_list + tech_list

    # Save to JSON as a list of dictionaries
    output_path = os.path.join(ARTIFACT_DIR, "skill_pool.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(master_pool, f, indent=4)

    print(f"   ↳ ✅ Success: Saved {len(master_pool)} identified skills to {output_path}")


def create_ocean_benchmarks(conn):
    print("⏳ Executing OCEAN Crosswalk (The Soul)...")

    # Read the Work Styles data (O*NET 30.2 uses 'WI' for Work Styles Impact)
    query = """
            SELECT onet_soc_code, element_name, data_value
            FROM work_styles
            WHERE scale_id = 'WI' \
            """
    df = pd.read_sql_query(query, conn)

    # The Updated O*NET 30.2 Crosswalk Dictionary (21 Traits instead of 16)
    crosswalk = {
        'Openness': ['Innovation', 'Intellectual Curiosity', 'Tolerance for Ambiguity'],
        'Conscientiousness': ['Achievement Orientation', 'Initiative', 'Perseverance', 'Cautiousness',
                              'Attention to Detail', 'Dependability', 'Integrity'],
        'Extraversion': ['Leadership Orientation', 'Optimism', 'Social Orientation'],
        'Agreeableness': ['Humility', 'Sincerity', 'Empathy', 'Cooperation'],
        # Neuroticism is inverted stability
        'Stability_Traits': ['Adaptability', 'Self-Confidence', 'Stress Tolerance', 'Self-Control']
    }

    ocean_dict = {}
    grouped = df.groupby('onet_soc_code')

    for soc, group in grouped:
        trait_scores = dict(zip(group['element_name'], group['data_value']))

        # Helper to safely average scores
        def get_avg(trait_list):
            scores = [trait_scores.get(t) for t in trait_list if t in trait_scores and pd.notna(trait_scores.get(t))]
            return sum(scores) / len(scores) if scores else 0

        # Helper to convert O*NET -1.5 to 3.0 impact scale into a 0-100 score
        def scale_to_100(val):
            scaled = ((val + 1.5) / 4.5) * 100
            return max(0, min(100, scaled))

        o_score = scale_to_100(get_avg(crosswalk['Openness']))
        c_score = scale_to_100(get_avg(crosswalk['Conscientiousness']))
        e_score = scale_to_100(get_avg(crosswalk['Extraversion']))
        a_score = scale_to_100(get_avg(crosswalk['Agreeableness']))

        # Calculate Neuroticism by inverting the stability average
        n_score = 100 - scale_to_100(get_avg(crosswalk['Stability_Traits']))

        ocean_dict[soc] = {
            "O": round(o_score, 2),
            "C": round(c_score, 2),
            "E": round(e_score, 2),
            "A": round(a_score, 2),
            "N": round(n_score, 2)
        }

    output_path = os.path.join(ARTIFACT_DIR, "soc_ocean_benchmarks.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ocean_dict, f, indent=4)

    print(f"   ↳ ✅ Success: Generated OCEAN benchmarks for {len(ocean_dict)} SOC codes to {output_path}")


def extract_work_activities(conn):
    print("⏳ Extracting Work Activities (The Intent Dictionary)...")

    # Grab the unique Work Activity names
    query = "SELECT DISTINCT element_name FROM work_activities WHERE element_name IS NOT NULL"
    df = pd.read_sql_query(query, conn)

    activities_list = sorted(df['element_name'].tolist())

    output_path = os.path.join(ARTIFACT_DIR, "work_activities_pool.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(activities_list, f, indent=4)

    print(f"   ↳ ✅ Success: Saved {len(activities_list)} Work Activities to {output_path}")


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    if not os.path.exists(DB_PATH):
        print(f"❌ Cannot find database at {DB_PATH}. Did you run the ingestion script?")
        return

    conn = sqlite3.connect(DB_PATH)

    create_skill_pool(conn)
    print("-" * 40)
    create_ocean_benchmarks(conn)
    print("-" * 40)
    extract_work_activities(conn)

    conn.close()
    print("\n🎉 Phase 1 is officially 100% COMPLETE.")


if __name__ == '__main__':
    main()
