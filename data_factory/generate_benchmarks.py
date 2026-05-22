import sqlite3
import json
import os

DB_PATH = "../data/onet.db"
BENCHMARKS_PATH = "../data/artifacts/soc_ocean_benchmarks.json"
OUTPUT_PATH = "../data/synthetic/benchmark_dataset.json"


def fetch_occupations(cursor):
    cursor.execute("SELECT onet_soc_code, title FROM occupation_data")
    return cursor.fetchall()


def fetch_tasks(cursor, soc_code):
    cursor.execute("SELECT task_id FROM task_statements WHERE onet_soc_code = ?", (soc_code,))
    return [row[0] for row in cursor.fetchall()]


def fetch_dwas(cursor, soc_code):
    cursor.execute("SELECT dwa_id FROM tasks_to_dwas WHERE onet_soc_code = ?", (soc_code,))
    return [row[0] for row in cursor.fetchall()]


def build_benchmarks():
    print("Connecting to onet.db to build perfect baselines...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Load the base OCEAN scores
    with open(BENCHMARKS_PATH, 'r', encoding='utf-8') as f:
        ocean_benchmarks = json.load(f)

    occupations = fetch_occupations(cursor)
    benchmark_dataset = []

    print(f"Generating 100% perfect benchmarks for {len(occupations)} occupations...")

    for soc_code, title in occupations:
        # 1. Fetch ALL tasks and DWAs (No sampling)
        tasks = fetch_tasks(cursor, soc_code)
        dwas = fetch_dwas(cursor, soc_code)

        # 2. Get exact OCEAN averages
        job_ocean = ocean_benchmarks.get(soc_code, {
            "openness": 50.0, "conscientiousness": 50.0,
            "extraversion": 50.0, "agreeableness": 50.0, "neuroticism": 50.0
        })

        benchmark_dataset.append({
            "soc_code": soc_code,
            "job_title": title,
            "perfect_tasks": tasks,
            "perfect_dwas": dwas,
            "base_ocean": job_ocean
        })

    conn.close()

    # Save to the new synthetic data folder
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_dataset, f, indent=4)

    print(f"Success! {len(benchmark_dataset)} Ground Truth records saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_benchmarks()
