import sqlite3
import json
import os
import csv

DB_PATH = "../data/onet.db"
BENCHMARKS_PATH = "../data/artifacts/soc_ocean_benchmarks.json"
OUTPUT_JSON_PATH = "../data/synthetic/benchmark_dataset.json"
OUTPUT_CSV_PATH = "../data/synthetic/benchmark_dataset.csv"


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
        job_ocean = ocean_benchmarks.get(soc_code)

        # If the exact variant (e.g., .03) is missing, try to inherit from the base (.00)
        if job_ocean is None:
            base_soc = soc_code.split('.')[0] + '.00'
            job_ocean = ocean_benchmarks.get(base_soc, {
                "O": None,
                "C": None,
                "E": None,
                "A": None,
                "N": None
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
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_dataset, f, indent=4)

    print(f"Success! {len(benchmark_dataset)} Ground Truth records saved to {OUTPUT_JSON_PATH}")

    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write the flat header
        writer.writerow([
            "soc_code", "job_title",
            "perfect_tasks", "perfect_dwas",
            "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"
        ])

        # Write flattened rows
        for job in benchmark_dataset:
            writer.writerow([
                job["soc_code"],
                job["job_title"],
                str(job["perfect_tasks"]),  # Saves array as a stringified list like '["1", "2"]'
                str(job["perfect_dwas"]),
                job["base_ocean"]["O"],
                job["base_ocean"]["C"],
                job["base_ocean"]["E"],
                job["base_ocean"]["A"],
                job["base_ocean"]["N"]
            ])

    print(f"CSV saved to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    build_benchmarks()
