import json
import random
import os
import csv

INPUT_PATH = "../data/synthetic/benchmark_dataset.json"
OUTPUT_PATH = "../data/synthetic/training_dataset.json"
CSV_OUTPUT_PATH = "../data/synthetic/training_dataset.csv"

VARIATIONS_PER_JOB = 10
NOISE_STD_DEV = 10.0

def generate_augmented_dataset():
    print(f"Loading Ground Truth benchmarks from {INPUT_PATH}...")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        benchmarks = json.load(f)

    training_dataset = []

    print(f"Applying data augmentation (Skill Sampling + N(0,{NOISE_STD_DEV}) Personality Noise)...")
    print(f"Generating {VARIATIONS_PER_JOB} variations per job...")

    for job in benchmarks:
        soc = job['soc_code']
        perfect_tasks = job.get('perfect_tasks', [])
        perfect_dwas = job.get('perfect_dwas', [])
        base_ocean = job.get('base_ocean', {})

        for i in range(VARIATIONS_PER_JOB):
            # 1. Augment Technical Skills (Sample 60% to 80% randomly per candidate)
            task_sample_size = int(len(perfect_tasks) * random.uniform(0.6, 0.8))
            dwa_sample_size = int(len(perfect_dwas) * random.uniform(0.6, 0.8))

            sampled_tasks = random.sample(perfect_tasks, task_sample_size) if perfect_tasks else []
            sampled_dwas = random.sample(perfect_dwas, dwa_sample_size) if perfect_dwas else []

            synthetic_candidate = {
                "candidate_id": f"{soc}_var_{i+1}",
                "soc_code": soc,
                "job_title": job.get('job_title'),
                "synthetic_tasks": sampled_tasks,
                "synthetic_dwas": sampled_dwas
            }

            # 2. Augment Personality Scores (Add Gaussian Noise)
            noisy_ocean = {}
            for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
                mu = base_ocean.get(trait, 50.0)
                noisy_score = random.gauss(mu, NOISE_STD_DEV)
                noisy_score = max(0.0, min(100.0, noisy_score))
                noisy_ocean[trait] = round(noisy_score, 2)

            synthetic_candidate["ocean_vector"] = noisy_ocean
            training_dataset.append(synthetic_candidate)

    # Save the final ML training dataset
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(training_dataset, f, indent=4)

    print(f"Success! {len(training_dataset)} fully augmented records saved to {OUTPUT_PATH}")

    with open(CSV_OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write the flat header
        writer.writerow([
            "candidate_id", "soc_code", "job_title",
            "synthetic_tasks", "synthetic_dwas",
            "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"
        ])

        # Write flattened rows
        for candidate in training_dataset:
            writer.writerow([
                candidate["candidate_id"],
                candidate["soc_code"],
                candidate["job_title"],
                str(candidate["synthetic_tasks"]),
                str(candidate["synthetic_dwas"]),
                candidate["ocean_vector"]["openness"],
                candidate["ocean_vector"]["conscientiousness"],
                candidate["ocean_vector"]["extraversion"],
                candidate["ocean_vector"]["agreeableness"],
                candidate["ocean_vector"]["neuroticism"]
            ])

    print(f"CSV saved to {CSV_OUTPUT_PATH}")

if __name__ == "__main__":
    generate_augmented_dataset()
