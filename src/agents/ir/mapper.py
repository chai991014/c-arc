import os
import json
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# Load environment variables
load_dotenv()


class IRMapper:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the semantic search engine.
        Using all-MiniLM-L6-v2 because it is 5x faster than mpnet with 95% of the accuracy.
        """
        self.artifact_dir = os.getenv("ARTIFACT_DIR")
        if not self.artifact_dir:
            raise ValueError("ARTIFACT_DIR environment variable is not set. Check your .env file.")

        print(f"🧠 Loading local embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

        # 1. Load the raw JSON data
        self.skills_data = self._load_json("skill_pool.json")
        self.activities_data = self._load_json("work_activities_pool.json")

        # 2. Extract strings to embed
        self.skill_texts = self.skills_data
        self.activity_texts = self.activities_data

        # 3. Load or compute embeddings (The Caching Mechanism)
        print("⚡ Verifying vector cache...")
        self.skill_embeddings = self._get_embeddings(self.skill_texts, "skill_embeddings.npy")
        self.activity_embeddings = self._get_embeddings(self.activity_texts, "activity_embeddings.npy")
        print("✅ Mapper initialization complete.")

    def _load_json(self, filename: str) -> list:
        path = os.path.join(self.artifact_dir, filename)
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_embeddings(self, texts: list, cache_filename: str):
        """Loads embeddings from disk if they exist, otherwise computes and saves them."""
        cache_path = os.path.join(self.artifact_dir, cache_filename)

        if os.path.exists(cache_path):
            return np.load(cache_path)
        else:
            print(f"   ↳ Cache not found for {cache_filename}. Generating vectors (this takes a moment)...")
            # encode() returns a numpy array by default
            embeddings = self.model.encode(texts, show_progress_bar=True)
            np.save(cache_path, embeddings)
            return embeddings

    def map_skill(self, raw_entity: str, threshold: float = 0.65) -> dict:
        """
        Takes a raw string (e.g., 'React JS') and returns the exact O*NET skill ID.
        If the confidence is below the threshold, it rejects the mapping.
        """
        if not raw_entity:
            return None

        # Convert user string to vector
        entity_vec = self.model.encode(raw_entity)

        # Calculate Cosine Similarity against all 8,800 skills simultaneously
        cosine_scores = util.cos_sim(entity_vec, self.skill_embeddings)[0]

        # Find the highest score
        best_idx = np.argmax(cosine_scores).item()
        best_score = cosine_scores[best_idx].item()

        if best_score >= threshold:
            best_match = self.skill_texts[best_idx]
            return {
                "onet_id": "N/A",  # Your current JSON doesn't have IDs yet
                "onet_name": best_match,
                "confidence": round(best_score, 3)
            }

        # If no match meets the threshold, return None to prevent hallucinated mappings
        return None
