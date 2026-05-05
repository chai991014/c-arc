import os
import json
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# Dynamic pathing to ensure .env and artifacts are always found
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# src/agents/ir/ -> src/agents/ -> src/ -> project_root/ (3 levels up)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

# Load .env using absolute path
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


class IRMapper:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.artifact_dir = os.path.join(PROJECT_ROOT, os.getenv("ARTIFACT_DIR", "data/artifacts"))

        print(f"🧠 Loading local embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

        # Load the JSON data
        self.skills_pool = self._load_json("skill_pool.json")

        # Determine if pool is a list of strings or list of dicts
        # We need a list of strings for the embedding model to read
        if isinstance(self.skills_pool[0], dict):
            self.skill_texts = [s['name'] for s in self.skills_pool]
        else:
            self.skill_texts = self.skills_pool

        print("⚡ Verifying vector cache...")
        self.skill_embeddings = self._get_embeddings(self.skill_texts, "skill_embeddings.npy")
        print("✅ Mapper initialization complete.")

    def _load_json(self, filename: str) -> list:
        path = os.path.join(self.artifact_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required artifact: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_embeddings(self, texts: list, cache_filename: str):
        cache_path = os.path.join(self.artifact_dir, cache_filename)
        if os.path.exists(cache_path):
            return np.load(cache_path)

        print(f"   ↳ Generating vectors for {cache_filename}...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        np.save(cache_path, embeddings)
        return embeddings

    def map_skill(self, raw_entity: str, threshold: float = 0.65) -> dict:
        """
        Returns a dictionary even on failure to prevent NoneType errors in callers.
        """
        if not raw_entity or not isinstance(raw_entity, str):
            return self._empty_match("N/A")

        entity_vec = self.model.encode(raw_entity)
        cosine_scores = util.cos_sim(entity_vec, self.skill_embeddings)[0]
        best_idx = np.argmax(cosine_scores).item()
        best_score = cosine_scores[best_idx].item()

        if best_score >= threshold:
            # Check if source data has IDs
            source_data = self.skills_pool[best_idx]

            # If the source is a dict, extract ID. If string, it stays N/A.
            onet_id = source_data.get('id', 'N/A') if isinstance(source_data, dict) else 'N/A'
            onet_name = source_data.get('name', source_data) if isinstance(source_data, dict) else source_data

            return {
                "onet_id": onet_id,
                "onet_name": onet_name,
                "confidence": round(best_score, 3)
            }

        return self._empty_match(raw_entity)

    def _empty_match(self, raw_entity: str) -> dict:
        """Standardized failure response."""
        return {
            "onet_id": "Unmapped",
            "onet_name": "Unmapped",
            "confidence": 0.0,
            "original": raw_entity
        }
