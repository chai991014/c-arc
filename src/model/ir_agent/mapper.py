import sqlite3
import json
import logging
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


class IRMapper:
    def __init__(self, db_path: str = "../data_factory/onet.db", pool_dir: str = "../data_factory/artifacts/"):
        self.db_path = db_path
        logger.info("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

        self.skill_ids = []
        self.skill_texts = []

        # 1. Load the JSON artifacts safely handling both Dicts and Lists
        try:
            file_path = f"{pool_dir}skill_pool.json"
            with open(file_path, "r") as f:
                raw_pool = json.load(f)

                if isinstance(raw_pool, list):
                    # Handles the [{"id": "...", "name": "..."}] format
                    self.skill_ids = [item.get("id", "") for item in raw_pool]
                    self.skill_texts = [item.get("name", item.get("title", "")) for item in raw_pool]
                elif isinstance(raw_pool, dict):
                    # Handles the {"id": "name"} format
                    self.skill_ids = list(raw_pool.keys())
                    self.skill_texts = list(raw_pool.values())

            logger.info(f"Successfully loaded {len(self.skill_ids)} skills into VRAM.")
        except FileNotFoundError:
            logger.warning(f"Could not find {file_path}. Initializing empty pool.")

        # 2. Pre-compute embeddings for semantic search
        if self.skill_texts:
            logger.info("Encoding SKILL_POOL to VRAM for fast semantic search...")
            self.skill_embeddings = self.embed_model.encode(self.skill_texts, convert_to_tensor=True)
        else:
            self.skill_embeddings = None

    def ground_phrase(self, raw_text: str, entity_type: str = "skill") -> dict:
        """
        Step 1: Vector search against the JSON pool to find the ID.
        Step 2: SQL query against onet.db using that ID.
        """
        if entity_type != "skill" or self.skill_embeddings is None:
            return {"id": None, "context": None, "score": "N/A (Empty Pool)"}

        # Step 1: Semantic Vector Search
        query_embedding = self.embed_model.encode(raw_text, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.skill_embeddings, top_k=1)[0]

        if not hits:
            return {"id": None, "context": None, "score": "N/A (No Hits)"}

        best_match_idx = hits[0]['corpus_id']
        best_score = float(hits[0]['score'])
        matched_id = self.skill_ids[best_match_idx]
        matched_text = self.skill_texts[best_match_idx]

        # Return the score even if it fails the threshold so we can debug it!
        if best_score < 0.70:
            return {
                "id": None,
                "matched_text": f"[FAILED THRESHOLD] {matched_text}",
                "score": round(best_score, 3),
                "context": "N/A"
            }

        # Step 2: Relational SQL Lookup
        context = self._fetch_db_context(matched_id)

        return {
            "id": matched_id,
            "matched_text": matched_text,
            "score": round(best_score, 3),
            "context": context
        }

    def _fetch_db_context(self, element_id: str) -> str:
        """Fetches the official element name or description from O*NET SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Strategy 1: Check if it's a Technology Skill
            if str(element_id).startswith("TECH-"):
                # Your JSON pool formats tech IDs as "TECH-[Skill Name]"
                # Since the name is built into the ID, we can safely extract it here
                conn.close()
                return element_id.replace("TECH-", "")

            # Strategy 2: Standard Skills Lookup (for IDs like 2.B.3.e)
            # Your ingest_excel_table.py created a 'skills' table, not content_model_reference
            query = "SELECT element_name FROM skills WHERE element_id = ? LIMIT 1"

            cursor.execute(query, (element_id,))
            row = cursor.fetchone()
            conn.close()

            return row[0] if row else "Context unavailable"

        except sqlite3.Error as e:
            logger.error(f"Database query failed for {element_id}: {e}")
            return "Context unavailable"
