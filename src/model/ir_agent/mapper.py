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

        # Dictionary to hold the 3 distinct mapping pools
        self.pools = {
            "skill": {"ids": [], "texts": [], "embeddings": None},
            "task": {"ids": [], "texts": [], "embeddings": None},
            "dwa": {"ids": [], "texts": [], "embeddings": None}
        }

        self._load_pool("skill", f"{pool_dir}skill_pool.json")
        self._load_pool("task", f"{pool_dir}task_pool.json")
        self._load_pool("dwa", f"{pool_dir}dwa_pool.json")

    def _load_pool(self, pool_name: str, file_path: str):
        """Loads a JSON artifact and encodes it into VRAM."""
        try:
            with open(file_path, "r") as f:
                raw_pool = json.load(f)

                if isinstance(raw_pool, list):
                    self.pools[pool_name]["ids"] = [item.get("id", "") for item in raw_pool]
                    self.pools[pool_name]["texts"] = [item.get("name", item.get("title", "")) for item in raw_pool]

            logger.info(f"Successfully loaded {len(self.pools[pool_name]['ids'])} items for {pool_name}.")

            if self.pools[pool_name]["texts"]:
                logger.info(f"Encoding {pool_name.upper()} pool to VRAM...")
                self.pools[pool_name]["embeddings"] = self.embed_model.encode(
                    self.pools[pool_name]["texts"], convert_to_tensor=True
                )
        except FileNotFoundError:
            logger.warning(f"Could not find {file_path}. Initializing empty pool for {pool_name}.")

    def ground_phrase(self, raw_text: str, entity_type: str) -> dict:
        """Vector search against the specified JSON pool, followed by SQL lookup."""
        pool = self.pools.get(entity_type)
        if not pool or pool["embeddings"] is None:
            return {"id": None, "context": None, "score": "N/A (Empty/Invalid Pool)"}

        query_embedding = self.embed_model.encode(raw_text, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, pool["embeddings"], top_k=1)[0]

        if not hits:
            return {"id": None, "context": None, "score": "N/A (No Hits)"}

        best_match_idx = hits[0]['corpus_id']
        best_score = float(hits[0]['score'])
        matched_id = pool["ids"][best_match_idx]
        matched_text = pool["texts"][best_match_idx]

        # Return the score even if it fails the threshold so we can debug it!
        if best_score < 0.70:
            return {
                "id": None,
                "matched_text": f"[FAILED THRESHOLD] {matched_text}",
                "score": round(best_score, 3),
                "context": "N/A"
            }

        context = self._fetch_db_context(matched_id, entity_type)

        return {
            "id": matched_id,
            "matched_text": matched_text,
            "score": round(best_score, 3),
            "context": context
        }

    def _fetch_db_context(self, element_id: str, entity_type: str) -> str:
        """Fetches the official element name or description from the appropriate O*NET table."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Strategy 1: Check if it's a Technology Skill
            if str(element_id).startswith("TECH-"):
                # Your JSON pool formats tech IDs as "TECH-[Skill Name]"
                # Since the name is built into the ID, we can safely extract it here
                conn.close()
                return str(element_id).replace("TECH-", "")

            # Dynamic SQL routing based on entity type
            if entity_type == "skill":
                cursor.execute("SELECT element_name FROM skills WHERE element_id = ? LIMIT 1", (element_id,))
            elif entity_type == "task":
                cursor.execute("SELECT task FROM task_statements WHERE task_id = ? LIMIT 1", (element_id,))
            elif entity_type == "dwa":
                cursor.execute("SELECT dwa_title FROM dwa_reference WHERE dwa_id = ? LIMIT 1", (element_id,))
            else:
                conn.close()
                return "Unknown entity type"

            row = cursor.fetchone()
            conn.close()

            return row[0] if row else "Context unavailable"

        except sqlite3.Error as e:
            logger.error(f"Database query failed for {element_id}: {e}")
            return "Context unavailable"
