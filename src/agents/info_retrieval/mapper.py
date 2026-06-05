import sqlite3
import json
import logging
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


class IRMapper:
    def __init__(self, llm_client, db_path: str = "../data_factory/onet.db", pool_dir: str = "../data_factory/artifacts/"):
        self.db_path = db_path
        logger.info("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

        self.client = llm_client

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
        """Global Hybrid RAG: Searches Skills, Tasks, and DWAs simultaneously."""

        all_candidates = []
        query_embedding = self.embed_model.encode(raw_text, convert_to_tensor=True)

        pools_to_search = ["skill", "task", "dwa"] if entity_type == "experience" else [entity_type]

        for pool_name in pools_to_search:
            pool = self.pools.get(pool_name)
            if not pool or pool["embeddings"] is None:
                continue

            hits = util.semantic_search(query_embedding, pool["embeddings"], top_k=5)[0]
            for hit in hits:
                idx = hit['corpus_id']
                all_candidates.append({
                    "id": pool["ids"][idx],
                    "text": pool["texts"][idx],
                    "score": round(float(hit['score']), 3),
                    "resolved_type": pool_name  # Track which pool this candidate came from
                })

        if not all_candidates:
            return {"id": None, "context": None, "score": "N/A (No Hits)", "resolved_type": None}

        all_candidates = sorted(all_candidates, key=lambda x: x['score'], reverse=True)[:10]

        candidates_str = "\n".join(
            [f"- ID: {c['id']} | Type: [{c['resolved_type'].upper()}] | Description: {c['text']}" for c in
             all_candidates])

        print(f"\n[+] DEBUG - IRMapper GLOBAL Vector Retrieval for '{raw_text}':")
        print(candidates_str)

        # 3. LLM Reranking
        prompt = (
            f"You are a strict O*NET taxonomy mapper.\n"
            f"The candidate described this professional experience: \"{raw_text}\"\n\n"
            f"Here are the top 10 closest official O*NET matches retrieved across all databases:\n{candidates_str}\n\n"
            f"A single candidate statement might contain multiple distinct skills or tasks (e.g., 'Wrote Python scripts for data analysis' = Python AND Data Analysis).\n"
            f"Select ALL options from the top 10 that accurately map to distinct components of the candidate's statement.\n"
            f"If none of them accurately represent the candidate's statement, output [\"NONE\"].\n"
            f"Output STRICTLY a valid JSON list of strings containing the exact 'ID's of the best matches. Do not include any other text or markdown."
        )

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )

            raw_content = response.choices[0].message.content.strip()
            print(f"\n[+] DEBUG - IRMapper DeepSeek Decision: {raw_content}")

            # Clean markdown if present
            if raw_content.startswith("```json"):
                raw_content = raw_content.replace("```json", "").replace("```", "").strip()
            elif raw_content.startswith("```"):
                raw_content = raw_content.replace("```", "").strip()

            matched_ids = json.loads(raw_content)
            print(f"\n[+] DEBUG - IRMapper DeepSeek Multi-Decision: {matched_ids}")

        except Exception as e:
            logger.error(f"Hybrid RAG LLM Error or JSON parse failure: {e}")
            matched_ids = []

        candidate_ids_as_strings = [str(c['id']) for c in all_candidates]
        final_results = []

        if not isinstance(matched_ids, list):
            matched_ids = []

        for m_id in matched_ids:
            if m_id == "NONE" or m_id not in candidate_ids_as_strings:
                continue

            best_match = next(c for c in all_candidates if str(c['id']) == m_id)
            context = self._fetch_db_context(best_match['id'], best_match['resolved_type'])

            final_results.append({
                "id": best_match['id'],
                "matched_text": best_match['text'],
                "score": f"Vector Score: {best_match['score']} (LLM Verified)",
                "context": context,
                "resolved_type": best_match['resolved_type']
            })

        return final_results

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
