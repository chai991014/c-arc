import sqlite3
import json
import logging
import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


class IRMapper:
    def __init__(self, db_path: str = "../data_factory/onet.db", pool_dir: str = "../data_factory/artifacts/"):
        self.db_path = db_path
        logger.info("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Retrieve the API key from your .env file
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.error("DEEPSEEK_API_KEY not found! Please check your .env file.")

        # DeepSeek's API is natively compatible with the OpenAI Python client
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

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
        """Hybrid RAG: Vector search (Top 5) + DeepSeek Reasoning."""
        pool = self.pools.get(entity_type)
        if not pool or pool["embeddings"] is None:
            return {"id": None, "context": None, "score": "N/A (Empty/Invalid Pool)"}

        # 1. Fast Vector Retrieval (Pull Top 5 instead of Top 1)
        query_embedding = self.embed_model.encode(raw_text, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, pool["embeddings"], top_k=10)[0]

        if not hits:
            return {"id": None, "context": None, "score": "N/A (No Hits)"}

        # 2. Build Candidates List
        candidates = []
        for hit in hits:
            idx = hit['corpus_id']
            candidates.append({
                "id": pool["ids"][idx],
                "text": pool["texts"][idx],
                "score": round(float(hit['score']), 3)
            })

        candidates_str = "\n".join([f"- ID: {c['id']} | Description: {c['text']}" for c in candidates])

        print(f"\n[+] DEBUG - IRMapper Vector Retrieval for '{raw_text}' ({entity_type}):")
        print(candidates_str)

        # 3. LLM Reranking (Using the faster, cheaper chat model)
        prompt = (
            f"You are a strict O*NET taxonomy mapper.\n"
            f"The candidate described this specific {entity_type}: \"{raw_text}\"\n\n"
            f"Here are the top 10 closest official O*NET matches retrieved from the database:\n{candidates_str}\n\n"
            f"Which of these 10 options is the most accurate semantic match?\n"
            f"If none of them accurately represent the candidate's statement, you must output 'NONE'.\n"
            f"Output strictly the exact 'ID' of the best match or 'NONE'. Do not include any other text."
        )

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )

            raw_content = response.choices[0].message.content.strip()
            print(f"\n[+] DEBUG - IRMapper DeepSeek Decision: {raw_content}")

        except Exception as e:
            logger.error(f"Hybrid RAG LLM Error: {e}")
            raw_content = "NONE"

        # 4. Process LLM Decision
        candidate_ids_as_strings = [str(c['id']) for c in candidates]
        if raw_content == "NONE" or raw_content not in candidate_ids_as_strings:
            # LLM rejected all candidates, or hallucinated an ID
            top_vector = candidates[0]
            return {
                "id": None,
                "matched_text": f"[LLM REJECTED] Top match was: {top_vector['text']}",
                "score": f"Vector Score was {top_vector['score']}",
                "context": "N/A"
            }

        # 5. Hydrate the winning match
        best_match = next(c for c in candidates if str(c['id']) == raw_content)
        context = self._fetch_db_context(best_match['id'], entity_type)

        return {
            "id": best_match['id'],
            "matched_text": best_match['text'],
            "score": f"Vector Score: {best_match['score']} (LLM Verified)",
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
