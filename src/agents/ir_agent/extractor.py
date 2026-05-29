import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

# Load environment variables from the .env file in your root directory
load_dotenv()


class IRExtractor:
    def __init__(self):
        # Retrieve the API key from your .env file
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.error("DEEPSEEK_API_KEY not found! Please check your .env file.")

        # DeepSeek's API is natively compatible with the OpenAI Python client
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

    def extract_intents(self, dialogue_history: list) -> list[dict]:
        """
        Uses DeepSeek Reasoner API to parse dialogue history for professional changes.
        """
        system_prompt = """
        You are the C-Arc Information Retrieval Extractor.
        Analyze the ENTIRE provided dialogue history. Extract technical skills, tools, specific job tasks, and generalized work activities mentioned by the candidate.
        Categorize each entity with an intent: ADD, UPDATE, or DELETE.

        Entity types must be EXACTLY one of the following:
        - "skill" (software, programming languages, tools, and core professional abilities)
        - "task" (specific job duties, responsibilities, or accomplishments)
        - "dwa" (detailed work activities or generalized actions/verbs)

        Output strictly as a JSON list of dictionaries. Example:
        [
            {"intent": "ADD", "type": "skill", "value": "Python programming"},
            {"intent": "ADD", "type": "task", "value": "Designed database schemas"},
            {"intent": "ADD", "type": "dwa", "value": "Analyze data to identify trends"}
        ]

        If no professional entities are found in the text, return an empty list [].
        """

        # Look at the most recent context to save tokens
        recent_context = dialogue_history[-4:] if len(dialogue_history) >= 4 else dialogue_history
        user_text = "\n".join([
            f"{msg.get('role', 'unknown') if isinstance(msg, dict) else msg.type}: "
            f"{msg.get('content', '') if isinstance(msg, dict) else msg.content}"
            for msg in recent_context
        ])

        user_prompt = f"Dialogue:\n{user_text}\n\nExtract intents in JSON format:"

        print(f"\n[+] DEBUG - IRExtractor Parsed History:\n{user_prompt}\n")

        try:
            # Call the DeepSeek-Reasoner (R1) via API
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )

            # --- DEBUGGING: The API separates reasoning from the final answer! ---
            # You can view the thought process without it breaking your code
            reasoning = getattr(response.choices[0].message, 'reasoning_content', '')
            if reasoning:
                logger.info(f"DeepSeek Reasoning (API):\n{reasoning}")

            # The content field now ONLY contains the final JSON output
            raw_content = response.choices[0].message.content.strip()
            logger.info(f"DeepSeek Final Output (API):\n{raw_content}")

            # Clean markdown formatting if DeepSeek wraps the output
            if raw_content.startswith("```json"):
                raw_content = raw_content.replace("```json", "").replace("```", "").strip()
            elif raw_content.startswith("```"):
                raw_content = raw_content.replace("```", "").strip()

            intents = json.loads(raw_content)

            if isinstance(intents, list):
                return intents
            return []

        except json.JSONDecodeError:
            logger.error(f"IRExtractor failed to parse JSON from API response: {raw_content}")
            return []
        except Exception as e:
            logger.error(f"IRExtractor Error with DeepSeek API: {e}")
            return []
