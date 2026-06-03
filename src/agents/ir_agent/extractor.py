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
        system_prompt = (
            "You are a precise data extraction agent. Your job is to analyze the entire conversation history "
            "and extract professional competencies, work experience, and candidate demographics to construct a resume profile.\n\n"

            "Categorize every extracted entity into exactly one of these types:\n"
            "1. 'skill' - Core professional capabilities or tools (e.g., 'Python', 'Machine learning').\n"
            "2. 'task' - Highly specific job duties or project achievements (e.g., 'Developed a RAG system').\n"
            "3. 'dwa' - Generalized transferable work activities (e.g., 'Analyze data to identify trends').\n"
            "4. 'basic_info' - Candidate personal metadata. Return a dictionary containing keys: 'full_name', 'email', 'phone', 'location'.\n"
            "5. 'education' - Academic achievements. Return a dictionary containing keys: 'degree', 'major', 'institution', 'grad_year'.\n\n"

            "Rules:\n"
            "- Use intent 'ADD' for newly mentioned details, or 'DELETE' if the user corrects/removes info.\n"
            "- For 'basic_info' and 'education', only capture fields explicitly stated by the user. Leave missing keys as null.\n"
            "- Output your final answer STRICTLY as a valid JSON list of objects. No thinking process, no markdown wrappers outside the json block.\n\n"

            "CRITICAL OUTPUT FORMAT EXAMPLE:\n"
            "[\n"
            "    {\"intent\": \"ADD\", \"type\": \"basic_info\", \"value\": {\"full_name\": \"JY\", \"location\": \"Cheras, Malaysia\", \"email\": null, \"phone\": null}},\n"
            "    {\"intent\": \"ADD\", \"type\": \"education\", \"value\": {\"degree\": \"Master\", \"major\": \"AI\", \"institution\": \"UM\", \"grad_year\": 2026}},\n"
            "    {\"intent\": \"ADD\", \"type\": \"skill\", \"value\": \"Python\"}\n"
            "]"
        )

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
