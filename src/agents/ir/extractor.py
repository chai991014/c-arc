import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Dynamic pathing to ensure .env and artifacts are always found
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# src/agents/ir/ -> src/agents/ -> src/ -> project_root/ (3 levels up)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

# Load .env using absolute path
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


class IRExtractor:
    def __init__(self):
        # Initialize DeepSeek via LangChain using exact .env variables
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
            temperature=0.0,  # Zero creativity, strict extraction
            model_kwargs={"response_format": {"type": "json_object"}}  # Force JSON
        )
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Loads the text prompt and injects the 41 work activities."""
        # 1. Load the text file (Path is relative to where this script lives)
        prompt_path = os.path.join(SCRIPT_DIR, "extractor_prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read()

        # 2. Load the Work Activities JSON using the .env path
        artifact_dir = os.path.join(PROJECT_ROOT, os.getenv("ARTIFACT_DIR", "data/artifacts"))
        if not artifact_dir:
            raise ValueError("ARTIFACT_DIR environment variable is not set. Check your .env file.")

        artifact_path = os.path.join(artifact_dir, "work_activities_pool.json")
        with open(artifact_path, "r", encoding="utf-8") as f:
            activities = json.load(f)

        # Extract just the names for the LLM menu
        activity_menu = "\n".join([f"- {item}" for item in activities])

        # 3. Inject and return
        return template.replace("{work_activities}", activity_menu)

    def extract(self, user_text: str) -> dict:
        """Sends the user text to DeepSeek and returns the parsed JSON packet."""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_text)
        ]

        response = self.llm.invoke(messages)

        try:
            # Parse the string response into a Python dictionary
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback safety
            return {"extractions": []}
