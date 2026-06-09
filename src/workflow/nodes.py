import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from workflow.state import CArcState
from utils.llm_provider import LocalLLMProvider
from agents.info_retrieval.extractor import IRExtractor
from agents.info_retrieval.mapper import IRMapper
from agents.career_expert.inference import CareerExpert

from agents.mentor.mentor import execute_mentor
from agents.profiler.profiler import execute_profiler
from agents.info_retrieval.ir import execute_ir
from agents.career_expert.expert import execute_career_expert
from agents.evaluator.evaluator import execute_evaluator
from agents.resume_generator.generator import generate_resume

# Initialize engines
logger = logging.getLogger(__name__)
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    logger.error("DEEPSEEK_API_KEY not found! Please check your .env file.")

llm_client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)
llm = LocalLLMProvider()
ir_extractor = IRExtractor(llm_client)
ir_mapper = IRMapper(llm_client, db_path="../data_factory/onet.db", pool_dir="../data_factory/artifacts/")
expert_engine = CareerExpert()


def mentor_node(state: CArcState) -> dict:
    """Conversational interface powered by local Gemma 4 (Dual-Mode)."""
    turn = state.get("turn_count", 0)
    print(f"\n[➔] STARTING NODE: mentor_node | Turn: {turn}")
    return execute_mentor(state, llm)


def profiler_node(state: CArcState) -> dict:
    """Silent psychometrics powered by Logit-Based Rating Engine."""
    turn = state.get("turn_count", 1)
    print(f"\n[➔] STARTING NODE: profiler_node | Turn: {turn}")
    return execute_profiler(state, llm)


def ir_node(state: CArcState) -> dict:
    """Information retrieval and ONET grounding engine."""
    turn = state.get("turn_count", 0)
    print(f"\n[➔] STARTING NODE: ir_node | Turn: {turn}")
    return execute_ir(state, ir_extractor, ir_mapper)


def career_expert_node(state: CArcState) -> dict:
    """ML inference. Formats data and triggers Phase 2 Mentor Mode."""
    print(f"\n[➔] STARTING NODE: career_expert_node")
    return execute_career_expert(state, expert_engine)


def evaluator_node(state: CArcState) -> dict:
    """Pass-through node to funnel workers into the Evaluator Router."""
    turn = state.get("turn_count", 0)
    print(f"\n[➔] STARTING NODE: evaluate_state_node | Turn: {turn}")
    return execute_evaluator(state)


def resume_generator_node(state: CArcState) -> dict:
    # Pass your initialized DeepSeek client or local LLM here
    return generate_resume(state, llm_client)
