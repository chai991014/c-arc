import json
from .state import CArcState
from utils.llm_provider import LocalLLMProvider
from agents.ir_agent.extractor import IRExtractor
from agents.ir_agent.mapper import IRMapper
from agents.career_expert.inference import CareerExpert

from agents.mentor_agent.mentor import execute_mentor
from agents.profiler_agent.profiler import execute_profiler
from agents.ir_agent.ir import execute_ir
from agents.career_expert.expert import execute_career_expert

# Initialize engines
llm = LocalLLMProvider()
ir_extractor = IRExtractor()
ir_mapper = IRMapper(db_path="../data_factory/onet.db", pool_dir="../data_factory/artifacts/")
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
    """Pure XGBoost inference. Formats data and triggers Phase 2 Mentor Mode."""
    print(f"\n[➔] STARTING NODE: career_expert_node (XGBoost Only)")
    return execute_career_expert(state, expert_engine)


def evaluator_node(state: CArcState) -> dict:
    """Pass-through node to funnel workers into the Evaluator Router."""
    turn = state.get("turn_count", 0)
    print(f"\n[➔] STARTING NODE: evaluate_state_node | Turn: {turn}")
    return {}
