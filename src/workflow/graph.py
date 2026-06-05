from langgraph.graph import StateGraph, END
from .state import CArcState
from .nodes import mentor_node, profiler_node, ir_node, career_expert_node, evaluator_node

IR_TURN = 2
PROFILER_TURN = 2
EVALUATOR_TURN = 2


def dispatch_workers(state: CArcState) -> str:
    """Determines the first node to run in the sequence (Waterfall entry)."""

    if state.get("mentor_mode") == "counselor":
        return "mentor"

    if state.get("mentor_mode") == "validation":
        if state.get("profile_verified") is True:
            print("    -> Permission confirmed. Moving to XGBoost Inference Engine.")
            return "career_expert"
        else:
            print("    -> Enter validation loop.")
            return "profiler"

    if state.get("missing_demographics"):
        print("    -> Missing demographics active. Forcing Extraction Chain.")
        return "ir_extractor"

    turn = state.get("turn_count", 0)

    if turn == 0:
        return "mentor"

    if turn % PROFILER_TURN == 0:
        return "profiler"
    elif turn % IR_TURN == 0:
        return "ir_extractor"
    elif turn % EVALUATOR_TURN == 0:
        return "evaluator_agent"
    else:
        return "mentor"


def profiler_post_router(state: CArcState) -> str:
    """Routes after the Profiler finishes."""
    if state.get("mentor_mode") == "validation":
        return "ir_extractor"
    turn = state.get("turn_count", 0)
    if turn > 0 and turn % IR_TURN == 0:
        return "ir_extractor"
    if turn > 0 and turn % EVALUATOR_TURN == 0:
        return "evaluator_agent"
    return "mentor"


def ir_post_router(state: CArcState) -> str:
    """Routes after the IR Extractor finishes."""
    if state.get("mentor_mode") == "validation":
        return "mentor"
    if state.get("missing_demographics"):
        return "evaluator_agent"
    turn = state.get("turn_count", 0)
    # Check if we hit the 10-turn milestone
    if turn > 0 and turn % EVALUATOR_TURN == 0:
        return "evaluator_agent"
    return "mentor"


def build_graph():
    workflow = StateGraph(CArcState)

    # 1. Register all the Nodes
    workflow.add_node("mentor", mentor_node)
    workflow.add_node("profiler", profiler_node)
    workflow.add_node("ir_extractor", ir_node)
    workflow.add_node("career_expert", career_expert_node)
    workflow.add_node("evaluator_agent", evaluator_node)

    # 2. Set Conditional Entry Point (Waterfall Dispatch)
    workflow.set_conditional_entry_point(
        dispatch_workers,
        {
            "profiler": "profiler",
            "ir_extractor": "ir_extractor",
            "evaluator_agent": "evaluator_agent",
            "mentor": "mentor",
            "career_expert": "career_expert"
        }
    )

    # 3. Profiler Routing
    workflow.add_conditional_edges(
        "profiler",
        profiler_post_router,
        {
            "ir_extractor": "ir_extractor",
            "evaluator_agent": "evaluator_agent",
            "mentor": "mentor"
        }
    )

    # 4. IR Extractor Routing
    workflow.add_conditional_edges(
        "ir_extractor",
        ir_post_router,
        {
            "evaluator_agent": "evaluator_agent",
            "mentor": "mentor"
        }
    )

    # 5. Conclude the graph after Action Nodes generate responses
    workflow.add_edge("evaluator_agent", "mentor")
    workflow.add_edge("career_expert", "mentor")
    workflow.add_edge("mentor", END)

    return workflow.compile()
