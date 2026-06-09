from langgraph.graph import StateGraph, END
from workflow.state import CArcState
from workflow.nodes import mentor_node, profiler_node, ir_node, career_expert_node, evaluator_node, resume_generator_node

IR_TURN = 2
PROFILER_TURN = 2
EVALUATOR_TURN = 2


def dispatch_workers(state: CArcState) -> str:
    """Determines the first node to run in the sequence (Waterfall entry)."""

    if state.get("mentor_mode") == "generate_resume":
        return "resume_generator"

    if state.get("mentor_mode") == "counselor":
        return "mentor"

    if state.get("mentor_mode") == "validation":
        if state.get("profile_verified") is True:
            print("    -> Permission confirmed. Moving to Career Expert Inference Engine.")
            return "career_expert"
        else:
            print("    -> Enter validation loop.")
            return "profiler"

    if state.get("missing_demographics"):
        print("    -> Missing demographics active. Forcing Extraction Chain.")
        return "ir"

    turn = state.get("turn_count", 0)

    if turn == 0:
        return "mentor"

    if turn % PROFILER_TURN == 0:
        return "profiler"
    elif turn % IR_TURN == 0:
        return "ir"
    elif turn % EVALUATOR_TURN == 0:
        return "evaluator"
    else:
        return "mentor"


def profiler_post_router(state: CArcState) -> str:
    """Routes after the Profiler finishes."""
    if state.get("mentor_mode") == "validation":
        return "ir"
    turn = state.get("turn_count", 0)
    if turn > 0 and turn % IR_TURN == 0:
        return "ir"
    if turn > 0 and turn % EVALUATOR_TURN == 0:
        return "evaluator"
    return "mentor"


def ir_post_router(state: CArcState) -> str:
    """Routes after the IR Extractor finishes."""
    if state.get("mentor_mode") == "validation":
        return "mentor"
    if state.get("missing_demographics"):
        return "evaluator"
    turn = state.get("turn_count", 0)
    # Check if we hit the 10-turn milestone
    if turn > 0 and turn % EVALUATOR_TURN == 0:
        return "evaluator"
    return "mentor"


def build_graph():
    workflow = StateGraph(CArcState)

    # 1. Register all the Nodes
    workflow.add_node("mentor", mentor_node)
    workflow.add_node("profiler", profiler_node)
    workflow.add_node("ir", ir_node)
    workflow.add_node("career_expert", career_expert_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("resume_generator", resume_generator_node)

    # 2. Set Conditional Entry Point (Waterfall Dispatch)
    workflow.set_conditional_entry_point(
        dispatch_workers,
        {
            "profiler": "profiler",
            "ir": "ir",
            "evaluator": "evaluator",
            "mentor": "mentor",
            "career_expert": "career_expert",
            "resume_generator": "resume_generator"
        }
    )

    # 3. Profiler Routing
    workflow.add_conditional_edges(
        "profiler",
        profiler_post_router,
        {
            "ir": "ir",
            "evaluator": "evaluator",
            "mentor": "mentor"
        }
    )

    # 4. IR Extractor Routing
    workflow.add_conditional_edges(
        "ir",
        ir_post_router,
        {
            "evaluator": "evaluator",
            "mentor": "mentor"
        }
    )

    # 5. Conclude the graph after Action Nodes generate responses
    workflow.add_edge("evaluator", "mentor")
    workflow.add_edge("career_expert", "mentor")
    workflow.add_edge("mentor", END)
    workflow.add_edge("resume_generator", END)

    return workflow.compile()
