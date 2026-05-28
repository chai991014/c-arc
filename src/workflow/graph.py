from langgraph.graph import StateGraph, END
from .state import CArcState
from .nodes import mentor_node, profiler_node, ir_node, career_expert_node, evaluate_state_node
from .evaluator import evaluator_router


def dispatch_workers(state: CArcState) -> str:
    turn = state.get("turn_count", 0)
    should_run_ir = (turn > 0 and turn % 3 == 0)
    should_run_profiler = (turn > 0 and turn % 5 == 0)

    if should_run_profiler and should_run_ir:
        return "run_both"
    elif should_run_ir:
        return "run_ir_only"
    elif should_run_profiler:
        return "run_profiler_only"
    else:
        return "end"


def profiler_post_router(state: CArcState) -> str:
    """
    Checks if the IR Extractor also needs to run after the Profiler finishes.
    """
    turn = state.get("turn_count", 0)
    if turn > 0 and turn % 3 == 0:
        return "ir_extractor"  # Chain to IR if it's a shared turn (e.g., Turn 30)
    return "evaluate_state"


def build_graph():
    workflow = StateGraph(CArcState)

    # 1. Register all the Nodes (Including the new funnel node)
    workflow.add_node("mentor", mentor_node)
    workflow.add_node("profiler", profiler_node)
    workflow.add_node("ir_extractor", ir_node)
    workflow.add_node("career_expert", career_expert_node)
    workflow.add_node("evaluate_state", evaluate_state_node)

    # 2. Set Entry Point
    workflow.set_entry_point("mentor")

    # 3. Add Conditional Dispatch
    workflow.add_conditional_edges(
        "mentor",
        dispatch_workers,
        {
            "run_profiler_only": "profiler",
            "run_ir_only": "ir_extractor",
            "run_both": "profiler",
            "end": END
        }
    )

    # 4. Route workers
    # Dynamically route Profiler so it doesn't skip IR on overlapping turns
    workflow.add_conditional_edges(
        "profiler",
        profiler_post_router,
        {
            "ir_extractor": "ir_extractor",
            "evaluate_state": "evaluate_state"
        }
    )

    # IR always goes straight to evaluation when finished
    workflow.add_edge("ir_extractor", "evaluate_state")

    # 5. Add Conditional Evaluator (The Brain)
    workflow.add_conditional_edges(
        "evaluate_state",
        evaluator_router,
        {
            "mentor": "mentor",
            "career_expert": "career_expert"
        }
    )

    # 6. Conclude the graph
    workflow.add_edge("career_expert", END)

    return workflow.compile()
