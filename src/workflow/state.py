import operator
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph.message import add_messages


class CArcState(TypedDict, total=False):
    # Immutable log of all agent interactions
    messages: Annotated[list, add_messages]

    # Master Profile: extracted technical skills and history
    master_profile: Dict[str, Any]

    # Live OCEAN Vector (Updated by Profiler)
    ocean_vector: Dict[str, float]

    # History of OCEAN vectors to calculate Euclidean Stability (Gate 3)
    ocean_history: List[Dict[str, float]]

    # Turn tracking to manage learning rate decay and dispatch logic
    turn_count: Annotated[int, operator.add]

    # Tracks which behavioral mode the Mentor should use ('exploratory' or 'directive')
    mentor_mode: str
