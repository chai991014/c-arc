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

    knn_rec: str

    final_rec: str

    final_recommendations: str

    trait_maturity: Dict[str, float]

    profile_verified: bool

    missing_demographics: list[str]

    weak_ocean_traits: list[str]

    ir_last_extracted_index: int

    # Stores the formatted profile overview for the UI
    profile_summary: str

    # Tracks the specific job title the user wants to build the resume for
    target_career: str

    # Stores the generated Markdown resume
    resume_content: str
