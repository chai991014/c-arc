from typing import Annotated, TypedDict, List, Dict, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class MasterProfile(TypedDict):
    """The hard-fact professional data grounded in O*NET."""
    skills: List[Dict[str, str]]  # List of {"id": "1.A.1", "name": "Python"}
    work_activities: List[Dict[str, str]]  # List of {"id": "4.A.1", "name": "Analyzing Data"}
    role_history: List[Dict]  # Experience details
    education: Optional[str]
    verification_status: bool  # True if user confirmed "Digital Twin"


class CArcState(TypedDict):
    """The central state for the C-Arc Multi-Agent System."""
    # Annotated with add_messages so LangGraph appends chat history automatically
    messages: Annotated[List[BaseMessage], add_messages]

    # Psychometric Data [0-100 scalar]
    ocean_vector: Dict[str, float]  # {"O": 0.0, "C": 0.0, "E": 0.0, "A": 0.0, "N": 0.0}

    # Grounded Professional Data
    master_profile: MasterProfile

    # Logic Controller signals
    state_signal: str  # e.g., "DISCOVERY", "VALIDATION", "MATCHING"
    next_topic: Optional[str]  # Signal from Evaluator (e.g., "FORCE_TOPIC [SKILL]")
