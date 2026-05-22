from typing import TypedDict, List, Dict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class OceanVector(TypedDict):
    """The Big Five personality dimensions updated via Exponential Moving Average."""
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float


class MasterProfile(TypedDict):
    """The candidate's grounded professional trajectory."""
    skills: List[str]  # Array of verified O*NET skill/DWA IDs
    experience: List[Dict[str, str]]  # Array of past roles and descriptions
    education: List[Dict[str, str]]  # Educational background


class GateStatus(TypedDict):
    """Tracks the Evaluator Agent's Triple-Gate Logic."""
    trait_variance_cleared: bool
    density_audit_cleared: bool
    stability_cleared: bool


class CArcState(TypedDict):
    """
    The central state machine (Nervous System) for the C-Arc multi-agent framework.
    Passed between Mentor, Profiler, IR, and Evaluator agents.
    """

    # 1. Message History
    # The `add_messages` reducer ensures new dialogue is appended, not overwritten.
    messages: Annotated[list[AnyMessage], add_messages]

    # 2. Candidate Profiles
    master_profile: MasterProfile
    ocean_vector: OceanVector

    # 3. System Routing & Control Signals
    current_mode: str  # Defines the active agent phase (e.g., "interview", "evaluation", "synthesis")
    gate_status: GateStatus  # Tracks if the interview can conclude
    turn_count: int  # Tracks conversation length for stability dampening math
    force_topic: str  # Signal for the Mentor to shift the conversation focus
