import math
from .state import CArcState


def gate_2_density_audit(master_profile: dict) -> bool:
    """Hard fact check: Requires at least 5 ONET-grounded skills."""
    skills = master_profile.get("skills", [])
    return len(skills) >= 5


def gate_3_euclidean_stability(ocean_history: list, threshold: float = 0.1) -> bool:
    """
    Monitors the velocity of change in the OCEAN vector.
    Calculates distance: d = sqrt( sum( (P_t - P_t-1)^2 ) )
    """
    if len(ocean_history) < 2:
        return False

    p_t = ocean_history[-1]
    p_t_1 = ocean_history[-2]

    distance = math.sqrt(sum((p_t.get(k, 0) - p_t_1.get(k, 0)) ** 2 for k in p_t.keys()))
    return distance < threshold


def evaluator_router(state: CArcState) -> str:
    """
    The Triple-Gate router.
    Returns 'mentor' if gates fail, or 'career_expert' if gates pass.
    """
    master_profile = state.get("master_profile", {})
    ocean_history = state.get("ocean_history", [])

    # Gate 2: Check Skill Density
    if not gate_2_density_audit(master_profile):
        return "mentor"

    # Gate 3: Check Profile Stability
    if not gate_3_euclidean_stability(ocean_history):
        return "mentor"

    # Gate 1 (Trait Variance) logic would be added here.

    # All gates passed: Transition to the asset synthesis / matching phase
    return "career_expert"
