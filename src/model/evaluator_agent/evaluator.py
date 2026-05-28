import math
from workflow.state import CArcState


def gate_1_trait_variance(ocean_vector: dict, threshold: float = 0.05) -> bool:
    """
    Ensures the profile has meaningfully diverged from the initial 0.5 baseline.
    Calculates the sum of absolute differences from 0.5.
    """
    if not ocean_vector:
        return False

    variance = sum(abs(val - 0.5) for val in ocean_vector.values())
    return variance >= threshold


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
    ocean_vector = state.get("ocean_vector", {})

    print("\n[?] Evaluator Agent running Diagnostics...")

    # Gate 1: Check Trait Variance (Has the personality profile moved?)
    pass_gate_1 = gate_1_trait_variance(ocean_vector)
    print(f"    -> Gate 1 (Trait Variance): {'PASS' if pass_gate_1 else 'FAIL'}")
    if not pass_gate_1:
        return "mentor"

    # Gate 2: Check Skill Density (Do we have enough O*NET data?)
    pass_gate_2 = gate_2_density_audit(master_profile)
    print(f"    -> Gate 2 (Skill Density >= 5): {'PASS' if pass_gate_2 else 'FAIL'}")
    if not pass_gate_2:
        return "mentor"

    # Gate 3: Check Profile Stability (Has the OCEAN vector settled?)
    pass_gate_3 = gate_3_euclidean_stability(ocean_history)
    print(f"    -> Gate 3 (Profile Stability): {'PASS' if pass_gate_3 else 'FAIL'}")
    if not pass_gate_3:
        return "mentor"

    print("    -> ALL GATES PASSED! Transitioning to Career Expert.")
    return "career_expert"
