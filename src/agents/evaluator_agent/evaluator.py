import math
from workflow.state import CArcState


def check_signal_strength(ocean_hits: dict, min_total_hits: int = 5) -> bool:
    """Ensures the LLM has extracted several strong, unambiguous personality signals."""
    print(f"\n[+] DEBUG - Signal Strength Check:")

    if not ocean_hits:
        return False

    total_hits = sum(ocean_hits.values())
    print(f"    -> Trait Hits: {ocean_hits}")
    print(f"    -> Total Strong Signals: {total_hits} | Threshold: >= {min_total_hits}")

    return total_hits >= min_total_hits


def check_logit_confidence(cumulative_confidence: float, threshold: float = 3.0) -> bool:
    """Ensures the LLM has maintained high overall certainty across the conversation."""
    print(f"\n[+] DEBUG - Logit Confidence Check:")
    print(f"    -> Accumulated Certainty Score: {cumulative_confidence:.3f} | Threshold: >= {threshold}")

    return cumulative_confidence >= threshold


def audit_entity_density(master_profile: dict, min_entities: int = 5) -> bool:
    """Hard fact check: Requires a total of at least X ONET-grounded entities across all dimensions."""
    if not master_profile:
        return False

    total_density = (
            len(master_profile.get("tasks", [])) +
            len(master_profile.get("dwas", [])) +
            len(master_profile.get("skills", [])) +
            len(master_profile.get("tech_skills", []))
    )
    return total_density >= min_entities


def verify_demographic_completeness(master_profile: dict) -> bool:
    """Strict gate: Enforces that basic info and education are explicitly extracted."""
    print(f"\n[+] DEBUG - Demographic Completeness Check:")
    basic_info = master_profile.get("basic_info", {})
    education = master_profile.get("education", [])

    required_fields = ["full_name", "location", "email", "phone"]
    missing_fields = [f for f in required_fields if not basic_info.get(f)]

    if not education:
        missing_fields.append("education")

    print(f"    -> Missing Mandatory Fields: {missing_fields if missing_fields else 'NONE'}")
    return len(missing_fields) == 0


def verify_profile_stability(ocean_history: list, threshold: float = 0.1) -> bool:
    """
    Monitors the velocity of change in the OCEAN vector.
    Calculates distance: d = sqrt( sum( (P_t - P_t-1)^2 ) )
    """
    print(f"\n[+] DEBUG - Profile Stability Check:")

    if len(ocean_history) < 2:
        print("    -> Result: FAILED (Not enough history to calculate distance)")
        return False

    p_t = ocean_history[-1]
    p_t_1 = ocean_history[-2]

    # Calculate Euclidean distance
    distance = math.sqrt(sum((p_t.get(k, 0) - p_t_1.get(k, 0)) ** 2 for k in p_t.keys()))

    # Format vectors for clean console output
    pt_str = {k: f"{v:.3f}" for k, v in p_t.items()}
    pt1_str = {k: f"{v:.3f}" for k, v in p_t_1.items()}

    print(f"    -> Vector T-1: {pt1_str}")
    print(f"    -> Vector T  : {pt_str}")
    print(f"    -> Euclidean Distance: {distance:.6f} | Threshold: < {threshold}")

    passed = distance < threshold
    print(f"    -> Stability Result: {'PASS' if passed else 'FAIL'}")

    return passed


def evaluator_router(state: CArcState) -> str:
    """The diagnostic router."""
    master_profile = state.get("master_profile", {})
    ocean_history = state.get("ocean_history", [])
    ocean_hits = state.get("ocean_hits", {})
    cumulative_confidence = state.get("cumulative_confidence", 0.0)

    print("\n[?] Evaluator Agent running Diagnostics...")

    # Run ALL diagnostic checks
    has_demographics = verify_demographic_completeness(master_profile)
    has_strong_signals = check_signal_strength(ocean_hits)
    has_high_confidence = check_logit_confidence(cumulative_confidence)
    has_sufficient_density = audit_entity_density(master_profile)
    is_profile_stable = verify_profile_stability(ocean_history)

    print("\n[=] Diagnostic Scorecard:")
    print(f"    -> Profile Completeness : {'PASS' if has_demographics else 'FAIL'}")
    print(f"    -> Signal Hits (>= 5)   : {'PASS' if has_strong_signals else 'FAIL'}")
    print(f"    -> High Certainty       : {'PASS' if has_high_confidence else 'FAIL'}")
    print(f"    -> Entity Density (>= 5): {'PASS' if has_sufficient_density else 'FAIL'}")
    print(f"    -> Profile Stability    : {'PASS' if is_profile_stable else 'FAIL'}")

    if has_demographics and has_strong_signals and has_high_confidence and has_sufficient_density and is_profile_stable:
        print("    -> ALL CHECKS PASSED! Transitioning to Career Expert.")
        return "career_expert"

    print("    -> ROUTING: Diagnostics incomplete. Continuing Phase 1 (Mentor Mode).")
    return "mentor"
