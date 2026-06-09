import math
from workflow.state import CArcState


def get_weak_ocean_traits(trait_maturity: dict, threshold: float = 1.0) -> list[str]:
    """Returns the weakest OCEAN traits (maturity < 1.0) sorted in ascending order."""
    traits = ["O", "C", "E", "A", "N"]
    weak = [t for t in traits if trait_maturity.get(t, 0.0) < threshold]
    weak.sort(key=lambda t: trait_maturity.get(t, 0.0))
    return weak[:2]


def check_psychometric_readiness(trait_maturity: dict, overall_threshold: float = 5.0) -> bool:
    """Ensures the total volume of data collected across all traits is sufficient."""
    total_maturity = sum(trait_maturity.values())
    print(f"\n[+] DEBUG - Psychometric Readiness Check:")
    print(f"    -> Total Data Maturity: {total_maturity:.3f} | Threshold: >= {overall_threshold}")
    return total_maturity >= overall_threshold


def audit_entity_density(master_profile: dict, min_entities: int = 5) -> bool:
    """Hard fact check: Requires a total of at least X ONET-grounded entities across all dimensions."""
    if not master_profile:
        return False

    total_density = (
            len(master_profile.get("tasks", [])) +
            len(master_profile.get("dwas", [])) +
            len(master_profile.get("work_activities", [])) +
            len(master_profile.get("skills", [])) +
            len(master_profile.get("tech_skills", []))
    )
    return total_density >= min_entities


def get_missing_demographics(master_profile: dict) -> list[str]:
    """Returns a list of missing mandatory demographic fields."""
    basic_info = master_profile.get("basic_info", {})
    education = master_profile.get("education", [])

    missing_fields = [f for f in ["full_name", "location", "email", "phone"] if not basic_info.get(f)]
    if not education:
        missing_fields.append("education")

    return missing_fields


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


def execute_evaluator(state: CArcState) -> dict:
    # Skip checks if already in validation or counselor mode
    if state.get("mentor_mode") in ["validation", "counselor"]:
        return {}

    master_profile = state.get("master_profile", {})
    trait_maturity = state.get("trait_maturity", {"O": 0.0, "C": 0.0, "E": 0.0, "A": 0.0, "N": 0.0})
    ocean_history = state.get("ocean_history", [])

    missing_demo = get_missing_demographics(master_profile)
    weak_traits = get_weak_ocean_traits(trait_maturity)
    is_psychometrically_ready = check_psychometric_readiness(trait_maturity)
    has_sufficient_density = audit_entity_density(master_profile)
    is_profile_stable = verify_profile_stability(ocean_history)

    state_update = {
        "missing_demographics": missing_demo,
        "weak_ocean_traits": weak_traits
    }

    print("\n[=] Diagnostic Scorecard:")
    print(f"    -> Missing Demographics : {'PASS' if not missing_demo else missing_demo}")
    print(f"    -> Targeted Weak Traits : {'PASS' if not weak_traits else weak_traits}")
    print(f"    -> Psychometric Ready   : {'PASS' if is_psychometrically_ready else 'FAIL'}")
    print(f"    -> Entity Density (>= 5): {'PASS' if has_sufficient_density else 'FAIL'}")
    print(f"    -> Profile Stability    : {'PASS' if is_profile_stable else 'FAIL'}")

    if not missing_demo and is_psychometrically_ready and has_sufficient_density and is_profile_stable:
        print("    -> ALL CHECKS PASSED! Shifting to Validation Loop.")
        state_update["mentor_mode"] = "validation"
        return state_update

    print("    -> ROUTING: Diagnostics incomplete. Continuing Phase 1 (Mentor Mode).")
    return state_update
