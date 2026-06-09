from workflow.state import CArcState

def execute_career_expert(state: CArcState, expert_engine) -> dict:
    master_profile = state.get("master_profile") or {}

    # 1. Grab the live OCEAN state (0.0 - 1.0 scale, short keys)
    state_ocean = state.get("ocean_vector", {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5})

    # 2. Format it for XGBoost (0 - 100 scale, full word keys expected by inference.py)
    formatted_ocean = {
        "OCEAN_O": float(state_ocean.get("O", 0.5)),
        "OCEAN_C": float(state_ocean.get("C", 0.5)),
        "OCEAN_E": float(state_ocean.get("E", 0.5)),
        "OCEAN_A": float(state_ocean.get("A", 0.5)),
        "OCEAN_N": float(state_ocean.get("N", 0.5))
    }

    try:
        # Pass the full master_profile and normalized OCEAN vector directly
        predictions = expert_engine.predict(master_profile, formatted_ocean)

    except Exception as e:
        print(f"[X] XGBoost Inference crashed: {e}")
        predictions = []

    # 4. Format the output for the LLM Counselor
    if not predictions:
        formatted_recs = "Error: Not enough data to generate recommendations."
        print("    -> Inference failed to generate predictions.")
    else:
        rec_details = []
        for i, p in enumerate(predictions):
            soc = p['soc_code']
            conf = p['probability']
            details = expert_engine.get_soc_details(soc)
            rec_details.append(
                f"{i + 1}. **{details['title']}** (Match Score: {conf}%)\n   *Overview:* {details['description']}"
            )

        formatted_recs = "\n\n".join(rec_details)
        print(f"    -> XGBoost generated {len(predictions)} recommendations. Transitioning to Counselor Mode.")
        print(f"    -> Recommendations:")
        print(formatted_recs)

    # 5. Push the state updates to trigger Phase 2
    return {
        "mentor_mode": "counselor",
        "final_recommendations": formatted_recs
    }
