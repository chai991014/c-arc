from workflow.state import CArcState

def execute_career_expert(state: CArcState, expert_engine) -> dict:
    master_profile = state.get("master_profile") or {}

    # 1. Grab the live OCEAN state (0.0 - 1.0 scale, short keys)
    state_ocean = state.get("ocean_vector", {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5})
    formatted_ocean = {k: float(v) for k, v in state_ocean.items()}

    try:
        # Pass the full master_profile and normalized OCEAN vector directly
        predictions = expert_engine.predict(master_profile, formatted_ocean)

    except Exception as e:
        print(f"[X] Career Expert Inference crashed: {e}")
        predictions = []

    # 4. Format the output for the LLM Counselor
    if not predictions:
        formatted_recs = "Error: Not enough data to generate recommendations."
        print("    -> Inference failed to generate predictions.")
    else:
        rec_details = []
        # Return top 5 out of the 15 retrieved by KNN
        for i, p in enumerate(predictions[:5]):
            soc = p['soc_code']
            score = p['match_score']
            title = p['job_title']

            # Fetch deeper description from the DB if needed, or use the base payload
            details = expert_engine.get_soc_details(soc) if hasattr(expert_engine, 'get_soc_details') else {
                "description": "O*NET Role mapping."}

            rec_details.append(
                f"{i + 1}. **{title}** (Capability Match: {score}%)\n   *Overview:* {details['description']}"
            )

        formatted_recs = "\n\n".join(rec_details)
        print(f"    -> Engine evaluated {len(predictions)} candidates. Transitioning top 5 to Counselor.")
        print(f"    -> Recommendations:")
        print(formatted_recs)

    return {
        "mentor_mode": "counselor",
        "final_recommendations": formatted_recs
    }
