from workflow.state import CArcState

def execute_profiler(state: CArcState, llm) -> dict:
    messages = state.get("messages", [])
    turn = state.get("turn_count", 1)
    current_ocean = state.get("ocean_vector", {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5})
    current_hits = state.get("ocean_hits") or {"O": 0, "C": 0, "E": 0, "A": 0, "N": 0}
    current_confidence = state.get("cumulative_confidence", 0.0)

    chat_history = "\n".join([
        f"{(msg.get('role', 'unknown') if isinstance(msg, dict) else msg.type).capitalize()}: "
        f"{msg.get('content', '') if isinstance(msg, dict) else msg.content}"
        for msg in messages
    ])

    # Starts at ~0.18, decays down towards a floor of 0.05 as turns accumulate
    alpha = max(0.05, 0.2 / (1 + (turn * 0.1)))

    updated_ocean = {}
    trait_names = {
        'O': 'Openness to Experience',
        'C': 'Conscientiousness',
        'E': 'Extraversion',
        'A': 'Agreeableness',
        'N': 'Neuroticism'
    }

    print("\n[+] DEBUG - Profiler Raw Logits:")
    for trait_key, trait_full in trait_names.items():
        # Force the model to answer with a single classification token
        prompt = (
            f"You are an expert psychological profiler analyzing a candidate.\n\n"
            f"Conversation Transcript:\n{chat_history}\n\n"
            f"Based strictly on the candidate's statements above, evaluate their '{trait_full}' trait.\n"
            f"Is it High or Low? Answer strictly with one word: High or Low:"
        )

        # 1. Get the RAW logit probability
        new_val = float(llm.get_logit_ratio(prompt, "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"))

        # 2. HIT COUNTER: If probability is heavily skewed (>75% or <25%), it's a strong signal
        is_hit = new_val >= 0.75 or new_val <= 0.25
        if is_hit:
            current_hits[trait_key] += 1

        # 3. ENTROPY / CONFIDENCE ACCUMULATION: Measure certainty (distance from 0.5 neutral)
        confidence_delta = abs(new_val - 0.5)
        current_confidence += confidence_delta

        # 4. Apply standard EMA smoothing for the actual profile
        old_val = current_ocean.get(trait_key, 0.5)
        updated_ocean[trait_key] = round((alpha * new_val) + ((1 - alpha) * float(old_val)), 3)

        print(f"    -> [{trait_key}] Raw Ratio: {new_val:.4f} | Signal Hit: {'YES' if is_hit else 'NO '} | Conf Delta: +{confidence_delta:.3f} | New EMA: {updated_ocean[trait_key]:.3f}")

    current_history = state.get("ocean_history", [])
    updated_history = current_history + [updated_ocean]

    print(f"    => Updated Hit Counter: {current_hits}")
    print(f"    => New Cumulative Confidence: {current_confidence:.3f}")

    return {
        "ocean_vector": updated_ocean,
        "ocean_history": updated_history,
        "ocean_hits": current_hits,
        "cumulative_confidence": round(current_confidence, 3)
    }
