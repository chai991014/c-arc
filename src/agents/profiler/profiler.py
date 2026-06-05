from workflow.state import CArcState


def execute_profiler(state: CArcState, llm) -> dict:
    messages = state.get("messages", [])
    turn = state.get("turn_count", 1)
    current_ocean = state.get("ocean_vector", {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5})
    trait_maturity = state.get("trait_maturity") or {"O": 0.0, "C": 0.0, "E": 0.0, "A": 0.0, "N": 0.0}
    base_rate = 0.15  # Base information gained just by processing a conversational turn

    chat_history = "\n".join([
        f"{(msg.get('role', 'unknown') if isinstance(msg, dict) else msg.type).capitalize()}: "
        f"{msg.get('content', '') if isinstance(msg, dict) else msg.content}"
        for msg in messages
    ])

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

        # 1. Get raw probability
        p_t = float(llm.get_logit_ratio(prompt, "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"))

        # 2. Calculate the weight (Data Volume) of this specific turn
        displacement = abs(p_t - 0.5)
        w_t = base_rate + displacement

        # 3. Retrieve historical values
        old_score = float(current_ocean.get(trait_key, 0.5))
        old_maturity = float(trait_maturity.get(trait_key, 0.0))

        # 4. Apply the Confidence-Weighted Math
        new_maturity = old_maturity + w_t
        new_score = ((old_score * old_maturity) + (p_t * w_t)) / new_maturity

        # 5. Save the updates
        trait_maturity[trait_key] = round(new_maturity, 3)
        updated_ocean[trait_key] = round(new_score, 3)

        print(
            f"    -> [{trait_key}] Raw: {p_t:.4f} | Weight Added: +{w_t:.3f} | Total Maturity: {trait_maturity[trait_key]:.3f} | New Score: {updated_ocean[trait_key]:.3f}")

    current_history = state.get("ocean_history", [])
    updated_history = current_history + [updated_ocean]

    print(f"    => Updated Trait Maturity: {trait_maturity}")

    return {
        "ocean_vector": updated_ocean,
        "ocean_history": updated_history,
        "trait_maturity": trait_maturity
    }
