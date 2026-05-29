import json
from .state import CArcState
from utils.llm_provider import LocalLLMProvider
from model.ir_agent.extractor import IRExtractor
from model.ir_agent.mapper import IRMapper
from model.career_expert.inference import CareerExpert

# Initialize engines
llm = LocalLLMProvider()
ir_extractor = IRExtractor()
ir_mapper = IRMapper(db_path="../data_factory/onet.db", pool_dir="../data_factory/artifacts/")
expert_engine = CareerExpert()


def mentor_node(state: CArcState) -> dict:
    """Conversational interface powered by local Gemma 4 (Dual-Mode)."""
    ocean = state.get("ocean_vector", {})
    turn = state.get("turn_count", 0)
    print(f"\n[➔] STARTING NODE: mentor_node | Turn: {turn}")
    messages = state.get("messages", []) # 1. Fetch chat history

    # 2. Format chat history into a readable string for the LLM
    chat_history = "\n".join([
        f"{(msg.get('role', 'unknown') if isinstance(msg, dict) else msg.type).capitalize()}: "
        f"{msg.get('content', '') if isinstance(msg, dict) else msg.content}"
        for msg in messages
    ])

    mode = state.get("mentor_mode", "interviewer")  # Default Phase 1 mode
    final_recs = state.get("final_recommendations", "")

    if mode == "interviewer":
        system_prompt = (
            "You are the C-Arc Career Interviewer. Your goal is to get to know the candidate "
            "naturally and make them feel relaxed. Engage in a conversational, friendly, and empathetic "
            "dialogue to explore their personality, past experiences, educational background, and basic info.\n"
            "Instead of asking for a dry list of skills, gently guide the conversation to encourage them "
            "to share their technical background, tools they've used, and projects they are proud of.\n"
            "CRITICAL: Act like a real human in a chat app. Keep your responses highly concise. "
            "Limit your replies to 2-3 short sentences. Never write long paragraphs or monologues. "
            "Never output your internal thinking process. Output ONLY your direct dialogue."
        )
        user_prompt = f"User Profile: OCEAN={ocean}. Turn={turn}.\n\nConversation History:\n{chat_history}\n\nPlease respond to the candidate's last message."

    elif mode == "counselor":
        system_prompt = (
            "You are the C-Arc Career Counselor. Phase 1 is complete. "
            "Your role is to review ML predictions and provide empathetic, action-oriented coaching.\n"
            "CRITICAL: Act like a real human mentor. Keep your conversational responses highly concise "
            "(under 3 sentences) unless you are explicitly formatting and delivering the final recommendation list. "
            "Never output your internal thinking process. Output ONLY the final response."
        )

        # Inject the XGBoost results ONLY if they haven't been delivered yet
        if final_recs and "Match Score:" not in chat_history:
            user_prompt = (
                f"Conversation History:\n{chat_history}\n\n"
                f"Our XGBoost ML model generated these top career matches:\n{final_recs}\n\n"
                f"Please deliver these recommendations to the candidate gracefully and congratulate them."
            )
        else:
            # For all subsequent turns in Phase 2, act as a standard counselor answering their questions
            user_prompt = f"Conversation History:\n{chat_history}\n\nPlease continue counseling the candidate."

    response = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model="google/gemma-4-E4B-it",
        temperature=0.7
    )

    # If the model disobeys and leaks the "Thinking Process:" block, strip it out.
    if "Thinking Process:" in response:
        # The model usually drops a double newline (\n\n) right before it starts speaking to the user
        parts = response.split("\n\n")
        if len(parts) > 1:
            # Grab the very last block of text, which is the actual dialogue
            response = parts[-1].strip()
        else:
            # Fallback if no double newline exists: try to split at the last known header
            response = response.split("Construct the Response:")[-1].strip()

    return {
        "messages": [{"role": "assistant", "content": response}]
    }


def profiler_node(state: CArcState) -> dict:
    """Silent psychometrics powered by Logit-Based Rating Engine."""
    messages = state.get("messages", [])
    turn_count = state.get("turn_count", 1)
    print(f"\n[➔] STARTING NODE: profiler_node | Turn: {turn_count}")
    current_ocean = state.get("ocean_vector", {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5})
    current_hits = state.get("ocean_hits") or {"O": 0, "C": 0, "E": 0, "A": 0, "N": 0}
    current_confidence = state.get("cumulative_confidence", 0.0)

    chat_history = "\n".join([
        f"{(msg.get('role', 'unknown') if isinstance(msg, dict) else msg.type).capitalize()}: "
        f"{msg.get('content', '') if isinstance(msg, dict) else msg.content}"
        for msg in messages
    ])

    # Starts at ~0.18, decays down towards a floor of 0.05 as turns accumulate
    alpha = max(0.05, 0.2 / (1 + (turn_count * 0.1)))

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


def ir_node(state: CArcState) -> dict:
    """Information retrieval and ONET grounding engine."""
    turn = state.get("turn_count", 0)
    print(f"\n[➔] STARTING NODE: ir_node | Turn: {turn}")
    messages = state.get("messages", [])

    # Initialize the 4-part profile structure if it doesn't exist
    current_profile = state.get("master_profile", {})
    updated_profile = {
        "tasks": list(current_profile.get("tasks", [])),
        "dwas": list(current_profile.get("dwas", [])),
        "skills": list(current_profile.get("skills", [])),
        "tech_skills": list(current_profile.get("tech_skills", []))
    }

    if not messages:
        return {}

    extracted_changes = ir_extractor.extract_intents(messages)

    if not extracted_changes:
        print(f"    -> No entities extracted on this turn. Preserving profile history.")
        print(f"\n[+] DEBUG - Master Profile Updated: {updated_profile}")
        return {}

    for change in extracted_changes:
        intent = change.get("intent")
        entity_type = change.get("type")  # "skill", "task", or "dwa"
        value = change.get("value")

        if not value or entity_type not in ["skill", "task", "dwa"]:
            continue

        grounded_data = ir_mapper.ground_phrase(value, entity_type)
        onet_code = grounded_data.get("id")

        if not onet_code:
            continue

        print(f"\n[?] Mapping Attempt for '{value}' ({entity_type}):")
        print(f"    -> Score: {grounded_data.get('score', 'N/A')}")
        print(f"    -> Resulting ID: {onet_code}")

        # Route the ID to the correct array
        if entity_type == "task":
            target_list = updated_profile["tasks"]
        elif entity_type == "dwa":
            target_list = updated_profile["dwas"]
        elif entity_type == "skill":
            if str(onet_code).startswith("TECH-"):
                target_list = updated_profile["tech_skills"]
            else:
                target_list = updated_profile["skills"]

        # Conflict Resolution
        if intent == "ADD" and onet_code not in target_list:
            target_list.append(onet_code)
        elif intent == "DELETE" and onet_code in target_list:
            target_list.remove(onet_code)

    # Cleanly return the reconciled state
    print(f"\n[+] DEBUG - Master Profile Updated: {updated_profile}")
    return {
        "master_profile": updated_profile
    }


def career_expert_node(state: CArcState) -> dict:
    """Pure XGBoost inference. Formats data and triggers Phase 2 Mentor Mode."""
    print(f"\n[➔] STARTING NODE: career_expert_node (XGBoost Only)")

    master_profile = state.get("master_profile") or {}

    # 1. Grab the live OCEAN state (0.0 - 1.0 scale, short keys)
    state_ocean = state.get("ocean_vector", {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5})

    # 2. Format it for XGBoost (0 - 100 scale, full word keys expected by inference.py)
    formatted_ocean = {
        "openness": state_ocean.get("O", 0.5) * 100,
        "conscientiousness": state_ocean.get("C", 0.5) * 100,
        "extraversion": state_ocean.get("E", 0.5) * 100,
        "agreeableness": state_ocean.get("A", 0.5) * 100,
        "neuroticism": state_ocean.get("N", 0.5) * 100
    }

    # 3. Extract the 4 professional arrays
    user_tasks = master_profile.get("tasks", [])
    user_dwas = master_profile.get("dwas", [])
    user_skills = master_profile.get("skills", [])
    user_tech = master_profile.get("tech_skills", [])

    try:
        # Pass all dimensions and the formatted OCEAN vector to the model
        predictions = expert_engine.predict(
            user_tasks, user_dwas, user_skills, user_tech, formatted_ocean
        )
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

    # 5. Push the state updates to trigger Phase 2
    return {
        "mentor_mode": "counselor",
        "final_recommendations": formatted_recs
    }


def evaluator_node(state: CArcState) -> dict:
    """Pass-through node to funnel workers into the Evaluator Router."""
    turn = state.get("turn_count", 0)
    print(f"\n[➔] STARTING NODE: evaluate_state_node | Turn: {turn}")
    return {}
