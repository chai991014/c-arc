import json
from .state import CArcState
from utils.llm_provider import LocalLLMProvider
from model.ir_agent.extractor import IRExtractor
from model.ir_agent.mapper import IRMapper
from model.career_expert.inference import CareerExpert

# Initialize engines
llm = LocalLLMProvider()
ir_extractor = IRExtractor()
ir_mapper = IRMapper(db_path="../data/onet.db", pool_dir="../data/artifacts/")
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

    # Step 4: Profile Inertia (Alpha decay based on turn count)
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

    # Step 3: Logit-Based Rating Engine
    for trait_key, trait_full in trait_names.items():
        # Force the model to answer with a single classification token
        prompt = (
            f"Dialogue History: {messages}\n\n"
            f"Based on this dialogue, is the candidate's {trait_full} trait High or Low?\n"
            f"Answer strictly with the word High or Low:"
        )

        # Extract the probability ratio instead of text generation
        new_val = llm.get_logit_ratio(prompt, "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
        old_val = current_ocean.get(trait_key, 0.5)

        # Apply the Exponential Moving Average (EMA)
        updated_ocean[trait_key] = round((alpha * float(new_val)) + ((1 - alpha) * float(old_val)), 3)

    current_history = state.get("ocean_history", [])
    updated_history = current_history + [updated_ocean]

    return {
        "ocean_vector": updated_ocean,
        "ocean_history": updated_history
    }


def ir_node(state: CArcState) -> dict:
    """Information retrieval and ONET grounding engine."""
    turn = state.get("turn_count", 0)
    print(f"\n[➔] STARTING NODE: ir_node | Turn: {turn}")
    messages = state.get("messages", [])
    current_profile = state.get("master_profile", {"skills": []})

    if not messages:
        return {}

    # 1. The Engine does the heavy ML lifting
    extracted_changes = ir_extractor.extract_intents(messages)

    # 2. Conflict Resolution happening natively in the Node
    updated_skills = list(current_profile.get("skills", []))

    for change in extracted_changes:
        intent = change.get("intent")
        entity_type = change.get("type")
        value = change.get("value")

        if not value:
            continue

        if intent == "ADD" and entity_type == "skill":
            # Map raw text to ID
            grounded_data = ir_mapper.ground_phrase(value, entity_type="skill")
            onet_code = grounded_data.get("id")

            print(f"\n[?] Mapping Attempt for '{value}':")
            print(f"    -> Score: {grounded_data.get('score', 'N/A')}")
            print(f"    -> Matched Text: {grounded_data.get('matched_text', 'N/A')}")
            print(f"    -> Resulting ID: {onet_code}")

            # Conflict Resolution: Prevent duplicate skills
            if onet_code and onet_code not in updated_skills:
                updated_skills.append(onet_code)

        elif intent == "DELETE" and entity_type == "skill":
            # Map the raw text to ID, then safely remove it if it exists
            grounded_data = ir_mapper.ground_phrase(value, entity_type="skill")
            onet_code = grounded_data.get("id")

            if onet_code in updated_skills:
                updated_skills.remove(onet_code)

    # Cleanly return the reconciled state
    print(f"\n[+] DEBUG - Master Profile Updated: {updated_skills}")
    return {
        "master_profile": {"skills": updated_skills}
    }


def career_expert_node(state: CArcState) -> dict:
    """Pure XGBoost inference. Formats data and triggers Phase 2 Mentor Mode."""
    print(f"\n[➔] STARTING NODE: career_expert_node (XGBoost Only)")
    return {}



def evaluator_node(state: CArcState) -> dict:
    """Pass-through node to funnel workers into the Evaluator Router."""
    turn = state.get("turn_count", 0)
    print(f"\n[➔] STARTING NODE: evaluate_state_node | Turn: {turn}")
    return {}
