from workflow.state import CArcState

def execute_mentor(state: CArcState, llm) -> dict:
    ocean = state.get("ocean_vector", {})
    turn = state.get("turn_count", 0)
    messages = state.get("messages", [])  # 1. Fetch chat history

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
