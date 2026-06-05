import sqlite3
from workflow.state import CArcState


def translate_onet_ids(master_profile: dict, db_path: str = "../data_factory/onet.db") -> dict:
    translated = {"tasks": [], "dwas": [], "skills": [], "tech_skills": []}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for t_id in master_profile.get("tasks", []):
            cursor.execute("SELECT task FROM task_statements WHERE task_id = ?", (t_id,))
            row = cursor.fetchone()
            translated["tasks"].append(row[0] if row else t_id)
        for d_id in master_profile.get("dwas", []):
            cursor.execute("SELECT dwa_title FROM dwa_reference WHERE dwa_id = ?", (d_id,))
            row = cursor.fetchone()
            translated["dwas"].append(row[0] if row else d_id)
        for s_id in master_profile.get("skills", []):
            cursor.execute("SELECT element_name FROM skills WHERE element_id = ?", (s_id,))
            row = cursor.fetchone()
            translated["skills"].append(row[0] if row else s_id)
        conn.close()
    except Exception as e:
        print(f"[X] Translation Error: {e}")
        translated["tasks"] = master_profile.get("tasks", [])
        translated["dwas"] = master_profile.get("dwas", [])
        translated["skills"] = master_profile.get("skills", [])

    translated["tech_skills"] = [tech.replace("TECH-", "") for tech in master_profile.get("tech_skills", [])]
    return translated


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
    master_profile = state.get("master_profile") or {}

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

        missing_demo = state.get("missing_demographics", [])
        weak_traits = state.get("weak_ocean_traits", [])

        steering_directive = ""

        # Priority 0 (The Icebreaker) ---
        if turn <= 1:
            steering_directive = "\n- [STARTING TOPIC]: This is the beginning of the interview. Warmly ask the candidate to provide a simple self-introduction to get things started."

        # Priority 1: Secure mandatory demographics
        if missing_demo:
            steering_directive = f"\n- [URGENT MISSION]: You must ask the user to provide their missing profile information: {', '.join(missing_demo)}."

        # Priority 2: If demographics are complete, target the single weakest OCEAN trait
        elif weak_traits:
            trait_translations = {
                "O": "openness: adaptability to new ideas, technical curiosity, and creative problem solving",
                "C": "conscientiousness: attention to detail, organization, and project execution",
                "E": "extraversion: communication style, teamwork, and leadership dynamics",
                "A": "agreeableness: empathy, conflict resolution, and cross-functional collaboration",
                "N": "neuroticism: stress management, handling high-pressure deadlines, and resilience"
            }

            # Grab the absolute lowest trait (index 0) from the sorted array
            lowest_trait = weak_traits[0]
            probe = trait_translations.get(lowest_trait, "general work experience")
            steering_directive = f"\n- [FOCUS TOPIC]: To better evaluate their personality, steer the conversation to explore their {probe}. Ask a natural interview question about this."

        user_prompt = f"User Profile: OCEAN={ocean}. Turn={turn}."
        if steering_directive:
            user_prompt += f"\n\nSYSTEM DIRECTIVES:{steering_directive}"

        user_prompt += f"\n\nConversation History:\n{chat_history}\n\nPlease respond to the candidate's last message."


    elif mode == "validation":
        system_prompt = (
            "You are the C-Arc Profile Validator. Phase 1 exploration is complete. "
            "Your task is to acknowledge the candidate's latest message in 1-2 short sentences. "
            "If they provided corrections, acknowledge them warmly. "
            "CRITICAL: Keep your response ultra-short. Do NOT attempt to generate or type out their profile data. "
            "Just tell them to review the profile below and click 'Confirm & Proceed'."
        )
        user_prompt = f"Conversation History:\n{chat_history}\n\nPlease respond to the candidate."

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

    # Manually attach the exact Markdown summary to the final output so the UI displays it perfectly.
    if mode == "validation":
        readable_data = translate_onet_ids(master_profile)
        all_skills = readable_data["skills"] + readable_data["tech_skills"]
        skills_display = ", ".join(all_skills) if all_skills else "None"
        all_experience = readable_data["tasks"] + readable_data["dwas"]
        exp_display = "\n* ".join([""] + all_experience) if all_experience else "None"

        profile_summary = (
            f"### 📋 Your Profile Summary\n"
            f"**Name:** {master_profile.get('basic_info', {}).get('full_name', 'Not provided')}\n"
            f"**Location:** {master_profile.get('basic_info', {}).get('location', 'Not provided')}\n"
            f"**Contact:** {master_profile.get('basic_info', {}).get('email', 'N/A')} | {master_profile.get('basic_info', {}).get('phone', 'N/A')}\n\n"
            f"**Education:** {', '.join([f['degree'] + ' in ' + f['major'] for f in master_profile.get('education', [])]) if master_profile.get('education') else 'Not provided'}\n\n"
            f"**Extracted Skills:**\n{skills_display}\n\n"
            f"**Tracked Work Responsibilities:**{exp_display}\n\n"
            f"---\n"
            f"Please check through these details. If everything is correct and you have no more modifications, "
            f"please click the **Confirm & Proceed** button below to generate your expert career matches!"
        )

        response = f"{response}\n\n{profile_summary}"

    return {
        "messages": [{"role": "assistant", "content": response}]
    }
