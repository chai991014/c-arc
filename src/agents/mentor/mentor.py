import sqlite3
from workflow.state import CArcState


def translate_onet_ids(master_profile: dict, db_path: str = "../data_factory/onet.db") -> dict:
    translated = {"tasks": [], "dwas": [], "skills": [], "tech_skills": []}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Use a Set to automatically handle duplicate DWA IDs
        rolled_up_dwas = set(master_profile.get("dwas", []))
        unlinked_tasks = []

        # 1. Roll-up Tasks to DWAs
        for t_id in master_profile.get("tasks", []):
            try:
                # Find the parent DWA for this specific task
                cursor.execute("SELECT dwa_id FROM tasks_to_dwas WHERE task_id = ? LIMIT 1", (t_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    rolled_up_dwas.add(row[0])  # Escalate task to DWA!
                else:
                    unlinked_tasks.append(t_id)  # No DWA linkage, keep as granular task
            except sqlite3.Error:
                unlinked_tasks.append(t_id)

        # 2. Translate the combined DWAs
        for d_id in rolled_up_dwas:
            cursor.execute("SELECT dwa_title FROM dwa_reference WHERE dwa_id = ?", (d_id,))
            row = cursor.fetchone()
            translated["dwas"].append(row[0] if row else str(d_id))

        # 3. Translate remaining unlinked Tasks
        for t_id in unlinked_tasks:
            cursor.execute("SELECT task FROM task_statements WHERE task_id = ?", (t_id,))
            row = cursor.fetchone()
            translated["tasks"].append(row[0] if row else str(t_id))

        # 4. Translate Skills
        for s_id in master_profile.get("skills", []):
            cursor.execute("SELECT element_name FROM skills WHERE element_id = ?", (s_id,))
            row = cursor.fetchone()
            translated["skills"].append(row[0] if row else str(s_id))
        conn.close()

    except Exception as e:
        print(f"[X] Translation Error: {e}")
        translated["tasks"] = master_profile.get("tasks", [])
        translated["dwas"] = master_profile.get("dwas", [])
        translated["skills"] = master_profile.get("skills", [])

    translated["tech_skills"] = [str(tech).replace("TECH-", "") for tech in master_profile.get("tech_skills", [])]
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
            "You are the C-Arc Career Counselor, an expert, empathetic, and practical career mentor.\n"
            "Phase 1 (Discovery) is complete. You are now in Phase 2 (Counseling) helping the candidate navigate their ML-generated career matches.\n\n"
            "CORE DIRECTIVES:\n"
            "1. THE BOUNDARY: You MUST ground every piece of advice, skill gap analysis, and mock interview question STRICTLY in the candidate's specific XGBoost career matches. Never suggest alternative career paths.\n"
            "2. THE PERSONA: Speak like a real human mentor in a chat app. Be warm, encouraging, and direct.\n"
            "3. THE FORMAT: Keep responses highly concise and conversational (2-4 short sentences). Avoid robotic AI transitions, bulleted lists, or long essays unless the user explicitly asks for detailed formatting.\n"
            "4. THE GOAL: Answer the user's immediate question, then help them take actionable steps (e.g., preparing for the resume generator, interview prep) for their chosen path.\n\n"
            "Never output your internal thinking process. Output ONLY your direct dialogue."
        )

        # Safely extract the content of the very last message in the conversation
        last_msg_content = ""
        if messages:
            last_msg = messages[-1]
            last_msg_content = last_msg.get("content", "") if isinstance(last_msg, dict) else getattr(last_msg, 'content', "")

        # Initial Delivery: The Big Reveal
        if final_recs and "Profile Confirmed by User" in last_msg_content:
            user_prompt = (
                f"Conversation History:\n{chat_history}\n\n"
                f"Our XGBoost ML model generated these top career matches:\n{final_recs}\n\n"
                f"This is the big reveal! Congratulate the candidate on completing the profiling phase, gracefully present their top career matches (including scores), and ask them which path catches their eye."
            )
        # Ongoing Counseling: Dynamic Follow-ups
        else:
            user_prompt = (
                f"[SYSTEM CONTEXT - DO NOT READ THIS TO THE USER]\n"
                f"Candidate's Official XGBoost Career Matches:\n{final_recs}\n"
                f"--------------------------------------------------\n\n"
                f"Conversation History:\n{chat_history}\n\n"
                f"Respond directly to the candidate's last message. Keep the conversation flowing naturally while strictly anchoring your advice to the career matches above."
            )

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

    # Format the Document Panel summary (only during validation)
    if mode == "validation":
        readable_data = translate_onet_ids(master_profile)

        # Combine and format skills as bullets (Eliminates exact string duplicates)
        all_skills = list(dict.fromkeys(readable_data["skills"] + readable_data["tech_skills"]))
        skills_display = "\n".join([f"* {s}" for s in all_skills]) if all_skills else "* None"

        # Combine and format DWAs and remaining Tasks as bullets (Eliminates exact string duplicates)
        all_experience = list(dict.fromkeys(readable_data["dwas"] + readable_data["tasks"]))
        exp_display = "\n".join([f"* {e}" for e in all_experience]) if all_experience else "* None"

        # Format Education as bullets
        edu_list = [f"{ed.get('degree') or 'Unknown Degree'} in {ed.get('major') or 'Unknown Major'}" for ed in master_profile.get('education', [])]
        edu_display = "\n".join([f"* {ed}" for ed in edu_list]) if edu_list else "* Not provided"

        profile_summary = (
            f"### 📋 Candidate Profile Overview\n\n"
            f"**Name:** {master_profile.get('basic_info', {}).get('full_name') or 'Not provided'}\n\n"
            f"**Location:** {master_profile.get('basic_info', {}).get('location') or 'Not provided'}\n\n"
            f"**Contact:** {master_profile.get('basic_info', {}).get('email') or 'N/A'} | {master_profile.get('basic_info', {}).get('phone') or 'N/A'}\n\n"
            f"### 🎓 Education\n"
            f"{edu_display}\n\n"
            f"### 💻 Technical & Professional Skills\n"
            f"{skills_display}\n\n"
            f"### 🏢 Work Responsibilities & Activities\n"
            f"{exp_display}\n\n"
            f"---\n"
            f"*Please review these details. If everything is correct, click **Confirm & Proceed** below.*"
        )

        # Return the chat response normally, but route the summary directly to its specific state variable
        return {
            "messages": [{"role": "assistant", "content": response}],
            "profile_summary": profile_summary
        }

    # Default return for Interviewer and Counselor modes
    return {
        "messages": [{"role": "assistant", "content": response}]
    }