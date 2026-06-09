from workflow.state import CArcState
from utils.utils import translate_onet_ids


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
            "You are the C-Arc Career Counselor, an empathetic, insightful, and highly conversational career mentor.\n"
            "Phase 1 (Discovery) is complete. You are now in Phase 2 (Counseling) helping the candidate navigate their ML-generated career matches.\n\n"
            "CORE DIRECTIVES:\n"
            "1. THE PERSONA (DEFAULT MODE): Act like a real human chatting in a messaging app. Your default responses MUST be short, warm, and conversational (1-3 sentences maximum). Do not use bullet points or long paragraphs. Always end with a single, gentle question to keep the dialogue moving.\n"
            "2. THE EMPATHY MODE (HANDLING DOUBT): If the user expresses self-doubt, anxiety, or feels overwhelmed, temporarily stop explaining data. Prioritize emotional reassurance. Validate their feelings, calm them down, and gently build their confidence by reminding them of their proven capabilities. Be highly supportive and human.\n"
            "3. THE EXPLANATION MODE (ONLY WHEN ASKED): IF and ONLY IF the user explicitly asks 'why' they fit a role or requests a detailed explanation, you may provide a deeper analysis. Use a brief intro, 2-3 concise bullet points mapping their specific extracted skills and past experiences to the role, and a closing question.\n"
            "4. THE BOUNDARY: Ground every piece of advice and explanation STRICTLY in the candidate's specific XGBoost career matches and their extracted profile data. Do not invent or assume skills they have not provided.\n\n"
            "FORMATTING RULE: You must respond ONLY with the direct dialogue intended for the user. Begin your response immediately with your conversational reply."
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
                f"Our ML model generated these top career matches:\n{final_recs}\n\n"
                f"This is the big reveal! Congratulate the candidate on completing the profiling phase, gracefully present their top career matches (including scores), and ask them which path catches their eye."
            )
        # Ongoing Counseling: Dynamic Follow-ups
        else:
            readable_data = translate_onet_ids(master_profile)

            # Combine and format skills as bullets (Eliminates exact string duplicates)
            all_skills = list(dict.fromkeys(readable_data["skills"] + readable_data["tech_skills"]))
            extracted_skills = ", ".join([f"{s}" for s in all_skills]) if all_skills else "None"

            # Combine and format DWAs and remaining Tasks as bullets (Eliminates exact string duplicates)
            all_experience = list(dict.fromkeys(
                readable_data["work_activities"] +
                readable_data["dwas"] +
                readable_data["tasks"]
            ))
            extracted_exp = ", ".join([f"{e}" for e in all_experience]) if all_experience else "None"

            user_prompt = (
                f"[SYSTEM CONTEXT - DO NOT READ THIS TO THE USER]\n"
                f"Candidate's Official Career Matches:\n{final_recs}\n\n"
                f"Candidate's Extracted Profile Data (Use this to explain WHY they fit a role):\n"
                f"- Candidate OCEAN personality score: {ocean}"
                f"- Skills: {extracted_skills}"
                f"- Experience: {extracted_exp}"
                f"--------------------------------------------------\n\n"
                f"Conversation History:\n{chat_history}\n\n"
                f"Respond directly to the candidate's last message. If they ask for an explanation of a role fit, provide a scannable, bulleted analysis mapping their specific extracted data to that role."
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

    # Fallback to strip out "thought" blocks if Gemma hallucinates one
    if response.lower().startswith("thought"):
        # Split by double newline and grab the last block, which is usually the actual response
        parts = response.split("\n\n")
        response = parts[-1].strip() if len(parts) > 1 else response

    # Format the Document Panel summary (only during validation)
    if mode == "validation":
        readable_data = translate_onet_ids(master_profile)

        # Combine and format skills as bullets (Eliminates exact string duplicates)
        all_skills = list(dict.fromkeys(readable_data["skills"] + readable_data["tech_skills"]))
        skills_display = "\n".join([f"* {s}" for s in all_skills]) if all_skills else "* None"

        # Combine and format DWAs and remaining Tasks as bullets (Eliminates exact string duplicates)
        all_experience = list(dict.fromkeys(
            readable_data["work_activities"] +
            readable_data["dwas"] +
            readable_data["tasks"]
        ))
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