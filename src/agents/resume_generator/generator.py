import json
from workflow.state import CArcState
from utils.utils import translate_onet_ids


def generate_resume(state: CArcState, llm_client) -> dict:
    """Takes the master profile and a target career, and outputs a tailored Markdown resume."""
    messages = state.get("messages", [])
    master_profile = state.get("master_profile", {})

    # 1. Pull the XGBoost recommendations that the user is referencing
    recommendations = state.get("final_recommendations", "No specific recommendations found.")

    # 2. Capture the user's vague or specific selection (e.g., "the first one")
    last_msg = messages[-1] if messages else {}
    user_choice = (last_msg.get("content", "") if isinstance(last_msg, dict) else last_msg.content).strip()

    # 3. Pre-translate O*NET IDs into human-readable text before passing to the LLM
    translated_data = translate_onet_ids(master_profile)
    readable_profile = {
        "basic_info": master_profile.get("basic_info", {}),
        "education": master_profile.get("education", []),
        "work_experience": list(dict.fromkeys(
            translated_data.get("tasks", []) + translated_data.get("dwas", []) + translated_data.get("work_activities", []))),
        "skills": list(dict.fromkeys(translated_data.get("skills", []) + translated_data.get("tech_skills", [])))
    }

    # 4. Inject into the reasoning prompt without asking the LLM to translate IDs
    prompt = (
        f"You are an Expert Executive Resume Writer.\n\n"
        f"CONTEXT:\n"
        f"The candidate was presented with these career recommendations:\n{recommendations}\n\n"
        f"The candidate selected their target career with this statement: \"{user_choice}\"\n\n"
        f"INSTRUCTIONS:\n"
        f"Step 1: Logically deduce which specific career path the candidate is choosing.\n"
        f"Step 2: Write a highly professional, ATS-friendly resume tailored for that role.\n\n"
        f"Use ONLY the following verified candidate data. Do not invent experience or add fake placeholder companies/dates:\n"
        f"{json.dumps(readable_profile, indent=2)}\n\n"
        f"STRICT FORMATTING RULES:\n"
        f"1. Output ONLY the final Markdown resume. Do not include your deduction steps.\n"
        f"2. Do not include any introductory greetings or concluding notes.\n"
        f"3. Start immediately with the Candidate's Name as an H1 Header (#).\n"
        f"4. Include a Professional Summary.\n"
        f"5. Format the provided work experience into professional, action-oriented bullet points."
    )

    print(f"\n[➔] STARTING NODE: resume_agent | Deduce user intent: '{user_choice}'")

    try:
        # Use deepseek-reasoner or deepseek-chat. Chat is usually smart enough for this deduction and much faster.
        response = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        resume_md = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Resume Generation Error: {e}")
        resume_md = "Sorry, I encountered an error generating your resume."

    return {
        "resume_content": resume_md,
        "mentor_mode": "counselor",
        "messages": [{"role": "assistant", "content": "I have successfully generated your tailored resume based on your selection! You can view and copy it from the Document Panel."}]
    }
