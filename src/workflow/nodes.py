import json
from .state import CArcState
from utils.llm_provider import LocalLLMProvider

# Initialize the local client
llm = LocalLLMProvider()


def mentor_node(state: CArcState) -> dict:
    """Conversational interface powered by local Gemma 4 (Dual-Mode)."""
    ocean = state.get("ocean_vector", {})
    turn = state.get("turn_count", 0)

    # Default to counselor if no mode is explicitly set
    mode = state.get("mentor_mode", "counselor")

    if mode == "counselor":
        system_prompt = (
            "You are the C-Arc Career Counselor. Your role is to provide empathetic, "
            "action-oriented career coaching. Use strategic nudging to encourage the user "
            "to reveal their professional skills and working style."
        )
    elif mode == "interviewer":
        system_prompt = (
            "You are the C-Arc Career Interviewer. The system requires "
            "specific factual data to build the user's Master Profile. Stop being conversational. "
            "Directly and politely ask the user to list their specific technical skills, tools, "
            "and past job titles."
        )

    user_prompt = f"User Profile: OCEAN={ocean}. Turn={turn}. Please respond to the candidate."

    response = llm.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model="google/gemma-4-E4B-it",
        temperature=0.7
    )
    return {
        "messages": [{"role": "assistant", "content": response}],
        "turn_count": 1
    }


def profiler_node(state: CArcState) -> dict:
    """Silent psychometrics powered by Logit-Based Rating Engine."""
    messages = state.get("messages", [])
    turn_count = state.get("turn_count", 1)
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
    """Information retrieval and ONET grounding."""
    return {}


def career_expert_node(state: CArcState) -> dict:
    """Final XGBoost inference and recommendation generation."""
    return {}


def evaluate_state_node(state: CArcState) -> dict:
    """Pass-through node to funnel workers into the Evaluator Router."""
    return {}
