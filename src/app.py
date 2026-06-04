import gradio as gr
from workflow.graph import build_graph
from workflow.nodes import llm
import warnings

# Suppress verbose Hugging Face/Torch warnings
warnings.filterwarnings("ignore")

print("Compiling C-Arc Graph...")
app = build_graph()

print("Preloading models into VRAM...")
# Eager load the models before the UI launches
llm.preload_models([
    "google/gemma-4-E4B-it",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
])


def add_user_message(user_input, c_arc_state):
    """Captures user input, builds context prompt, and updates UI instantly."""
    if not user_input.strip():
        return c_arc_state.get("messages", []), c_arc_state, ""

    # Update state with user message
    c_arc_state["messages"].append({"role": "user", "content": user_input})

    # Increment the turn count exactly once per user interaction
    c_arc_state["turn_count"] += 1

    # Return immediately to update the chatbox and clear the text input
    return c_arc_state["messages"], c_arc_state, ""


def handle_profile_confirmation(c_arc_state):
    """Triggered when the user clicks the verification button."""
    c_arc_state["profile_verified"] = True
    c_arc_state["messages"].append({
        "role": "user",
        "content": "[Profile Confirmed by User. Transitioning to Career Expert Inference Engine.]"
    })
    c_arc_state["turn_count"] += 1
    return c_arc_state["messages"], c_arc_state, gr.update(interactive=False, variant="success")


def run_graph(c_arc_state):
    """Executes the LangGraph workflow after the UI has updated."""
    # Prevent running if the last message isn't from the user
    if not c_arc_state["messages"] or c_arc_state["messages"][-1]["role"] != "user":
        return c_arc_state.get("messages", []), c_arc_state, c_arc_state.get("ocean_vector", {})

    config = {"recursion_limit": 15}

    try:
        # Run graph for one conversational turn
        for output in app.stream(c_arc_state, config=config):
            for node_name, state_update in output.items():
                if not state_update:
                    continue

                if "messages" in state_update and state_update['messages']:
                    last_msg = state_update['messages'][-1]
                    if last_msg['role'] == 'assistant':
                        c_arc_state["messages"].append(last_msg)

                if "ocean_vector" in state_update:
                    c_arc_state["ocean_vector"] = state_update["ocean_vector"]
                if "ocean_history" in state_update:
                    c_arc_state["ocean_history"] = state_update["ocean_history"]
                if "ocean_hits" in state_update:
                    c_arc_state["ocean_hits"] = state_update["ocean_hits"]
                if "cumulative_confidence" in state_update:
                    c_arc_state["cumulative_confidence"] = state_update["cumulative_confidence"]
                if "master_profile" in state_update:
                    c_arc_state["master_profile"] = state_update["master_profile"]
                if "mentor_mode" in state_update:
                    c_arc_state["mentor_mode"] = state_update["mentor_mode"]
                if "final_recommendations" in state_update:
                    c_arc_state["final_recommendations"] = state_update["final_recommendations"]

        is_validating = (c_arc_state.get("mentor_mode") == "validation")
        print(f"\n[DEBUG UI] Mentor Mode: '{c_arc_state.get('mentor_mode')}' | Showing Button: {is_validating}")

        if is_validating:
            button_update = gr.update(interactive=True, variant="primary")
        else:
            button_update = gr.update(interactive=False, variant="success")

        return c_arc_state["messages"], c_arc_state, c_arc_state["ocean_vector"], button_update

    except Exception as e:
        import traceback
        error_msg = f"❌ Execution crashed:\n{traceback.format_exc()}"
        c_arc_state["messages"].append({"role": "assistant", "content": error_msg})
        return c_arc_state["messages"], c_arc_state, c_arc_state.get("ocean_vector", {}), gr.update(interactive=False)


# Define the Gradio Interface
with gr.Blocks(title="C-Arc", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 C-Arc")

    # Initialize the LangGraph state securely in the background
    initial_state = gr.State({
        "messages": [{"role": "assistant", "content": "Hi there! I am the C-Arc Career Counselor. How can I help you today?"}],
        "master_profile": {
            "tasks": [],
            "dwas": [],
            "skills": [],
            "tech_skills": [],
            "basic_info": {
                "full_name": None,
                "email": None,
                "phone": None,
                "location": None
            },
            "education": []
        },
        "ocean_vector": {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5},
        "ocean_history": [],
        "ocean_hits": {"O": 0, "C": 0, "E": 0, "A": 0, "N": 0},
        "cumulative_confidence": 0.0,
        "turn_count": 0,
        "mentor_mode": "interviewer",
        "profile_verified": False
    })

    with gr.Row():
        # Chat Interface Column
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": "Hi there! I am the C-Arc Career Counselor. How can I help you today?"}],
                height=600
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your response here...",
                    show_label=False,
                    scale=4
                )
                submit_btn = gr.Button("Send", scale=1, variant="primary")

        # Telemetry & State Column
        with gr.Column(scale=1):
            gr.Markdown("### Live Telemetry")
            ocean_display = gr.JSON(
                value={"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5},
                label="DeepSeek Profiler (OCEAN)"
            )

            # Optional button to reset the state without restarting the server
            clear_btn = gr.Button("Reset Conversation")

            confirm_profile_btn = gr.Button(
                "✅ Confirm & Proceed to Career Recommendations",
                variant="success",
                interactive=False,
                visible=True
            )


            def reset_state():
                fresh_state = {
                    "messages": [{"role": "assistant", "content": "Hi there! I am the C-Arc Career Counselor. How can I help you today?"}],
                    "master_profile": {
                        "tasks": [],
                        "dwas": [],
                        "skills": [],
                        "tech_skills": [],
                        "basic_info": {
                            "full_name": None,
                            "email": None,
                            "phone": None,
                            "location": None
                        },
                        "education": []
                    },
                    "ocean_vector": {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5},
                    "ocean_history": [],
                    "ocean_hits": {"O": 0, "C": 0, "E": 0, "A": 0, "N": 0},
                    "cumulative_confidence": 0.0,
                    "turn_count": 0,
                    "mentor_mode": "interviewer",
                    "profile_verified": False
                }
                return fresh_state["messages"], fresh_state, fresh_state["ocean_vector"]


            clear_btn.click(reset_state, inputs=[], outputs=[chatbot, initial_state, ocean_display])

    # Wire up the text box
    msg.submit(
        add_user_message,
        inputs=[msg, initial_state],
        outputs=[chatbot, initial_state, msg]
    ).then(
        run_graph,
        inputs=[initial_state],
        outputs=[chatbot, initial_state, ocean_display, confirm_profile_btn]
    )

    submit_btn.click(
        add_user_message,
        inputs=[msg, initial_state],
        outputs=[chatbot, initial_state, msg]
    ).then(
        run_graph,
        inputs=[initial_state],
        outputs=[chatbot, initial_state, ocean_display, confirm_profile_btn]
    )

    confirm_profile_btn.click(
        handle_profile_confirmation,
        inputs=[initial_state],
        outputs=[chatbot, initial_state, confirm_profile_btn]
    ).then(
        run_graph,
        inputs=[initial_state],
        outputs=[chatbot, initial_state, ocean_display, confirm_profile_btn]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
