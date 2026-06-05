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


def handle_resume_initiation(c_arc_state):
    """Triggered when the user clicks the Generate Resume button."""
    c_arc_state["mentor_mode"] = "generate_resume"
    c_arc_state["messages"].append({
        "role": "user",
        "content": "[Generate Tailored Resume]",
    })
    c_arc_state["messages"].append({
        "role": "assistant",
        "content": "Awesome! Let's get your resume ready. **Which of the recommended career paths would you like to target?**"
    })
    return c_arc_state["messages"], c_arc_state, gr.update(interactive=False)


def run_graph(c_arc_state):
    """Executes the LangGraph workflow after the UI has updated."""
    # Prevent running if the last message isn't from the user
    if not c_arc_state["messages"] or c_arc_state["messages"][-1]["role"] != "user":
        return (c_arc_state.get("messages", []), c_arc_state, c_arc_state, gr.update(), gr.update(),
                c_arc_state.get("profile_summary", "*Your profile summary will appear here during validation.*"),
                c_arc_state.get("final_recommendations", "*Your XGBoost career matches will appear here.*"),
                c_arc_state.get("resume_content", "*Your tailored resume will appear here after generation.*"))

    config = {"recursion_limit": 15}

    try:
        # Run graph for one conversational turn
        for output in app.stream(c_arc_state, config=config):
            for node_name, state_update in output.items():
                if not state_update:
                    continue

                for key, val in state_update.items():
                    if key == "messages" and val:
                        last_msg = val[-1]
                        if last_msg['role'] == 'assistant':
                            c_arc_state["messages"].append(last_msg)
                    elif key != "messages":
                        c_arc_state[key] = val

        is_validating = (c_arc_state.get("mentor_mode") == "validation")
        is_counselor = (c_arc_state.get("mentor_mode") == "counselor")

        if is_validating:
            button_update = gr.update(interactive=True, variant="primary")
        else:
            button_update = gr.update(interactive=False, variant="success")

        if is_counselor:
            resume_btn_update = gr.update(interactive=True, variant="primary")
        else:
            resume_btn_update = gr.update(interactive=False, variant="success")

        current_profile = c_arc_state.get("profile_summary", "*Your profile summary will appear here during validation.*")
        current_recs = c_arc_state.get("final_recommendations", "*Your XGBoost career matches will appear here.*")
        current_resume = c_arc_state.get("resume_content", "*Your tailored resume will appear here after generation.*")

        return c_arc_state["messages"], c_arc_state, c_arc_state, button_update, resume_btn_update, current_profile, current_recs, current_resume

    except Exception as e:
        import traceback
        error_msg = f"❌ Execution crashed:\n{traceback.format_exc()}"
        c_arc_state["messages"].append({"role": "assistant", "content": error_msg})
        return (c_arc_state["messages"], c_arc_state, c_arc_state, gr.update(interactive=False), gr.update(interactive=False),
                c_arc_state.get("profile_summary", ""), c_arc_state.get("final_recommendations", ""), c_arc_state.get("resume_content", ""))

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
        "trait_maturity": {"O": 0.0, "C": 0.0, "E": 0.0, "A": 0.0, "N": 0.0},
        "turn_count": 0,
        "mentor_mode": "interviewer",
        "profile_verified": False,
        "missing_demographics": [],
        "weak_ocean_traits": [],
        "ir_last_extracted_index": 0,
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
            gr.Markdown("---")
            with gr.Tabs():
                with gr.Tab("📋 Profile Summary"):
                    profile_display = gr.Markdown(value="*Your profile summary will appear here during validation.*")
                with gr.Tab("🎯 Career Matches"):
                    recommendations_display = gr.Markdown(value="*Your XGBoost career matches will appear here.*")
                with gr.Tab("📄 Tailored Resume"):
                    resume_display = gr.Markdown(value="*Your tailored resume will appear here after generation.*")

        # Telemetry & State Column
        with gr.Column(scale=1):
            gr.Markdown("### Live Telemetry")
            state_display = gr.JSON(
                value=initial_state.value,
                label="Global C-Arc State (Debug)"
            )

            clear_btn = gr.Button("Reset Conversation")

            confirm_profile_btn = gr.Button(
                "✅ Confirm & Proceed to Career Recommendations",
                variant="success",
                interactive=False,
                visible=True
            )

            generate_resume_btn = gr.Button(
                "📄 Generate Tailored Resume",
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
                    "trait_maturity": {"O": 0.0, "C": 0.0, "E": 0.0, "A": 0.0, "N": 0.0},
                    "turn_count": 0,
                    "mentor_mode": "interviewer",
                    "profile_verified": False,
                    "missing_demographics": [],
                    "weak_ocean_traits": [],
                    "ir_last_extracted_index": 0,
                }
                return fresh_state["messages"], fresh_state, fresh_state


            clear_btn.click(reset_state, inputs=[], outputs=[chatbot, initial_state, state_display])

    # Wire up the text box
    msg.submit(
        add_user_message,
        inputs=[msg, initial_state],
        outputs=[chatbot, initial_state, msg]
    ).then(
        run_graph,
        inputs=[initial_state],
        outputs=[chatbot, initial_state, state_display, confirm_profile_btn, generate_resume_btn,
                 profile_display, recommendations_display, resume_display]
    )

    submit_btn.click(
        add_user_message,
        inputs=[msg, initial_state],
        outputs=[chatbot, initial_state, msg]
    ).then(
        run_graph,
        inputs=[initial_state],
        outputs=[chatbot, initial_state, state_display, confirm_profile_btn, generate_resume_btn,
                 profile_display, recommendations_display, resume_display]
    )

    confirm_profile_btn.click(
        handle_profile_confirmation,
        inputs=[initial_state],
        outputs=[chatbot, initial_state, confirm_profile_btn]
    ).then(
        run_graph,
        inputs=[initial_state],
        outputs=[chatbot, initial_state, state_display, confirm_profile_btn, generate_resume_btn,
                 profile_display, recommendations_display, resume_display]
    )

    generate_resume_btn.click(
        handle_resume_initiation,
        inputs=[initial_state],
        outputs=[chatbot, initial_state, generate_resume_btn]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
