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

        return c_arc_state["messages"], c_arc_state, c_arc_state["ocean_vector"]

    except Exception as e:
        import traceback
        error_msg = f"❌ Execution crashed:\n{traceback.format_exc()}"
        c_arc_state["messages"].append({"role": "assistant", "content": error_msg})
        return c_arc_state["messages"], c_arc_state, c_arc_state.get("ocean_vector", {})


# Define the Gradio Interface
with gr.Blocks(title="C-Arc", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 C-Arc")

    # Initialize the LangGraph state securely in the background
    initial_state = gr.State({
        "messages": [{"role": "assistant", "content": "Hi there! I am the C-Arc Career Counselor. How can I help you today?"}],
        "master_profile": {"skills": []},
        "ocean_vector": {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5},
        "ocean_history": [],
        "turn_count": 0,
        "mentor_mode": "interviewer"  # Defaulting to interviewer mode
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


            def reset_state():
                fresh_state = {
                    "messages": [{"role": "assistant", "content": "Hi there! I am the C-Arc Career Counselor. How can I help you today?"}],
                    "master_profile": {"skills": []},
                    "ocean_vector": {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5},
                    "ocean_history": [],
                    "turn_count": 0,
                    "mentor_mode": "interviewer"
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
        outputs=[chatbot, initial_state, ocean_display]
    )

    # Wire up the submit button
    submit_btn.click(
        add_user_message,
        inputs=[msg, initial_state],
        outputs=[chatbot, initial_state, msg]
    ).then(
        run_graph,
        inputs=[initial_state],
        outputs=[chatbot, initial_state, ocean_display]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
