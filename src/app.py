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


def chat_step(user_input, c_arc_state):
    if not user_input.strip():
        return c_arc_state.get("messages", []), c_arc_state, "", c_arc_state.get("ocean_vector", {})

    # 1. Update state with user message
    c_arc_state["messages"].append({"role": "user", "content": user_input})

    config = {"recursion_limit": 15}

    try:
        # 2. Run graph for one conversational turn
        for output in app.stream(c_arc_state, config=config):
            for node_name, state_update in output.items():
                if not state_update:
                    continue

                # Capture Mentor dialogue directly into state
                if "messages" in state_update and state_update['messages']:
                    last_msg = state_update['messages'][-1]
                    if last_msg['role'] == 'assistant':
                        c_arc_state["messages"].append(last_msg)

                # Capture Profiler updates
                if "ocean_vector" in state_update:
                    c_arc_state["ocean_vector"] = state_update["ocean_vector"]

                # Accumulate the turn counter
                if "turn_count" in state_update:
                    c_arc_state["turn_count"] += state_update["turn_count"]

        # 3. Return the native LangGraph messages directly to the UI
        return c_arc_state["messages"], c_arc_state, "", c_arc_state["ocean_vector"]

    except Exception as e:
        import traceback
        error_msg = f"❌ Execution crashed:\n{traceback.format_exc()}"
        c_arc_state["messages"].append({"role": "assistant", "content": error_msg})
        return c_arc_state["messages"], c_arc_state, "", c_arc_state.get("ocean_vector", {})


# Define the Gradio Interface
with gr.Blocks(title="C-Arc Test Interface", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 C-Arc Phase 4: Dual-Mode Mentor & Profiler")

    # Initialize the LangGraph state securely in the background
    initial_state = gr.State({
        "messages": [],
        "master_profile": {"skills": []},
        "ocean_vector": {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5},
        "ocean_history": [],
        "turn_count": 0,
        "mentor_mode": "counselor"  # Defaulting to counselor mode
    })

    with gr.Row():
        # Chat Interface Column
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=600)
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
                    "messages": [],
                    "master_profile": {"skills": []},
                    "ocean_vector": {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5},
                    "ocean_history": [],
                    "turn_count": 0,
                    "mentor_mode": "counselor"
                }
                return [], fresh_state, fresh_state["ocean_vector"]


            clear_btn.click(reset_state, inputs=[], outputs=[chatbot, initial_state, ocean_display])

    # Wire up the text box and submit button
    msg.submit(
        chat_step,
        inputs=[msg, initial_state],
        outputs=[chatbot, initial_state, msg, ocean_display]
    )
    submit_btn.click(
        chat_step,
        inputs=[msg, initial_state],
        outputs=[chatbot, initial_state, msg, ocean_display]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
