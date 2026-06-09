import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def format_dashboard_markdown(state):
    """Converts the raw LangGraph state into a clean, readable Markdown dashboard."""

    # Safely extract system states
    mentor_mode = state.get("mentor_mode", "Unknown").replace("_", " ").title()
    turn_count = state.get("turn_count", 0)
    is_verified = "✅ Verified" if state.get("profile_verified") else "⏳ Pending"

    # Extract psychometrics
    ocean = state.get("ocean_vector", {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5})
    trait_maturity = state.get("trait_maturity", {"O": 0, "C": 0, "E": 0, "A": 0, "N": 0})

    # Extract master profile
    profile = state.get("master_profile", {})
    basic = profile.get("basic_info", {})
    name = basic.get("full_name") or "Not provided"
    email = basic.get("email") or "Not provided"
    phone = basic.get("phone") or "N/A"
    location = basic.get("location") or "N/A"
    education = profile.get("education") or "Not provided"

    tasks = profile.get("tasks", [])
    tasks_display = ", ".join([f"{e}" for e in tasks]) if tasks else "*None*"
    dwas = profile.get("dwas", [])
    dwas_display = ", ".join([f"{e}" for e in dwas]) if dwas else "*None*"
    work_activities = profile.get("work_activities", [])
    work_activities_display = ", ".join([f"{e}" for e in work_activities]) if work_activities else "*None*"

    skills = profile.get("skills", [])
    skills_display = ", ".join([f"{e}" for e in skills]) if skills else "*None*"
    tech_skills = profile.get("tech_skills", [])
    tech_display = ", ".join([f"{e}" for e in tech_skills]) if tech_skills else "*None*"

    missing_demo = state.get("missing_demographics", [])
    missing_display = ", ".join(missing_demo).title() if missing_demo else "*None*"

    # Build the Markdown string
    md_content = f"""
## ⚙️ System Status

**Active Agent:** `{mentor_mode}` | **Turn Count:** `{turn_count}` | **Profile Status:** {is_verified}

---
## 🧠 Psychometric Vector (OCEAN)
| Trait | Score (0.0 - 1.0) |
| :--- | :--- |
| **Openness (O)** | {ocean.get('O', 0.5):.2f} |
| **Conscientiousness (C)** | {ocean.get('C', 0.5):.2f} |
| **Extraversion (E)** | {ocean.get('E', 0.5):.2f} |
| **Agreeableness (A)** | {ocean.get('A', 0.5):.2f} |
| **Neuroticism (N)** | {ocean.get('N', 0.5):.2f} |

## Trait Maturity
| Trait | Score (0.0 - 1.0) |
| :--- | :--- |
| **Openness (O)** | {trait_maturity.get('O', 0):.2f} |
| **Conscientiousness (C)** | {trait_maturity.get('C', 0):.2f} |
| **Extraversion (E)** | {trait_maturity.get('E', 0):.2f} |
| **Agreeableness (A)** | {trait_maturity.get('A', 0):.2f} |
| **Neuroticism (N)** | {trait_maturity.get('N', 0):.2f} |
---
## 👤 User Demographics
* **Name:** {name}
* **Email:** {email}
* **Phone:** {phone}
* **Location:** {location}
* **Education Background:** {education}
* **Currently Missing Data:** {missing_display}
---
## 🛠️ Extracted Competencies
---
* **Recorded Skills:** {len(profile.get("skills", []))} entries
* **Recorded Technical Skills:** {len(profile.get("tech_skills", []))} entries
* **Recorded Work Activities:** {len(profile.get("work_activities", []))} entries
* **Recorded Detail Work Activities:** {len(profile.get("dwas", []))} entries
* **Recorded Tasks:** {len(profile.get("tasks", []))} entries
---
* **O*NET Skills:** {skills_display}
* **O*NET Technical Skills:** {tech_display}
* **O*NET Work Activities:** {work_activities_display}
* **O*NET Detail Work Activities:** {dwas_display}
* **O*NET Tasks:** {tasks_display}
    """

    return md_content


def generate_ocean_plot(ocean_history):
    """Generates a matplotlib line plot of the OCEAN trait history."""
    fig, ax = plt.subplots(figsize=(8, 4))

    if not ocean_history:
        ax.set_title("OCEAN Trait History (No data yet)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig

    turns = list(range(1, len(ocean_history) + 1))
    traits = {'O': 'Openness', 'C': 'Conscientiousness', 'E': 'Extraversion', 'A': 'Agreeableness', 'N': 'Neuroticism'}
    colors = {'O': '#1f77b4', 'C': '#2ca02c', 'E': '#d62728', 'A': '#9467bd', 'N': '#ff7f0e'}

    for key, label in traits.items():
        scores = [history_step.get(key, 0.5) for history_step in ocean_history]
        ax.plot(turns, scores, marker='o', label=label, color=colors[key])

    ax.set_title("OCEAN Trait Evolution")
    ax.set_xlabel("Update Step")
    ax.set_ylabel("Score (0.0 - 1.0)")
    ax.set_ylim(0, 1.1)

    # Ensure x-axis only shows integer ticks for steps
    ax.set_xticks(turns)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()

    return fig


def add_user_message(user_input, c_arc_state):
    """Captures user input, builds context prompt, and updates UI instantly."""
    if not user_input.strip():
        return c_arc_state.get("messages", []), c_arc_state, gr.update(), gr.update(), gr.update()

    # Update state with user message
    c_arc_state["messages"].append({"role": "user", "content": user_input})

    # Increment the turn count exactly once per user interaction
    c_arc_state["turn_count"] += 1

    # Return immediately to update the chatbox and clear the text input
    return c_arc_state["messages"], c_arc_state, gr.update(value="", interactive=False), gr.update(interactive=False), gr.update(interactive=False)


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
        dashboard_view = format_dashboard_markdown(c_arc_state)
        ocean_plot = generate_ocean_plot(c_arc_state.get("ocean_history", []))
        return (c_arc_state.get("messages", []), c_arc_state, dashboard_view, ocean_plot, gr.update(), gr.update(),
                c_arc_state.get("knn_rec", ""), c_arc_state.get("final_rec", ""),
                c_arc_state.get("profile_summary", "*Your profile summary will appear here during validation.*"),
                c_arc_state.get("final_recommendations", "*Your career matches will appear here.*"),
                c_arc_state.get("resume_content", "*Your tailored resume will appear here after generation.*"),
                gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True))

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

        current_knn_rec = c_arc_state.get("knn_rec", "")
        current_final_rec = c_arc_state.get("final_rec", "")
        current_profile = c_arc_state.get("profile_summary", "*Your profile summary will appear here during validation.*")
        current_recs = c_arc_state.get("final_recommendations", "*Your career matches will appear here.*")
        current_resume = c_arc_state.get("resume_content", "*Your tailored resume will appear here after generation.*")

        dashboard_view = format_dashboard_markdown(c_arc_state)
        ocean_plot = generate_ocean_plot(c_arc_state.get("ocean_history", []))

        return (c_arc_state["messages"], c_arc_state, dashboard_view, ocean_plot, button_update, resume_btn_update,
                current_knn_rec, current_final_rec,
                current_profile, current_recs, current_resume,
                gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True))

    except Exception as e:
        import traceback
        error_msg = f"❌ Execution crashed:\n{traceback.format_exc()}"
        c_arc_state["messages"].append({"role": "assistant", "content": error_msg})
        dashboard_view = format_dashboard_markdown(c_arc_state)
        ocean_plot = generate_ocean_plot(c_arc_state.get("ocean_history", []))
        return (c_arc_state["messages"], c_arc_state, dashboard_view, ocean_plot, gr.update(interactive=False), gr.update(interactive=False),
                c_arc_state.get("knn_rec", ""), c_arc_state.get("final_rec", ""),
                c_arc_state.get("profile_summary", ""), c_arc_state.get("final_recommendations", ""), c_arc_state.get("resume_content", ""),
                gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True))

# Define the Gradio Interface
with gr.Blocks(title="C-Arc", theme=gr.themes.Soft(), css=".markdown-text { font-size: 18px; }") as demo:
    gr.Markdown("# 🧠 C-Arc")

    # Initialize the LangGraph state securely in the background
    initial_state = gr.State({
        "messages": [{"role": "assistant", "content": "Hi there! I am the C-Arc Career Counselor. How can I help you today?"}],
        "master_profile": {
            "tasks": [],
            "dwas": [],
            "work_activities": [],
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
        "ocean_history": [{"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5}],
        "trait_maturity": {"O": 0.0, "C": 0.0, "E": 0.0, "A": 0.0, "N": 0.0},
        "turn_count": 0,
        "mentor_mode": "interviewer",
        "profile_verified": False,
        "missing_demographics": [],
        "weak_ocean_traits": [],
        "ir_last_extracted_index": 0,
        "knn_rec": "",
        "final_rec": "",
        "profile_summary": "*Your profile summary will appear here during validation.*",
        "target_career": "*Your career matches will appear here.*",
        "resume_content": "*Your tailored resume will appear here after generation.*"
    })

    with gr.Row():
        # Chat Interface Column
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": "Hi there! I am the C-Arc Career Counselor. How can I help you today?"}],
                height=600
            )
            with gr.Row():
                with gr.Column(scale=4):
                    msg = gr.Textbox(
                        placeholder="Type your response here...",
                        show_label=False,
                        lines=4
                    )
                with gr.Column(scale=1):
                    with gr.Row():
                        submit_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Reset Conversation")
                    with gr.Row():
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
            gr.Markdown("---")
            with gr.Tabs():
                with gr.Tab("📊 Telemetry Dashboard"):
                    ocean_plot_display = gr.Plot(
                        value=generate_ocean_plot(initial_state.value.get("ocean_history", [])),
                        label="OCEAN Trait History"
                    )
                    state_display = gr.Markdown(
                        value=format_dashboard_markdown(initial_state.value)
                    )
                with gr.Tab("📋 Profile Summary"):
                    profile_display = gr.Markdown(value="*Your profile summary will appear here during validation.*")
                with gr.Tab("🎯 Career Matches"):
                    recommendations_display = gr.Markdown(value="*Your career matches will appear here.*")
                    gr.Markdown("---")
                    gr.Markdown("### DEBUG")
                    gr.Markdown("---")
                    knn_rec_display = gr.Markdown(value="")
                    gr.Markdown("---")
                    final_rec_display = gr.Markdown(value="")
                with gr.Tab("📄 Tailored Resume"):
                    resume_display = gr.Markdown(value="*Your tailored resume will appear here after generation.*")


            def reset_state():
                fresh_state = {
                    "messages": [{"role": "assistant", "content": "Hi there! I am the C-Arc Career Counselor. How can I help you today?"}],
                    "master_profile": {
                        "tasks": [],
                        "dwas": [],
                        "work_activities": [],
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
                    "ocean_history": [{"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5}],
                    "trait_maturity": {"O": 0.0, "C": 0.0, "E": 0.0, "A": 0.0, "N": 0.0},
                    "turn_count": 0,
                    "mentor_mode": "interviewer",
                    "profile_verified": False,
                    "missing_demographics": [],
                    "weak_ocean_traits": [],
                    "ir_last_extracted_index": 0,
                    "knn_rec": "",
                    "final_rec": "",
                    "profile_summary": "*Your profile summary will appear here during validation.*",
                    "target_career": "*Your career matches will appear here.*",
                    "resume_content": "*Your tailored resume will appear here after generation.*"
                }
                dashboard_view = format_dashboard_markdown(fresh_state)
                ocean_plot = generate_ocean_plot(fresh_state.get("ocean_history", []))
                return (fresh_state["messages"], fresh_state, dashboard_view, ocean_plot, gr.update(interactive=False), gr.update(interactive=False),
                        fresh_state.get("knn_rec", ""), fresh_state.get("final_rec", ""),
                        fresh_state.get("profile_summary", ""), fresh_state.get("final_recommendations", ""), fresh_state.get("resume_content", ""))


    clear_btn.click(
        reset_state,
        inputs=[],
        outputs=[chatbot, initial_state, state_display, ocean_plot_display, confirm_profile_btn, generate_resume_btn,
                 knn_rec_display, final_rec_display,
                 profile_display, recommendations_display, resume_display]
    )

    msg.submit(
        add_user_message,
        inputs=[msg, initial_state],
        outputs=[chatbot, initial_state, msg, submit_btn, clear_btn]
    ).then(
        run_graph,
        inputs=[initial_state],
        outputs=[chatbot, initial_state, state_display, ocean_plot_display, confirm_profile_btn, generate_resume_btn,
                 knn_rec_display, final_rec_display,
                 profile_display, recommendations_display, resume_display,
                 msg, submit_btn, clear_btn]
    )

    submit_btn.click(
        add_user_message,
        inputs=[msg, initial_state],
        outputs=[chatbot, initial_state, msg]
    ).then(
        run_graph,
        inputs=[initial_state],
        outputs=[chatbot, initial_state, state_display, ocean_plot_display, confirm_profile_btn, generate_resume_btn,
                 knn_rec_display, final_rec_display,
                 profile_display, recommendations_display, resume_display,
                 msg, submit_btn, clear_btn]
    )

    confirm_profile_btn.click(
        handle_profile_confirmation,
        inputs=[initial_state],
        outputs=[chatbot, initial_state, confirm_profile_btn]
    ).then(
        run_graph,
        inputs=[initial_state],
        outputs=[chatbot, initial_state, state_display, ocean_plot_display, confirm_profile_btn, generate_resume_btn,
                 knn_rec_display, final_rec_display,
                 profile_display, recommendations_display, resume_display,
                 msg, submit_btn, clear_btn]
    )

    generate_resume_btn.click(
        handle_resume_initiation,
        inputs=[initial_state],
        outputs=[chatbot, initial_state, generate_resume_btn]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
