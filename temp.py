import json
from src.agents.ir.node import IRAgentNode


def run_test():
    # 1. Initialize the Node (This loads the LLM and the Vector Cache)
    print("🚀 Initializing IR Agent for Testing...")
    ir_node = IRAgentNode()

    # 2. Define a Mock Initial State
    state = {
        "messages": [],
        "master_profile": {
            "skills": [],
            "verification_status": False
        }
    }

    # 3. Test Case 1: The "ADD" Intent
    print("\n--- Test 1: Skill Extraction (ADD) ---")
    state["messages"] = [type('msg', (object,),
                              {"content": "I am proficient in Python and have used SQL for database management.",
                               "type": "human"})]

    result = ir_node(state)
    state["master_profile"] = result.get("master_profile", state["master_profile"])

    print(f"Profile Skills: {[s['name'] for s in state['master_profile']['skills']]}")

    # 4. Test Case 2: The "UPDATE" Intent (Contextual Mapping)
    print("\n--- Test 2: Context Update (UPDATE) ---")
    state["messages"].append(type('msg', (object,),
                                  {"content": "Actually, I mainly use Python for building Machine Learning models.",
                                   "type": "human"}))

    result = ir_node(state)
    state["master_profile"] = result.get("master_profile", state["master_profile"])

    # Check if 'Building Machine Learning models' was added to Python's contexts
    python_skill = next(s for s in state['master_profile']['skills'] if "Python" in s['name'])
    print(f"Python Contexts: {python_skill['contexts']}")

    # 5. Test Case 3: The "DELETE" Intent
    print("\n--- Test 3: Skill Removal (DELETE) ---")
    state["messages"].append(type('msg', (object,),
                                  {"content": "I want to remove SQL from my profile, I haven't used it in years.",
                                   "type": "human"}))

    result = ir_node(state)
    state["master_profile"] = result.get("master_profile", state["master_profile"])

    print(f"Final Profile Skills: {[s['name'] for s in state['master_profile']['skills']]}")


if __name__ == "__main__":
    run_test()