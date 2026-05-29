from workflow.state import CArcState

def execute_ir(state: CArcState, ir_extractor, ir_mapper) -> dict:
    messages = state.get("messages", [])

    # Initialize the 4-part profile structure if it doesn't exist
    current_profile = state.get("master_profile", {})
    updated_profile = {
        "tasks": list(current_profile.get("tasks", [])),
        "dwas": list(current_profile.get("dwas", [])),
        "skills": list(current_profile.get("skills", [])),
        "tech_skills": list(current_profile.get("tech_skills", []))
    }

    if not messages:
        return {}

    extracted_changes = ir_extractor.extract_intents(messages)

    if not extracted_changes:
        print(f"    -> No entities extracted on this turn. Preserving profile history.")
        print(f"\n[+] DEBUG - Master Profile Updated: {updated_profile}")
        return {}

    for change in extracted_changes:
        intent = change.get("intent")
        entity_type = change.get("type")  # "skill", "task", or "dwa"
        value = change.get("value")

        if not value or entity_type not in ["skill", "task", "dwa"]:
            continue

        grounded_data = ir_mapper.ground_phrase(value, entity_type)
        onet_code = grounded_data.get("id")

        if not onet_code:
            continue

        print(f"\n[?] Mapping Attempt for '{value}' ({entity_type}):")
        print(f"    -> Score: {grounded_data.get('score', 'N/A')}")
        print(f"    -> Resulting ID: {onet_code}")

        # Route the ID to the correct array
        if entity_type == "task":
            target_list = updated_profile["tasks"]
        elif entity_type == "dwa":
            target_list = updated_profile["dwas"]
        elif entity_type == "skill":
            if str(onet_code).startswith("TECH-"):
                target_list = updated_profile["tech_skills"]
            else:
                target_list = updated_profile["skills"]

        # Conflict Resolution
        if intent == "ADD" and onet_code not in target_list:
            target_list.append(onet_code)
        elif intent == "DELETE" and onet_code in target_list:
            target_list.remove(onet_code)

    # Cleanly return the reconciled state
    print(f"\n[+] DEBUG - Master Profile Updated: {updated_profile}")
    return {
        "master_profile": updated_profile
    }
