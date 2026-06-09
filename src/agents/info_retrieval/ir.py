from workflow.state import CArcState

def execute_ir(state: CArcState, ir_extractor, ir_mapper) -> dict:
    messages = state.get("messages", [])
    last_extracted_idx = state.get("ir_last_extracted_index", 0)
    unextracted_messages = messages[last_extracted_idx:]

    # Initialize the 4-part profile structure if it doesn't exist
    current_profile = state.get("master_profile", {})
    updated_profile = {
        "tasks": list(current_profile.get("tasks", [])),
        "dwas": list(current_profile.get("dwas", [])),
        "work_activities": list(current_profile.get("work_activities", [])),
        "skills": list(current_profile.get("skills", [])),
        "tech_skills": list(current_profile.get("tech_skills", [])),
        "basic_info": dict(current_profile.get("basic_info", {"full_name": None, "email": None, "phone": None, "location": None})),
        "education": list(current_profile.get("education", []))
    }

    if not unextracted_messages:
        print(f"    -> [DEBUG IR] No new messages to extract. Skipping API call.")
        return {}

    extracted_changes = ir_extractor.extract_intents(unextracted_messages)

    if not extracted_changes:
        print(f"    -> No entities extracted on this turn. Preserving profile history.")
        print(f"\n[+] DEBUG - Master Profile Updated: {updated_profile}")
        return {"ir_last_extracted_index": len(messages)}

    for change in extracted_changes:
        intent = change.get("intent")
        entity_type = change.get("type")
        value = change.get("value")

        if not value:
            continue

        if entity_type == "basic_info" and isinstance(value, dict):
            for k, v in value.items():
                if v:  # Only update if the LLM actually found a value
                    updated_profile["basic_info"][k] = v
            continue

        elif entity_type == "education" and isinstance(value, dict):
            # Prevent appending exact duplicate education records
            if value not in updated_profile["education"]:
                updated_profile["education"].append(value)
            continue

        # Fall back to your legacy O*NET database matching for standard skills/tasks
        if entity_type not in ["experience", "skill", "task", "dwa", "wa", "tech"]:
            continue

        grounded_data_list = ir_mapper.ground_phrase(value, entity_type)

        if not grounded_data_list:
            continue

        # Iterate through every valid match the LLM approved
        for grounded_data in grounded_data_list:
            onet_code = grounded_data.get("id")
            resolved_type = grounded_data.get("resolved_type", entity_type)

            if not onet_code:
                continue

            print(f"\n[?] Mapping Attempt for '{value}':")
            print(f"    -> Extracted as: {entity_type} | Resolved to: {resolved_type}")
            print(f"    -> Resulting ID: {onet_code}")

            if resolved_type == "task":
                target_list = updated_profile["tasks"]
            elif resolved_type == "dwa":
                target_list = updated_profile["dwas"]
            elif resolved_type == "wa":
                target_list = updated_profile["work_activities"]
            elif resolved_type == "skill":
                target_list = updated_profile["skills"]
            elif resolved_type == "tech":
                target_list = updated_profile["tech_skills"]
            else:
                continue

            # Conflict Resolution
            if intent == "ADD" and onet_code not in target_list:
                target_list.append(onet_code)
            elif intent == "DELETE" and onet_code in target_list:
                target_list.remove(onet_code)

    # Perform Relational Expansion ---
    print("\n[+] Expanding relational hierarchy (Tasks -> DWAs -> WAs)...")
    expanded_data = ir_mapper.expand_relational_hierarchy(
        updated_profile["tasks"],
        updated_profile["dwas"]
    )

    # Merge the expanded data back in, using sets to guarantee no duplicates
    updated_profile["dwas"] = list(set(updated_profile["dwas"] + expanded_data["dwas"]))
    updated_profile["work_activities"] = list(set(updated_profile["work_activities"] + expanded_data["was"]))

    # Cleanly return the reconciled and expanded state
    print(f"\n[+] DEBUG - Master Profile Updated: {updated_profile}")
    return {
        "master_profile": updated_profile,
        "ir_last_extracted_index": len(messages)
    }
