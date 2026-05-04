class ProfileReconciler:
    def __init__(self):
        """
        Initializes the Reconciler.
        This acts as the Traffic Controller to maintain database integrity.
        """
        pass

    def process_action(self, current_profile: dict, action_packet: dict) -> dict:
        """
        Takes the current LangGraph master_profile and an action packet
        (containing Extractor intent + Mapper O*NET ID) and safely updates the profile.
        """
        intent = action_packet.get("intent")
        behavior_intent = action_packet.get("behavior_intent")
        mapped_skill = action_packet.get("mapped_skill")  # The dictionary returned by the Mapper

        # If the Mapper rejected the entity (below threshold), we safely ignore it
        if not mapped_skill:
            return current_profile

        onet_id = mapped_skill.get("onet_id")
        onet_name = mapped_skill.get("onet_name")

        # Ensure the skills list exists in the profile
        if "skills" not in current_profile:
            current_profile["skills"] = []

        skills_list = current_profile["skills"]

        # Identity Reconciliation: Check if the O*NET ID already exists
        existing_skill_idx = next(
            (i for i, s in enumerate(skills_list) if s["id"] == onet_id),
            None
        )

        if intent == "ADD":
            if existing_skill_idx is not None:
                # CONFLICT: Skill exists. Convert to UPDATE silently.
                self._append_context(skills_list[existing_skill_idx], behavior_intent)
            else:
                # Clean ADD
                skills_list.append(self._create_skill_node(onet_id, onet_name, behavior_intent))

        elif intent == "UPDATE":
            if existing_skill_idx is not None:
                # Normal UPDATE: Append new context
                self._append_context(skills_list[existing_skill_idx], behavior_intent)
            else:
                # CONFLICT: Skill doesn't exist but user is updating context.
                # Convert to ADD to preserve the historical fact.
                skills_list.append(self._create_skill_node(onet_id, onet_name, behavior_intent))

        elif intent == "DELETE":
            if existing_skill_idx is not None:
                # Clean DELETE
                del skills_list[existing_skill_idx]
            # If DELETE but skill doesn't exist, do nothing (safe ignore).

        # Update the state and return
        current_profile["skills"] = skills_list
        return current_profile

    def _create_skill_node(self, onet_id: str, onet_name: str, behavior_intent: str) -> dict:
        """Helper to standardize the skill dictionary structure."""
        return {
            "id": onet_id,
            "name": onet_name,
            "contexts": [behavior_intent] if behavior_intent else []
        }

    def _append_context(self, skill_node: dict, behavior_intent: str):
        """Helper to add new work activities to an existing skill without duplicating them."""
        if "contexts" not in skill_node:
            skill_node["contexts"] = []

        if behavior_intent and behavior_intent not in skill_node["contexts"]:
            skill_node["contexts"].append(behavior_intent)
