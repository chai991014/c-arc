from typing import Dict, Any
from .extractor import IRExtractor
from .mapper import IRMapper
from .reconciler import ProfileReconciler


class IRAgentNode:
    def __init__(self):
        """
        Initializes the entire Information Retrieval module.
        Loading this once prevents the Mapper from recalculating vectors on every chat turn.
        """
        print("⚙️ Initializing IR Agent Node components...")
        self.extractor = IRExtractor()
        self.mapper = IRMapper()
        self.reconciler = ProfileReconciler()
        print("✅ IR Agent Node Ready.")

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        The entry point for LangGraph.
        Takes the current CArcState, processes the latest user message,
        and returns the updated master_profile to the graph.
        """
        print("\n🔍 [IR Node] Processing new state...")

        # 1. Isolate Input: Get the latest message from the state
        messages = state.get("messages", [])
        if not messages:
            return {}  # No state updates

        latest_message = messages[-1]

        # We only extract data from the Human, not the AI Mentor's questions
        if getattr(latest_message, "type", "") != "human":
            return {}

        user_text = latest_message.content

        # 2. Extract: Ask DeepSeek to find tools/skills
        extraction_packet = self.extractor.extract(user_text)
        extractions = extraction_packet.get("extractions", [])

        if not extractions:
            print("   ↳ No professional entities detected.")
            return {}

        # 3. Get the current profile from state (or initialize it safely)
        current_profile = state.get("master_profile", {
            "skills": [],
            "work_activities": [],
            "role_history": [],
            "verification_status": False
        })

        # 4. Map & Reconcile Loop
        for item in extractions:
            raw_entity = item.get("raw_entity")
            print(f"   ↳ Extracted Entity: '{raw_entity}' [Intent: {item.get('intent')}]")

            # Mathematical Grounding against O*NET
            mapped_skill = self.mapper.map_skill(raw_entity)

            if mapped_skill:
                print(f"      ↳ Grounded to O*NET: {mapped_skill['onet_name']} (ID: {mapped_skill['onet_id']})")

                # Attach the mapped data to the action packet for the Reconciler
                item["mapped_skill"] = mapped_skill

                # Traffic Control / Database Update
                current_profile = self.reconciler.process_action(current_profile, item)
            else:
                print(f"      ↳ Dropped: Could not confidently map '{raw_entity}' to O*NET.")

        # 5. Write State: Return ONLY the keys that need updating in LangGraph
        return {"master_profile": current_profile}
