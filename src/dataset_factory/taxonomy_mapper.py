import json
import os
import pandas as pd
from src.agents.ir.mapper import IRMapper


class TaxonomyMapper:
    def __init__(self, skills_db_path: str, output_path: str):
        self.skills_db_path = skills_db_path
        self.output_path = output_path
        self.mapper = IRMapper()  # Reuse Phase 2 Logic
        self.mapping_table = {}

    def run_mapping(self):
        """Iterates through Trendcart categories and skills to find O*NET matches."""
        if not os.path.exists(self.skills_db_path):
            raise FileNotFoundError(f"Missing Trendcart skills DB: {self.skills_db_path}")

        with open(self.skills_db_path, 'r') as f:
            skills_db = json.load(f)

        print(f"🧠 Grounding {len(skills_db)} domains to O*NET taxonomy...")

        for domain, skills in skills_db.items():
            print(f"   ↳ Processing Domain: {domain}")

            current_threshold = 0.60 if domain == "Soft Skills" else 0.65

            for skill in skills:
                # Use Phase 2 Vector Grounding
                match = self.mapper.map_skill(skill, threshold=current_threshold)

                if match and match["onet_id"] != "Unmapped":
                    self.mapping_table[skill] = {
                        "original_skill": skill,
                        "trendcart_domain": domain,
                        "onet_name": match["onet_name"],
                        "onet_id": match["onet_id"],
                        "confidence": match["confidence"]
                    }
                else:
                    # Log the failure for manual audit later
                    print(f"      ⚠️ No high-confidence match for: '{skill}' (Threshold: {current_threshold})")
                    self.mapping_table[skill] = {
                        "original_skill": skill,
                        "trendcart_domain": domain,
                        "onet_name": "Unmapped",
                        "onet_id": "Unmapped",
                        "confidence": 0.0
                    }

        self._save_results()

    def _save_results(self):
        """Saves the lookup table for the Skeleton Builder to use."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(self.mapping_table, f, indent=4)

        # Also save as CSV for easy manual audit
        df = pd.DataFrame.from_dict(self.mapping_table, orient='index')
        df.to_csv(self.output_path.replace('.json', '.csv'), index=False)

        print(f"✅ Taxonomy Mapping Complete: saved to {self.output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    SKILLS_JSON = os.path.join(project_root, "data", "raw_resume", "trendcart", "skills_database.json")
    OUTPUT_JSON = os.path.join(project_root, "data", "interim", "skill_taxonomy_lookup.json")

    t_mapper = TaxonomyMapper(SKILLS_JSON, OUTPUT_JSON)
    t_mapper.run_mapping()
