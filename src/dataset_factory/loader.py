import pandas as pd
import re
import os
from bs4 import BeautifulSoup


class DatasetLoader:
    def __init__(self, raw_path: str):
        self.raw_path = raw_path
        self.df = None

    def load(self):
        """Loads the training_data.csv from the raw directory."""
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"Missing source file at {self.raw_path}")

        print(f"📂 Loading Trendcart dataset: {self.raw_path}...")
        self.df = pd.read_csv(self.raw_path)
        return self

    def clean(self):
        """Standardizes columns and scrubs the Resume_str content."""
        print("🧹 Cleaning 10,000 resume records...")

        # List of possible names for the resume column and category column
        resume_variants = ['Resume Text', 'Resume_str', 'Resume_text', 'resume_text', 'text', 'Resume']
        category_variants = ['Category', 'Job Role', 'category', 'label', 'target_domain']

        # Find which column exists in the dataframe
        found_resume_col = next((col for col in resume_variants if col in self.df.columns), None)
        found_category_col = next((col for col in category_variants if col in self.df.columns), None)

        if not found_resume_col:
            print(f"❌ Error: Could not find resume text column. Available: {list(self.df.columns)}")
            raise KeyError("Resume text column missing.")

        # 1. Perform the rename dynamically
        rename_map = {found_resume_col: 'raw_text'}
        if found_category_col:
            rename_map[found_category_col] = 'target_domain'

        self.df = self.df.rename(columns=rename_map)

        # 2. Apply text cleaning
        print(f"   ↳ Using column '{found_resume_col}' as raw source.")
        self.df['cleaned_text'] = self.df['raw_text'].apply(self._scrub_text)

        # 3. Handle additional metadata (Experience/Education) if they exist
        if 'Experience Years' in self.df.columns:
            self.df = self.df.rename(columns={'Experience Years': 'years_exp'})

        return self

    def _scrub_text(self, text: str) -> str:
        """Internal helper to remove noise from resume strings."""
        if not isinstance(text, str):
            return ""

        # Remove HTML tags (common in scraped Kaggle sets)
        text = BeautifulSoup(text, "html.parser").get_text()

        # Remove non-ASCII characters and excessive whitespace
        text = re.sub(r'[^\x00-\x7f]', r' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def save_interim(self, output_path: str):
        """Saves the cleaned data to the interim folder for Step 2."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"✅ Step 1 Complete: Cleaned data saved to {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    RAW_CSV = os.path.join(project_root, "data", "raw_resume", "trendcart", "training_data.csv")
    INTERIM_CSV = os.path.join(project_root, "data", "interim", "cleaned_training_data.csv")

    loader = DatasetLoader(RAW_CSV)
    loader.load().clean().save_interim(INTERIM_CSV)
