import sqlite3
import pandas as pd
import os

# Configuration
RAW_DATA_DIR = "./raw_onet"
DB_PATH = "./onet.db"


def create_connection(db_file):
    """Create a database connection to the SQLite database."""
    try:
        os.makedirs(os.path.dirname(db_file), exist_ok=True)
        conn = sqlite3.connect(db_file)
        print(f"✅ Successfully connected to SQLite database: {db_file}")
        return conn
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return None


def ingest_excel_table(conn, table_name, file_name):
    """Reads an O*NET Excel file and loads it into an indexed SQLite table."""
    file_path = os.path.join(RAW_DATA_DIR, file_name)

    if not os.path.exists(file_path):
        print(f"⚠️ Warning: File not found: {file_path}. Skipping.")
        return

    print(f"⏳ Ingesting {file_name} into table '{table_name}'...")

    try:
        # Read the Excel file using pandas (requires openpyxl)
        df = pd.read_excel(file_path, engine='openpyxl')

        # Clean column names for SQL (e.g., 'O*NET-SOC Code' -> 'o_net_soc_code')
        df.columns = [str(c).strip().replace(' ', '_').replace('-', '_').replace('*', '').lower() for c in df.columns]

        # Write the data_factory to SQLite
        df.to_sql(table_name, conn, if_exists='replace', index=False)

        # Identify the SOC column for indexing to ensure <10ms latency
        soc_col = next((c for c in df.columns if 'soc' in c), None)

        if soc_col:
            cursor = conn.cursor()
            # Create a high-speed index on the SOC code
            index_query = f"CREATE INDEX IF NOT EXISTS idx_{table_name}_soc ON {table_name} ({soc_col});"
            cursor.execute(index_query)
            print(f"   ↳ ✅ Success: Inserted {len(df)} rows. Indexed on '{soc_col}'.")
        else:
            print(f"   ↳ ✅ Success: Inserted {len(df)} rows. (No SOC index required).")

    except Exception as e:
        print(f"❌ Error ingesting {file_name}: {e}")


def main():
    print("Starting C-Arc Relational Database Ingestion (Excel Version)...")
    conn = create_connection(DB_PATH)

    if conn is not None:
        # 1. The Core Anchor
        ingest_excel_table(conn, "occupation_data", "Occupation Data.xlsx")

        # 2. The Technical Skeleton
        ingest_excel_table(conn, "skills", "Skills.xlsx")
        ingest_excel_table(conn, "technology_skills", "Technology Skills.xlsx")

        # 3. The Psychometric Soul
        ingest_excel_table(conn, "work_styles", "Work Styles.xlsx")

        # 4. The Interface Layer
        ingest_excel_table(conn, "work_activities", "Work Activities.xlsx")

        # 5. The Relational Output Bridge
        ingest_excel_table(conn, "task_statements", "Task Statements.xlsx")
        ingest_excel_table(conn, "tasks_to_dwas", "Tasks to DWAs.xlsx")
        ingest_excel_table(conn, "dwa_reference", "DWA Reference.xlsx")

        conn.close()
        print("\n🎉 Phase 1 Ingestion Complete. Your relational 'onet.db' is ready.")
    else:
        print("Cannot create the database connection.")


if __name__ == '__main__':
    main()
