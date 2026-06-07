import sqlite3

def translate_onet_ids(master_profile: dict, db_path: str = "../data_factory/onet.db") -> dict:
    translated = {"tasks": [], "dwas": [], "work_activities": [], "skills": [], "tech_skills": []}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Use Sets to automatically handle duplicate parent IDs
        rolled_up_dwas = set(master_profile.get("dwas", []))
        rolled_up_was = set(master_profile.get("work_activities", []))
        unlinked_tasks = []

        # Roll-up Tasks to DWAs
        for t_id in master_profile.get("tasks", []):
            try:
                # Find the parent DWA for this specific task
                cursor.execute("SELECT dwa_id FROM tasks_to_dwas WHERE task_id = ? LIMIT 1", (t_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    rolled_up_dwas.add(row[0])  # Escalate task to DWA!
                else:
                    unlinked_tasks.append(t_id)  # No linkage, keep as granular task
            except sqlite3.Error:
                unlinked_tasks.append(t_id)

        # Roll-up DWAs to Generalized Work Activities (WAs)
        for d_id in rolled_up_dwas:
            try:
                cursor.execute("SELECT element_id FROM dwa_reference WHERE dwa_id = ? LIMIT 1", (d_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    rolled_up_was.add(row[0])  # Escalate DWA to WA!
            except sqlite3.Error:
                continue

        # Translate unlinked granular Tasks
        for t_id in unlinked_tasks:
            cursor.execute("SELECT task FROM task_statements WHERE task_id = ? LIMIT 1", (t_id,))
            row = cursor.fetchone()
            if row and row[0]:
                translated["tasks"].append(row[0])

        # Translate DWAs
        for d_id in rolled_up_dwas:
            cursor.execute("SELECT dwa_title FROM dwa_reference WHERE dwa_id = ? LIMIT 1", (d_id,))
            row = cursor.fetchone()
            if row and row[0]:
                translated["dwas"].append(row[0])

        # Translate Work Activities (WAs)
        for wa_id in rolled_up_was:
            cursor.execute("SELECT element_name FROM work_activities WHERE element_id = ? LIMIT 1", (wa_id,))
            row = cursor.fetchone()
            if row and row[0] and row[0] not in translated["work_activities"]:
                translated["work_activities"].append(row[0])

        # Translate Core Skills
        for s_id in master_profile.get("skills", []):
            cursor.execute("SELECT element_name FROM skills WHERE element_id = ? LIMIT 1", (s_id,))
            row = cursor.fetchone()
            if row and row[0] and row[0] not in translated["skills"]:
                translated["skills"].append(row[0])

        # Translate Tech Skills
        # (Since tech pool IDs are built as 'TECH-[Name]', we can just string-replace)
        for tech_id in master_profile.get("tech_skills", []):
            tech_name = str(tech_id).replace("TECH-", "")
            if tech_name not in translated["tech_skills"]:
                translated["tech_skills"].append(tech_name)

        conn.close()

    except sqlite3.Error as e:
        print(f"[!] Database Error in translate_onet_ids: {e}")

    return translated