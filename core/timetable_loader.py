import json

def load_timetable_from_json(db, json_path = "config/timetable.json"):
    insert_query = """
    INSERT INTO timetable_slots
    (day_of_week, start_time, end_time, course_code, batch, section)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (day_of_week, start_time, end_time)
    DO UPDATE SET
        batch = EXCLUDED.batch,
        section = EXCLUDED.section,
        lab_name = EXCLUDED.lab_name,
        updated_at = NOW();
    """
    try:
        with open(json_path, "r") as f:
            timetable_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return False
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_path}")
        return False

    cur = db.conn.cursor()

    for slot in timetable_data:
        cur.execute(
            insert_query,
            (
                slot["day_of_week"],
                slot["start_time"],
                slot["end_time"],
                slot["course_code"],
                slot.get("batch"),
                slot.get("section"),
            )
        )

    
    db.conn.commit()
    cur.close()
