import sqlite3

DB_NAME = "experiments.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            picture TEXT,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create experiments table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            rf_power_w REAL,
            working_pressure_mtorr REAL,
            ar_flow_sccm REAL,
            o2_flow_sccm REAL,
            substrate_temp_c REAL,
            target_substrate_distance_mm REAL,
            sputtering_time_min REAL,
            film_thickness_nm REAL,
            deposition_rate_nm_min REAL,
            h2_response_time_s REAL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    # AUTO-FIX: Check if user_id column exists in existing experiments table, add it if missing
    cursor.execute("PRAGMA table_info(experiments)")
    columns = [column[1] for column in cursor.fetchall()]
    if "user_id" not in columns:
        cursor.execute("ALTER TABLE experiments ADD COLUMN user_id TEXT")
    if "film_thickness_nm" not in columns:
        cursor.execute("ALTER TABLE experiments ADD COLUMN film_thickness_nm REAL")
    if "deposition_rate_nm_min" not in columns:
        cursor.execute("ALTER TABLE experiments ADD COLUMN deposition_rate_nm_min REAL")

    conn.commit()
    conn.close()

def get_or_create_user(user_info, founder_email="pavannaidu9031@gmail.com"):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    user_id = user_info.get("sub") or user_info.get("id")
    email = user_info.get("email")
    name = user_info.get("name", "Researcher")
    picture = user_info.get("picture", "")
    
    initial_status = "approved" if email.lower() == founder_email.lower() else "pending"
    
    cursor.execute("SELECT id, email, name, picture, status FROM users WHERE id = ?", (user_id,))
    existing_user = cursor.fetchone()
    
    if existing_user:
        conn.close()
        return {
            "id": existing_user[0],
            "email": existing_user[1],
            "name": existing_user[2],
            "picture": existing_user[3],
            "status": existing_user[4]
        }
    else:
        cursor.execute(
            "INSERT INTO users (id, email, name, picture, status) VALUES (?, ?, ?, ?, ?)",
            (user_id, email, name, picture, initial_status)
        )
        conn.commit()
        conn.close()
        return {
            "id": user_id,
            "email": email,
            "name": name,
            "picture": picture,
            "status": initial_status
        }

def update_user_status(user_id, new_status):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET status = ? WHERE id = ?", (new_status, user_id))
    conn.commit()
    conn.close()

def get_user_by_id(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, name, picture, status FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "email": row[1], "name": row[2], "picture": row[3], "status": row[4]}
    return None

def add_experiment(user_id, rf_power_w, working_pressure_mtorr, ar_flow_sccm, o2_flow_sccm, 
                   substrate_temp_c, target_substrate_distance_mm, sputtering_time_min, 
                   film_thickness_nm, deposition_rate_nm_min, h2_response_time_s, notes):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO experiments (
            user_id, rf_power_w, working_pressure_mtorr, ar_flow_sccm, o2_flow_sccm,
            substrate_temp_c, target_substrate_distance_mm, sputtering_time_min,
            film_thickness_nm, deposition_rate_nm_min, h2_response_time_s, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, rf_power_w, working_pressure_mtorr, ar_flow_sccm, o2_flow_sccm,
          substrate_temp_c, target_substrate_distance_mm, sputtering_time_min,
          film_thickness_nm, deposition_rate_nm_min, h2_response_time_s, notes))
    conn.commit()
    conn.close()

def get_experiments_by_user(user_id):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM experiments WHERE user_id = ? ORDER BY id DESC", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def delete_experiment(exp_id, user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM experiments WHERE id = ? AND user_id = ?", (exp_id, user_id))
    conn.commit()
    conn.close()