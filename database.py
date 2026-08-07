import sqlite3
import os

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
            target_substrate_distance_cm REAL,
            sputtering_time_min REAL,
            film_thickness_nm REAL,
            rotation_speed_rpm REAL DEFAULT 5.0,
            substrate_type TEXT DEFAULT 'Si Wafer',
            xrd_phase TEXT DEFAULT 'Amorphous',
            grain_size_nm REAL,
            h2_response_time_s REAL,
            wavelength_shift_pm REAL,
            quality_score REAL DEFAULT 50.0,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    # Create suggestion_history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS suggestion_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            user_email TEXT,
            suggested_rf_power REAL,
            suggested_pressure REAL,
            suggested_distance REAL,
            suggested_thickness REAL,
            suggested_rotation REAL,
            suggested_ar_flow REAL,
            predicted_xrd_score REAL,
            predicted_wavelength REAL,
            confidence_score INTEGER,
            converged BOOLEAN DEFAULT 0,
            kappa_used REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # PRAGMA Auto-migrations
    cursor.execute("PRAGMA table_info(experiments)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if "quality_score" not in columns:
        cursor.execute("ALTER TABLE experiments ADD COLUMN quality_score REAL DEFAULT 50.0")
    if "target_substrate_distance_cm" not in columns and "target_substrate_distance_mm" in columns:
        cursor.execute("ALTER TABLE experiments RENAME COLUMN target_substrate_distance_mm TO target_substrate_distance_cm")
    if "target_substrate_distance_cm" not in columns:
        cursor.execute("ALTER TABLE experiments ADD COLUMN target_substrate_distance_cm REAL DEFAULT 7.0")

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
                   substrate_temp_c, target_substrate_distance_cm, sputtering_time_min, 
                   film_thickness_nm, rotation_speed_rpm, substrate_type, xrd_phase,
                   grain_size_nm, h2_response_time_s, wavelength_shift_pm, quality_score, notes):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO experiments (
            user_id, rf_power_w, working_pressure_mtorr, ar_flow_sccm, o2_flow_sccm,
            substrate_temp_c, target_substrate_distance_cm, sputtering_time_min,
            film_thickness_nm, rotation_speed_rpm, substrate_type, xrd_phase,
            grain_size_nm, h2_response_time_s, wavelength_shift_pm, quality_score, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, rf_power_w, working_pressure_mtorr, ar_flow_sccm, o2_flow_sccm,
          substrate_temp_c, target_substrate_distance_cm, sputtering_time_min,
          film_thickness_nm, rotation_speed_rpm, substrate_type, xrd_phase,
          grain_size_nm, h2_response_time_s, wavelength_shift_pm, quality_score, notes))
    conn.commit()
    conn.close()

def get_experiments_by_user(user_id):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Sorted by quality_score DESC by default
    cursor.execute("SELECT * FROM experiments WHERE user_id = ? ORDER BY quality_score DESC, id DESC", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def delete_experiment(exp_id, user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM experiments WHERE id = ? AND user_id = ?", (exp_id, user_id))
    conn.commit()
    conn.close()

def save_suggestion_history(user_id, user_email, suggestion):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    s = suggestion["suggested"]
    cursor.execute("""
        INSERT INTO suggestion_history (
            user_id, user_email, suggested_rf_power, suggested_pressure,
            suggested_distance, suggested_thickness, suggested_rotation,
            suggested_ar_flow, predicted_xrd_score, predicted_wavelength,
            confidence_score, converged, kappa_used
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, user_email, s["rf_power"], s["working_pressure"],
        s["target_distance"], s["film_thickness"], s["rotation_speed"],
        s["ar_flow"], suggestion["expected"]["xrd_score"],
        suggestion["expected"]["wavelength_shift_estimate"],
        suggestion["confidence"]["score"],
        1 if suggestion["convergence"]["converged"] else 0,
        suggestion["kappa_used"]
    ))
    conn.commit()
    conn.close()

def get_recent_suggestions(user_id, limit=3):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM suggestion_history WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
