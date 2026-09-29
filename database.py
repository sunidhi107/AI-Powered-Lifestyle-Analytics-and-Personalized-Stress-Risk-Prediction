import sqlite3
from pathlib import Path
from datetime import datetime


# ============================================================
# DATABASE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_PATH = BASE_DIR / "stressrisk.db"


# ============================================================
# GET DATABASE CONNECTION
# ============================================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# CREATE TABLES
# ============================================================

def create_tables():

    connection = get_connection()

    cursor = connection.cursor()


    # ========================================================
    # USERS TABLE
    # ========================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,

            email TEXT NOT NULL UNIQUE,

            password TEXT NOT NULL,

            gender TEXT,

            created_at TEXT NOT NULL

        )
    """)


    # ========================================================
    # ASSESSMENTS TABLE
    # ========================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assessments (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER NOT NULL,

            age REAL,

            gender TEXT,

            occupation TEXT,

            sleep_duration REAL,

            quality_of_sleep REAL,

            physical_activity REAL,

            bmi_category TEXT,

            heart_rate REAL,

            daily_steps REAL,

            systolic_bp REAL,

            diastolic_bp REAL,

            stress_prediction REAL,

            risk_level TEXT,

            assessment_date TEXT NOT NULL,

            stress_level INTEGER,

            created_at TEXT,

            health_data TEXT,

            recommendations TEXT,

            top_features TEXT,

            FOREIGN KEY (user_id)
                REFERENCES users(id)

        )
    """)

    # Ensure missing columns are added if migrating from earlier schema
    existing_columns = {
        row[1] for row in cursor.execute("PRAGMA table_info(assessments)").fetchall()
    }
    schema_additions = {
        "stress_level": "INTEGER",
        "created_at": "TEXT",
        "health_data": "TEXT",
        "recommendations": "TEXT",
        "top_features": "TEXT"
    }
    for col_name, col_type in schema_additions.items():
        if col_name not in existing_columns:
            cursor.execute(f"ALTER TABLE assessments ADD COLUMN {col_name} {col_type}")

    connection.commit()

    connection.close()


# ============================================================
# CREATE DATABASE
# ============================================================

if __name__ == "__main__":

    create_tables()

    print()
    print("=" * 55)
    print("       STRESSRISK AI DATABASE")
    print("=" * 55)

    print()

    print(
        "Database created successfully!"
    )

    print()

    print(
        "Location:"
    )

    print(
        DATABASE_PATH
    )

    print()

    print(
        "Tables created:"
    )

    print(
        "1. users"
    )

    print(
        "2. assessments"
    )

    print()

    print("=" * 55)