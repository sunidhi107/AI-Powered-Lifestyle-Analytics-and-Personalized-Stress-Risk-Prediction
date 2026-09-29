import sqlite3
from pathlib import Path


# ============================================================
# EXACT PROJECT DATABASE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_PATH = BASE_DIR / "stressrisk.db"


print()
print("=" * 60)
print("STRESSRISK AI - DATABASE FIX")
print("=" * 60)

print()
print("Database being modified:")
print(DATABASE_PATH)

print()
print("Database exists:")
print(DATABASE_PATH.exists())


# ============================================================
# CONNECT
# ============================================================

connection = sqlite3.connect(
    DATABASE_PATH
)

cursor = connection.cursor()


# ============================================================
# SHOW CURRENT ASSESSMENT TABLE
# ============================================================

cursor.execute(
    "PRAGMA table_info(assessments)"
)

columns = cursor.fetchall()


print()
print("CURRENT ASSESSMENTS COLUMNS:")
print("-" * 60)

for column in columns:

    print(
        column[1],
        "|",
        column[2]
    )


existing_columns = {
    column[1]
    for column in columns
}


# ============================================================
# ADD MISSING stress_level
# ============================================================

if "stress_level" not in existing_columns:

    print()
    print("Adding stress_level...")

    cursor.execute(
        """
        ALTER TABLE assessments
        ADD COLUMN stress_level INTEGER
        """
    )

    print("stress_level ADDED successfully.")

else:

    print()
    print("stress_level ALREADY EXISTS.")


# ============================================================
# ADD OTHER POSSIBLY MISSING COLUMNS
# ============================================================

columns_to_add = {

    "risk_level":
        "TEXT",

    "created_at":
        "TEXT",

    "health_data":
        "TEXT",

    "recommendations":
        "TEXT",

    "top_features":
        "TEXT"
}


# Refresh columns

cursor.execute(
    "PRAGMA table_info(assessments)"
)

existing_columns = {
    column[1]
    for column in cursor.fetchall()
}


for column_name, column_type in columns_to_add.items():

    if column_name not in existing_columns:

        print(
            f"Adding {column_name}..."
        )

        cursor.execute(
            f"""
            ALTER TABLE assessments
            ADD COLUMN {column_name} {column_type}
            """
        )


# ============================================================
# SAVE
# ============================================================

connection.commit()


# ============================================================
# VERIFY
# ============================================================

cursor.execute(
    "PRAGMA table_info(assessments)"
)

final_columns = cursor.fetchall()


print()
print("=" * 60)
print("FINAL ASSESSMENTS COLUMNS")
print("=" * 60)

for column in final_columns:

    print(
        column[1],
        "|",
        column[2]
    )


final_column_names = {
    column[1]
    for column in final_columns
}


print()
print("=" * 60)

if "stress_level" in final_column_names:

    print("SUCCESS: stress_level EXISTS")

else:

    print("ERROR: stress_level STILL MISSING")


print("=" * 60)


connection.close()