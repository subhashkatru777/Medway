"""
MedWay – seed_data.py
=====================
Loads all 5 CSV files from the data/ folder into the SQLite database.
Run ONCE after creating the database:

    python seed_data.py

Safe to re-run — uses INSERT OR IGNORE so no duplicates.
"""

import os
import csv
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, 'medway.db')
DATA_DIR = os.path.join(BASE_DIR, 'data')


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def create_tables(conn):
    """Create medical data tables (separate from user tables in app.py)."""
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ml_departments (
            dept_id         TEXT PRIMARY KEY,
            department_name TEXT NOT NULL
        )""")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ml_diseases (
            disease_id   TEXT PRIMARY KEY,
            disease_name TEXT NOT NULL,
            dept_id      TEXT NOT NULL,
            FOREIGN KEY (dept_id) REFERENCES ml_departments(dept_id)
        )""")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ml_symptoms (
            symptom_id   TEXT PRIMARY KEY,
            symptom_name TEXT NOT NULL
        )""")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ml_symptom_disease (
            disease_id  TEXT NOT NULL,
            symptom_id  TEXT NOT NULL,
            PRIMARY KEY (disease_id, symptom_id),
            FOREIGN KEY (disease_id) REFERENCES ml_diseases(disease_id),
            FOREIGN KEY (symptom_id) REFERENCES ml_symptoms(symptom_id)
        )""")

    conn.commit()
    print("[seed] Tables created / verified.")


def load_csv(filepath):
    rows = []
    with open(filepath, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v.strip() for k, v in row.items()})
    return rows


def seed_departments(conn):
    rows = load_csv(os.path.join(DATA_DIR, 'departments_master.csv'))
    cur  = conn.cursor()
    count = 0
    for r in rows:
        cur.execute(
            "INSERT OR IGNORE INTO ml_departments (dept_id, department_name) VALUES (?, ?)",
            (r['dept_id'], r['department_name'])
        )
        count += cur.rowcount
    conn.commit()
    print(f"[seed] Departments: {count} inserted ({len(rows)} total).")


def seed_diseases(conn):
    rows = load_csv(os.path.join(DATA_DIR, 'diseases_master.csv'))
    cur  = conn.cursor()
    count = 0
    for r in rows:
        cur.execute(
            "INSERT OR IGNORE INTO ml_diseases (disease_id, disease_name, dept_id) VALUES (?, ?, ?)",
            (r['disease_id'], r['disease_name'], r['dept_id'])
        )
        count += cur.rowcount
    conn.commit()
    print(f"[seed] Diseases: {count} inserted ({len(rows)} total).")


def seed_symptoms(conn):
    rows = load_csv(os.path.join(DATA_DIR, 'symptoms_master.csv'))
    cur  = conn.cursor()
    count = 0
    for r in rows:
        cur.execute(
            "INSERT OR IGNORE INTO ml_symptoms (symptom_id, symptom_name) VALUES (?, ?)",
            (r['symptom_id'], r['symptom_name'])
        )
        count += cur.rowcount
    conn.commit()
    print(f"[seed] Symptoms: {count} inserted ({len(rows)} total).")


def seed_symptom_disease_map(conn):
    rows = load_csv(os.path.join(DATA_DIR, 'symptom_disease_map.csv'))
    cur  = conn.cursor()
    count = 0
    for r in rows:
        cur.execute(
            "INSERT OR IGNORE INTO ml_symptom_disease (disease_id, symptom_id) VALUES (?, ?)",
            (r['disease_id'], r['symptom_id'])
        )
        count += cur.rowcount
    conn.commit()
    print(f"[seed] Symptom-Disease mappings: {count} inserted ({len(rows)} total).")


def verify(conn):
    cur = conn.cursor()
    print("\n[seed] Verification:")
    for table in ['ml_departments', 'ml_diseases', 'ml_symptoms', 'ml_symptom_disease']:
        n = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:30s} → {n} rows")


if __name__ == '__main__':
    import sys
    force = '--force' in sys.argv or '-f' in sys.argv

    print(f"[seed] Database: {DB_PATH}")
    conn = get_db()
    create_tables(conn)

    if force:
        print("[seed] --force flag: clearing existing ML data before re-seeding...")
        cur = conn.cursor()
        cur.execute("DELETE FROM ml_symptom_disease")
        cur.execute("DELETE FROM ml_symptoms")
        cur.execute("DELETE FROM ml_diseases")
        cur.execute("DELETE FROM ml_departments")
        conn.commit()
        print("[seed] Cleared.")

    seed_departments(conn)
    seed_diseases(conn)
    seed_symptoms(conn)
    seed_symptom_disease_map(conn)
    verify(conn)
    conn.close()
    print("\n[seed] Done! Database seeded successfully.")