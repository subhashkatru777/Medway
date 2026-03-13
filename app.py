"""
MedWay – app.py
Single-file Flask application.
  SQLite (medway.db auto-created) + bcrypt password hashing.
  Patient / Doctor / Admin portals fully wired.

Run:
    pip install flask bcrypt
    python app.py
"""

import os
import re
import json
import sqlite3
from datetime import datetime, timedelta

import bcrypt
from flask import (Flask, flash, jsonify, make_response, redirect, render_template,
                   request, session, url_for)

# ML prediction engine (loaded lazily on first use)
try:
    from ml.predict import match_symptoms, predict as ml_predict
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

# Global symptom name list — populated at startup after auto_seed()
SYMPTOMS_CACHE = []

# ══════════════════════════════════════════════════════════════
# FLASK SETUP
# ══════════════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = 'medway-secret-key-change-in-production'

# Session expires the moment browser is closed (no persistent cookie)
app.config['SESSION_PERMANENT']           = False
app.config['PERMANENT_SESSION_LIFETIME']  = timedelta(hours=8)
app.config['SESSION_COOKIE_HTTPONLY']     = True
app.config['SESSION_COOKIE_SAMESITE']     = 'Lax'
# Do NOT set SESSION_COOKIE_SECURE = True for localhost (http)


@app.before_request
def force_non_permanent_session():
    """Ensure every session is non-permanent so it dies on browser close."""
    session.permanent = False


# ══════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, 'medway.db')


def get_db():
    """
    Return a per-request SQLite connection stored on Flask's g object.
    One connection is reused for the entire request, then closed automatically
    by close_db() registered below. This prevents "database is locked" errors
    caused by multiple open connections.
    """
    from flask import g
    if 'db' not in g:
        conn = sqlite3.connect(
            DB_PATH,
            timeout=30,           # wait up to 30s if another write is in progress
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")  # 30s busy timeout at SQL level
        g.db = conn
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    """Close the DB connection at the end of every request — even if it crashed."""
    from flask import g
    db = g.pop('db', None)
    if db is not None:
        db.close()


# ── bcrypt helpers ────────────────────────────────────────────

def hash_password(plain):
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()


def check_password(plain, hashed):
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ── Schema + seed ─────────────────────────────────────────────

def _raw_conn():
    """Direct connection used only at startup (outside request context)."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def init_db():
    conn = _raw_conn()   # startup — no Flask g available yet
    c    = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            email      TEXT PRIMARY KEY COLLATE NOCASE,
            full_name  TEXT NOT NULL,
            phone      TEXT NOT NULL,
            dob        TEXT NOT NULL,
            gender     TEXT NOT NULL,
            password   TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        )""")

    c.execute("""
        CREATE TABLE IF NOT EXISTS doctors (
            doctor_id  TEXT PRIMARY KEY COLLATE NOCASE,
            name       TEXT NOT NULL,
            department TEXT NOT NULL,
            email      TEXT NOT NULL,
            phone      TEXT NOT NULL,
            password   TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        )""")

    c.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            admin_id   TEXT PRIMARY KEY COLLATE NOCASE,
            name       TEXT NOT NULL,
            email      TEXT NOT NULL,
            password   TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        )""")

    if c.execute("SELECT COUNT(*) FROM admins").fetchone()[0] == 0:
        c.execute(
            "INSERT INTO admins (admin_id,name,email,password) VALUES (?,?,?,?)",
            ('admin001', 'Surya Prakash',
             'surya.admin@medway.in', hash_password('admin123'))
        )
        print("[MedWay] Default admin created  ->  ID: admin001  |  Password: admin123")

    # ── Appointments table (Phase 2) ──────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_email  TEXT NOT NULL,
            doctor_id      TEXT NOT NULL,
            disease        TEXT NOT NULL,
            department     TEXT NOT NULL,
            dept_id        TEXT NOT NULL,
            symptoms_json  TEXT NOT NULL,
            status         TEXT NOT NULL DEFAULT 'pending',
            source         TEXT NOT NULL DEFAULT 'online',
            requested_at   TEXT NOT NULL DEFAULT (datetime('now','localtime')),
            notes          TEXT
        )""")

    # Migration: add source column if upgrading from older DB
    try:
        c.execute("ALTER TABLE appointments ADD COLUMN source TEXT NOT NULL DEFAULT 'online'")
        conn.commit()
    except Exception:
        pass  # Column already exists

    # ── Queue table (Phase 3) ─────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS queue (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            appointment_id    INTEGER NOT NULL,
            doctor_id         TEXT NOT NULL,
            position          INTEGER NOT NULL,
            slot_minutes      INTEGER NOT NULL DEFAULT 15,
            status            TEXT NOT NULL DEFAULT 'waiting',
            added_at          TEXT NOT NULL DEFAULT (datetime('now','localtime')),
            started_at        TEXT,
            completed_at      TEXT,
            FOREIGN KEY (appointment_id) REFERENCES appointments(id)
        )""")

    # ── Doctor settings table (Phase 3) ───────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS doctor_settings (
            doctor_id     TEXT PRIMARY KEY,
            slot_minutes  INTEGER NOT NULL DEFAULT 15
        )""")

    # ── Server-side session tokens (Phase 5) ─────────────────
    # Stores valid session tokens. On server restart or explicit logout,
    # tokens are wiped — forcing re-login no matter what the cookie says.
    c.execute("""
        CREATE TABLE IF NOT EXISTS active_sessions (
            token       TEXT PRIMARY KEY,
            role        TEXT NOT NULL,
            user_id     TEXT NOT NULL,
            created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        )""")

    conn.commit()
    conn.close()


# ── Appointment CRUD ──────────────────────────────────────────

def create_appointment(patient_email, doctor_id, disease, department,
                       dept_id, symptoms_json, source='online'):
    conn = get_db()
    conn.execute("""
        INSERT INTO appointments
            (patient_email, doctor_id, disease, department, dept_id,
             symptoms_json, status, source, requested_at)
        VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, datetime('now','localtime'))
    """, (patient_email, doctor_id, disease, department, dept_id,
          symptoms_json, source))
    appt_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    return appt_id


def create_offline_appointment(doctor_id, patient_name, disease,
                                department, dept_id, symptoms_text):
    """Doctor adds a walk-in patient directly to queue (no patient account needed)."""
    conn = get_db()
    # Use a placeholder email for offline patients
    placeholder_email = f'offline_{doctor_id}_{datetime.now().strftime("%Y%m%d%H%M%S")}@medway.local'
    symptoms_json = json.dumps([s.strip() for s in symptoms_text.split(',') if s.strip()])

    conn.execute("""
        INSERT INTO appointments
            (patient_email, doctor_id, disease, department, dept_id,
             symptoms_json, status, source, requested_at, notes)
        VALUES (?, ?, ?, ?, ?, ?, 'accepted', 'offline', datetime('now','localtime'), ?)
    """, (placeholder_email, doctor_id, disease, department, dept_id,
          symptoms_json, f'Walk-in: {patient_name}'))
    appt_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    return appt_id, placeholder_email


def get_appointments_for_patient(patient_email):
    conn = get_db()
    rows = conn.execute("""
        SELECT a.*, d.name as doctor_name
        FROM appointments a
        LEFT JOIN doctors d ON a.doctor_id = d.doctor_id
        WHERE a.patient_email = ?
        ORDER BY a.requested_at DESC
    """, (patient_email,)).fetchall()
    result = []
    for r in rows:
        appt = dict(r)
        syms = json.loads(appt.get('symptoms_json') or '[]')
        appt['symptoms_display'] = ', '.join(syms) if syms else '—'
        result.append(appt)
    return result


def get_appointment_by_id(appt_id):
    conn = get_db()
    row = conn.execute("""
        SELECT a.*, d.name as doctor_name
        FROM appointments a
        LEFT JOIN doctors d ON a.doctor_id = d.doctor_id
        WHERE a.id = ?
    """, (appt_id,)).fetchone()
    return dict(row) if row else None


# ── Doctor appointment queries ────────────────────────────────

def get_appointments_for_doctor(doctor_id):
    """Return all appointments for a doctor grouped by status."""
    conn = get_db()
    rows = conn.execute("""
        SELECT a.*, p.full_name as patient_name
        FROM appointments a
        LEFT JOIN patients p ON a.patient_email = p.email
        WHERE a.doctor_id = ?
        ORDER BY a.requested_at DESC
    """, (doctor_id,)).fetchall()
    result = []
    for r in rows:
        appt = dict(r)
        syms = json.loads(appt.get('symptoms_json') or '[]')
        appt['symptoms_display'] = ', '.join(syms[:4]) if syms else '—'
        # Resolve walk-in patient name from notes if no patient account
        if not appt.get('patient_name'):
            notes = appt.get('notes') or ''
            if notes.startswith('Walk-in:'):
                appt['patient_name'] = notes[len('Walk-in:'):].strip() + ' (Walk-in)'
            else:
                appt['patient_name'] = 'Walk-in Patient'
        result.append(appt)
    return result


def update_appointment_status(appt_id, status, notes=None):
    conn = get_db()
    if notes is not None:
        conn.execute("UPDATE appointments SET status=?, notes=? WHERE id=?",
                     (status, notes, appt_id))
    else:
        conn.execute("UPDATE appointments SET status=? WHERE id=?",
                     (status, appt_id))
    conn.commit()


# ── Queue CRUD ────────────────────────────────────────────────

def _fmt_secs(s):
    """Format seconds into human-readable string."""
    s = max(0, int(round(s)))
    if s == 0: return 'Soon'
    h, rem = divmod(s, 3600)
    m, sc  = divmod(rem, 60)
    if h:  return f"{h}h {m}m" if m else f"{h}h"
    if m:  return f"{m} min"
    return f"{sc}s"


def get_queue_for_doctor(doctor_id):
    """
    Return active queue entries with real-time wait calculations.

    Timing model:
    - Position 1 (in-progress or first waiting): timer runs from started_at
      if in-progress, OR from added_at if waiting and it is position 1.
    - Every waiting patient's wait = sum of remaining seconds of all entries ahead.
    - remaining_seconds stored for JS live countdown.
    """
    conn = get_db()
    rows = conn.execute("""
        SELECT q.*, a.disease, a.symptoms_json, a.patient_email,
               a.notes, a.source,
               p.full_name as patient_name
        FROM queue q
        JOIN appointments a ON q.appointment_id = a.id
        LEFT JOIN patients p ON a.patient_email = p.email
        WHERE q.doctor_id = ? AND q.status IN ('waiting','in-progress')
        ORDER BY q.position ASC
    """, (doctor_id,)).fetchall()

    now = datetime.now()

    def elapsed_secs_from(ts_str):
        if not ts_str: return 0
        try:
            return max(0, (now - datetime.fromisoformat(ts_str)).total_seconds())
        except Exception:
            return 0

    def resolve_name(entry):
        name = entry.get('patient_name')
        if name:
            return name
        notes = entry.get('notes') or ''
        if notes.startswith('Walk-in:'):
            return notes[len('Walk-in:'):].strip() + ' (Walk-in)'
        return 'Walk-in Patient'

    result       = []
    cumul_secs   = 0   # cumulative wait seconds for next waiting patient

    for r in rows:
        entry = dict(r)
        syms  = json.loads(entry.get('symptoms_json') or '[]')
        entry['symptoms_display'] = ', '.join(syms[:4]) if syms else '—'
        entry['patient_name']     = resolve_name(entry)   # fix offline walk-in names
        slot_secs = (entry.get('slot_minutes') or 15) * 60

        if entry['status'] == 'in-progress':
            elapsed   = elapsed_secs_from(entry.get('started_at'))
            remaining = max(0, slot_secs - elapsed)
            entry['remaining_seconds'] = int(remaining)
            entry['remaining_label']   = _fmt_secs(remaining) + ' left'
            entry['wait_label']        = 'In Consultation'
            entry['wait_seconds']      = 0
            cumul_secs = remaining   # waiting patients start counting after this

        else:
            # waiting: their turn starts after cumul_secs
            entry['remaining_seconds'] = None
            entry['remaining_label']   = None
            entry['wait_seconds']      = int(cumul_secs)
            entry['wait_label']        = _fmt_secs(cumul_secs) if cumul_secs > 0 else 'Next up'
            cumul_secs += slot_secs   # add this patient's slot for the next

        result.append(entry)
    return result


def add_to_queue(appointment_id, doctor_id, slot_minutes=15):
    """Add accepted appointment to the queue at the next position."""
    conn = get_db()
    max_pos = conn.execute(
        "SELECT COALESCE(MAX(position),0) FROM queue WHERE doctor_id=? AND status IN ('waiting','in-progress')",
        (doctor_id,)
    ).fetchone()[0]
    conn.execute("""
        INSERT INTO queue (appointment_id, doctor_id, position, slot_minutes, status, added_at)
        VALUES (?, ?, ?, ?, 'waiting', datetime('now','localtime'))
    """, (appointment_id, doctor_id, max_pos + 1, slot_minutes))
    conn.commit()


def start_queue_entry(queue_id):
    """Mark a queue entry as in-progress."""
    conn = get_db()
    conn.execute("""
        UPDATE queue SET status='in-progress', started_at=datetime('now','localtime')
        WHERE id=?
    """, (queue_id,))
    conn.commit()


def complete_queue_entry(queue_id):
    """Mark queue entry done and update appointment to completed."""
    conn = get_db()
    row = conn.execute("SELECT appointment_id FROM queue WHERE id=?",
                       (queue_id,)).fetchone()
    if row:
        conn.execute("""
            UPDATE queue SET status='done', completed_at=datetime('now','localtime')
            WHERE id=?
        """, (queue_id,))
        conn.execute("UPDATE appointments SET status='completed' WHERE id=?",
                     (row['appointment_id'],))
        # Re-number remaining queue
        conn.execute("""
            UPDATE queue SET position = position - 1
            WHERE doctor_id = (SELECT doctor_id FROM queue WHERE id=?)
            AND status = 'waiting' AND position > (SELECT position FROM queue WHERE id=?)
        """, (queue_id, queue_id))
    conn.commit()


def remove_queue_entry(queue_id):
    """Remove a waiting patient from the queue."""
    conn = get_db()
    row = conn.execute("SELECT appointment_id, doctor_id, position FROM queue WHERE id=?",
                       (queue_id,)).fetchone()
    if row:
        conn.execute("DELETE FROM queue WHERE id=?", (queue_id,))
        conn.execute("UPDATE appointments SET status='pending' WHERE id=?",
                     (row['appointment_id'],))
        # Re-number
        conn.execute("""
            UPDATE queue SET position = position - 1
            WHERE doctor_id=? AND status='waiting' AND position > ?
        """, (row['doctor_id'], row['position']))
    conn.commit()


def get_doctor_slot(doctor_id):
    conn = get_db()
    row = conn.execute(
        "SELECT slot_minutes FROM doctor_settings WHERE doctor_id=?",
        (doctor_id,)
    ).fetchone()
    return row['slot_minutes'] if row else 15


def set_doctor_slot(doctor_id, slot_minutes):
    conn = get_db()
    conn.execute("""
        INSERT INTO doctor_settings (doctor_id, slot_minutes)
        VALUES (?, ?)
        ON CONFLICT(doctor_id) DO UPDATE SET slot_minutes=excluded.slot_minutes
    """, (doctor_id, slot_minutes))
    conn.commit()


def get_queue_entry_for_appointment(appointment_id):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM queue WHERE appointment_id=? AND status IN ('waiting','in-progress')",
        (appointment_id,)
    ).fetchone()
    return dict(row) if row else None


def get_appointment_stats():
    """Returns global appointment counts for admin dashboard."""
    conn = get_db()
    stats = {}
    stats['total']    = conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0]
    stats['pending']  = conn.execute("SELECT COUNT(*) FROM appointments WHERE status='pending'").fetchone()[0]
    stats['accepted'] = conn.execute("SELECT COUNT(*) FROM appointments WHERE status='accepted'").fetchone()[0]
    stats['completed']= conn.execute("SELECT COUNT(*) FROM appointments WHERE status='completed'").fetchone()[0]
    stats['rejected'] = conn.execute("SELECT COUNT(*) FROM appointments WHERE status='rejected'").fetchone()[0]
    stats['in_queue'] = conn.execute("SELECT COUNT(*) FROM queue WHERE status IN ('waiting','in-progress')").fetchone()[0]
    return stats


# ── Server-side session helpers ───────────────────────────────

import secrets

def create_session_token(role, user_id):
    """Create a unique server-side session token and store it in the DB."""
    token = secrets.token_hex(32)
    conn  = get_db()
    conn.execute(
        "INSERT INTO active_sessions (token, role, user_id) VALUES (?, ?, ?)",
        (token, role, user_id)
    )
    conn.commit()
    return token

def validate_session_token(token, role):
    """Return user_id if token is valid for the given role, else None."""
    if not token:
        return None
    conn = get_db()
    row  = conn.execute(
        "SELECT user_id FROM active_sessions WHERE token=? AND role=?",
        (token, role)
    ).fetchone()
    return row['user_id'] if row else None

def destroy_session_token(token):
    """Delete a session token (called on logout)."""
    if not token:
        return
    conn = get_db()
    conn.execute("DELETE FROM active_sessions WHERE token=?", (token,))
    conn.commit()

def wipe_all_sessions():
    """Called on server startup — invalidates all existing sessions."""
    conn = _raw_conn()
    conn.execute("DELETE FROM active_sessions")
    conn.commit()
    conn.close()
    print("[MedWay] All sessions wiped — all users must log in again.")


def get_available_doctors(dept_id):
    """
    Return doctors whose department matches the given dept_id.
    Uses the shared get_db() connection — no extra SQLite connections opened.
    """
    conn = get_db()

    # Look up department name from ml_departments
    dept_row = conn.execute(
        "SELECT department_name FROM ml_departments WHERE dept_id = ?",
        (dept_id,)
    ).fetchone()

    if not dept_row:
        return []

    dept_name = dept_row['department_name']

    # Match doctors by department name (case-insensitive)
    doctors = conn.execute(
        "SELECT doctor_id, name, department, email, phone "
        "FROM doctors WHERE LOWER(department) = LOWER(?)",
        (dept_name,)
    ).fetchall()

    return [dict(d) for d in doctors]


# ── Patient CRUD ──────────────────────────────────────────────

def get_patient(email):
    conn = get_db()
    row  = conn.execute("SELECT * FROM patients WHERE email=?",
                        (email.lower(),)).fetchone()
    return dict(row) if row else None

def get_all_patients():
    conn = get_db()
    rows = conn.execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()
    return {r['email']: dict(r) for r in rows}

def patient_exists(email):
    return get_patient(email) is not None

def create_patient(email, full_name, phone, dob, gender, password):
    conn = get_db()
    conn.execute(
        "INSERT INTO patients (email,full_name,phone,dob,gender,password) VALUES (?,?,?,?,?,?)",
        (email.lower(), full_name, phone, dob, gender, hash_password(password))
    )
    conn.commit()

def verify_patient(email, password):
    p = get_patient(email)
    return p if (p and check_password(password, p['password'])) else None

def delete_patient(email):
    conn = get_db()
    conn.execute("DELETE FROM patients WHERE email=?", (email.lower(),))
    conn.commit()


# ── Doctor CRUD ───────────────────────────────────────────────

def get_doctor(doctor_id):
    conn = get_db()
    row  = conn.execute("SELECT * FROM doctors WHERE doctor_id=?",
                        (doctor_id.lower(),)).fetchone()
    return dict(row) if row else None

def get_all_doctors():
    conn = get_db()
    rows = conn.execute("SELECT * FROM doctors ORDER BY created_at DESC").fetchall()
    return {r['doctor_id']: dict(r) for r in rows}

def doctor_id_exists(doctor_id):
    return get_doctor(doctor_id) is not None

def create_doctor(doctor_id, name, department, email, phone, password):
    conn = get_db()
    conn.execute(
        "INSERT INTO doctors (doctor_id,name,department,email,phone,password) VALUES (?,?,?,?,?,?)",
        (doctor_id.lower(), name, department,
         email.lower(), phone, hash_password(password))
    )
    conn.commit()

def verify_doctor(doctor_id, password):
    d = get_doctor(doctor_id)
    return d if (d and check_password(password, d['password'])) else None

def delete_doctor(doctor_id):
    conn = get_db()
    conn.execute("DELETE FROM doctors WHERE doctor_id=?", (doctor_id.lower(),))
    conn.commit()


# ── Admin CRUD ────────────────────────────────────────────────

def get_admin(admin_id):
    conn = get_db()
    row  = conn.execute("SELECT * FROM admins WHERE admin_id=?",
                        (admin_id.lower(),)).fetchone()
    return dict(row) if row else None

def get_all_admins():
    conn = get_db()
    rows = conn.execute("SELECT * FROM admins ORDER BY created_at DESC").fetchall()
    return {r['admin_id']: dict(r) for r in rows}

def admin_id_exists(admin_id):
    return get_admin(admin_id) is not None

def create_admin(admin_id, name, email, password):
    conn = get_db()
    conn.execute(
        "INSERT INTO admins (admin_id,name,email,password) VALUES (?,?,?,?)",
        (admin_id.lower(), name, email.lower(), hash_password(password))
    )
    conn.commit()

def verify_admin(admin_id, password):
    a = get_admin(admin_id)
    return a if (a and check_password(password, a['password'])) else None


# ══════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════

EMAIL_RE        = re.compile(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$')
PHONE_RE        = re.compile(r'^\d{10}$')
PHONE_REPEAT_RE = re.compile(r'^(\d)\1{9}$')
PHONE_SEQ       = {'0123456789', '9876543210', '1234567890'}


def to_obj(d, **extra):
    class Obj: pass
    o = Obj()
    for k, v in {**d, **extra}.items():
        setattr(o, k, v)
    return o


# ══════════════════════════════════════════════════════════════
# ROUTES — ROOT
# ══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return redirect(url_for('patient_login'))


# ══════════════════════════════════════════════════════════════
# ROUTES — PATIENT
# ══════════════════════════════════════════════════════════════

@app.route('/patient/login', methods=['GET', 'POST'])
def patient_login():
    # Already logged in with valid token?
    token = session.get('patient_token')
    if token and validate_session_token(token, 'patient'):
        return redirect(url_for('patient_dashboard'))

    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        patient  = verify_patient(email, password)
        if patient:
            token = create_session_token('patient', email)
            session.permanent = False
            session['patient_token'] = token
            session['patient_email'] = email
            return redirect(url_for('patient_dashboard'))
        flash('Invalid email or password. Please try again.', 'error')
    return render_template('p_login.html')


def _get_patient_or_abort():
    """
    Validate server-side session token for patient routes.
    Returns (patient_dict, email) if valid, else (None, None).
    Token is checked against the DB — wiped on server restart or logout.
    """
    token = session.get('patient_token')
    email = validate_session_token(token, 'patient')
    if not email:
        session.clear()
        return None, None
    data = get_patient(email)
    if not data:
        destroy_session_token(token)
        session.clear()
        return None, None
    return data, email


@app.route('/patient/register', methods=['GET', 'POST'])
def patient_register():
    token = session.get('patient_token')
    if token and validate_session_token(token, 'patient'):
        return redirect(url_for('patient_dashboard'))

    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email     = request.form.get('email', '').strip().lower()
        phone     = request.form.get('phone', '').strip()
        dob       = request.form.get('dob', '').strip()
        gender    = request.form.get('gender', '').strip()
        password  = request.form.get('password', '')
        confirm   = request.form.get('confirm_password', '')

        errors = []
        if not full_name:
            errors.append('Full name is required.')
        if not email:
            errors.append('Email is required.')
        elif not EMAIL_RE.match(email):
            errors.append('Enter a valid email address.')
        elif patient_exists(email):
            errors.append('An account with this email already exists.')
        if not phone:
            errors.append('Phone number is required.')
        elif not PHONE_RE.match(phone):
            errors.append('Phone must be exactly 10 digits.')
        elif PHONE_REPEAT_RE.match(phone):
            errors.append('Phone cannot have all identical digits.')
        elif phone in PHONE_SEQ:
            errors.append('Phone cannot be a simple sequence.')
        if not dob:
            errors.append('Date of birth is required.')
        if not gender:
            errors.append('Please select a gender.')
        if not password:
            errors.append('Password is required.')
        elif len(password) < 8:
            errors.append('Password must be at least 8 characters.')
        elif password != confirm:
            errors.append('Passwords do not match.')

        if errors:
            for e in errors:
                flash(e, 'error')
            return render_template('p_register.html')

        create_patient(email, full_name, phone, dob, gender, password)
        flash('Account created! Please sign in.', 'success')
        return redirect(url_for('patient_login'))
    return render_template('p_register.html')


@app.route('/patient/dashboard')
def patient_dashboard():
    data, email = _get_patient_or_abort()
    if not data:
        flash('Your session has expired. Please sign in again.', 'error')
        return redirect(url_for('patient_login'))
    recent = get_appointments_for_patient(email)[:3]
    return render_template('p_dashboard.html', patient=to_obj(data), recent_appointments=recent)


@app.route('/patient/logout')
def patient_logout():
    token = session.get('patient_token')
    destroy_session_token(token)
    session.clear()
    # Redirect via a tiny HTML page that wipes sessionStorage first
    return make_response("""<!DOCTYPE html>
<html><head><title>Logging out…</title></head>
<body>
<script>
  try { sessionStorage.removeItem('medway_symptoms'); } catch(e) {}
  window.location.replace('/patient/login');
</script>
</body></html>""")


@app.route('/patient/forgot-password')
def patient_forgot_password():
    return render_template('p_forgot.html')


# ══════════════════════════════════════════════════════════════
# ROUTES — PHASE 2: SYMPTOM INPUT & PREDICTION
# ══════════════════════════════════════════════════════════════

@app.route('/patient/symptoms')
def patient_symptoms():
    data, email = _get_patient_or_abort()
    if not data:
        flash('Your session has expired. Please sign in again.', 'error')
        return redirect(url_for('patient_login'))

    try:
        conn  = get_db()
        rows  = conn.execute(
            "SELECT symptom_name FROM ml_symptoms ORDER BY symptom_name"
        ).fetchall()
        symptoms_list = [r['symptom_name'] for r in rows]
    except Exception:
        symptoms_list = SYMPTOMS_CACHE

    resp = make_response(render_template('p_symptoms.html',
                                         patient=to_obj(data),
                                         symptoms_list=symptoms_list))
    # Allow browser to cache this page so back-navigation restores it
    # from bfcache without a server round-trip (no reload, no flash)
    resp.headers['Cache-Control'] = 'private, max-age=0, must-revalidate'
    return resp


@app.route('/patient/predict', methods=['POST'])
def patient_predict():
    data, email = _get_patient_or_abort()
    if not data:
        return redirect(url_for('patient_login'))

    raw_json = request.form.get('symptoms_json', '[]')
    try:
        symptom_names = json.loads(raw_json)
    except Exception:
        symptom_names = []

    if not symptom_names:
        flash('Please add at least one symptom before analysing.', 'error')
        return redirect(url_for('patient_symptoms'))

    if not ML_AVAILABLE:
        flash('Prediction model not ready. Run: python ml/train_model.py', 'error')
        return redirect(url_for('patient_symptoms'))

    # ── Compute patient age and gender ───────────────────────
    age    = None
    gender = (data.get('gender') or '').lower().strip()
    try:
        from datetime import date as _date
        dob_str = data.get('dob', '')
        if dob_str:
            dob = _date.fromisoformat(dob_str)
            today = _date.today()
            age = today.year - dob.year - (
                (today.month, today.day) < (dob.month, dob.day)
            )
    except Exception:
        age = None

    try:
        matched = match_symptoms(symptom_names)
        if not matched:
            flash('None of your symptoms were recognised. Try common terms like "fever" or "headache".', 'error')
            return redirect(url_for('patient_symptoms'))
        raw_predictions = ml_predict(matched, top_n=8)
    except RuntimeError as e:
        flash(str(e), 'error')
        return redirect(url_for('patient_symptoms'))

    # ── Apply age + gender filters ────────────────────────────
    predictions = apply_smart_filters(raw_predictions, age, gender, matched)

    if not predictions:
        return render_template('p_results.html',
                               patient=to_obj(data),
                               symptoms=matched,
                               symptoms_json=raw_json,
                               predictions=[],
                               doctors=[],
                               patient_age=age,
                               patient_gender=gender)

    top_dept_id = predictions[0]['dept_id']
    doctors     = get_available_doctors(top_dept_id)

    return render_template('p_results.html',
                           patient=to_obj(data),
                           symptoms=matched,
                           symptoms_json=json.dumps(matched),
                           predictions=predictions,
                           doctors=doctors,
                           patient_age=age,
                           patient_gender=gender)


def _calc_age(dob_str):
    """Return integer age from ISO date string, or None."""
    try:
        from datetime import date
        dob   = date.fromisoformat(dob_str)
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except Exception:
        return None


def apply_smart_filters(predictions, age, gender, symptoms):
    """
    Smart post-processing of ML predictions using patient age + gender.

    Steps:
      1. Hard exclusions  — remove biologically/age impossible results
      2. Confidence tuning — boost or penalise based on age/gender context
      3. Rule-based engine — symptom patterns force a department override
      4. Re-rank and return top 3

    Department IDs:
      D01 General Medicine  D02 Cardiology     D03 Neurology
      D04 Orthopaedics      D05 Dermatology    D06 Gastroenterology
      D07 Pulmonology       D08 Endocrinology  D09 Nephrology
      D10 Urology           D11 Gynaecology    D12 Paediatrics
      D13 Ophthalmology     D14 ENT            D15 Psychiatry
      D16 Oncology          D17 Haematology    D18 Rheumatology
      D19 Infectious Disease D20 Emergency Medicine
    """
    if not predictions:
        return []

    g   = (gender or '').lower().strip()   # 'male' | 'female' | 'other' | ''
    a   = age                               # int or None
    sym = set(s.lower().strip() for s in (symptoms or []))

    # ─────────────────────────────────────────────────────────
    # STEP 1 — HARD EXCLUSIONS (remove impossible predictions)
    # ─────────────────────────────────────────────────────────

    # Diseases that ONLY apply to females
    FEMALE_ONLY_DISEASES = {
        'endometriosis', 'uterine fibroids', 'ovarian cyst', 'cervical cancer',
        'ovarian cancer', 'uterine cancer', 'polycystic ovarian syndrome',
        'pelvic inflammatory disease', 'vaginitis', 'menorrhagia', 'dysmenorrhoea',
        'premature ovarian insufficiency', 'ectopic pregnancy', 'preeclampsia',
        'gestational diabetes', 'vulvodynia', 'bartholin cyst', 'amenorrhoea',
        'menopausal syndrome', 'gestational hypertension', 'adenomyosis',
        "asherman's syndrome", 'mullerian anomalies', 'placenta praevia',
        'placental abruption', 'ovarian hyperstimulation syndrome',
        'chronic pelvic pain', 'pelvic organ prolapse',
    }

    # Diseases that ONLY apply to males
    MALE_ONLY_DISEASES = {
        'benign prostatic hyperplasia', 'prostate cancer', 'prostatitis',
        'epididymitis', 'hydrocele', 'varicocele', 'testicular torsion', 'orchitis',
    }

    # Diseases that only apply to children (age < 18)
    CHILD_ONLY_DISEASES = {
        'childhood asthma', 'febrile seizures', 'kawasaki disease', 'croup',
        'rsv infection', 'hand foot mouth disease', 'whooping cough', 'measles',
        'mumps', 'rubella', 'neonatal jaundice', 'failure to thrive',
        'cerebral malaria', 'congenital rubella syndrome', 'phenylketonuria',
        'down syndrome', 'neonatal sepsis', 'hirschsprung disease',
        'congenital adrenal hyperplasia', 'biliary atresia', 'galactosaemia',
        'maple syrup urine disease',
    }

    # Infant-only diseases (age ≤ 2)
    INFANT_ONLY_DISEASES = {
        'neonatal jaundice', 'neonatal sepsis', 'biliary atresia',
        'hirschsprung disease', 'galactosaemia', 'maple syrup urine disease',
        'failure to thrive',
    }

    # Diseases rare / inappropriate under age 18 (elderly-typical)
    ADULT_ONLY_DISEASES = {
        "alzheimer's disease", 'vascular dementia', "parkinson's disease",
        'osteoporosis', 'macular degeneration', 'benign prostatic hyperplasia',
        'prostate cancer', 'menopausal syndrome',
    }

    # Gestational diseases — reproductive-age females only (13–55)
    GESTATIONAL_DISEASES = {
        'gestational diabetes', 'gestational hypertension', 'preeclampsia',
        'ectopic pregnancy', 'placenta praevia', 'placental abruption',
        'ovarian hyperstimulation syndrome',
    }

    def _exclude(p):
        """Return True if this prediction should be hard-removed."""
        dept = p.get('dept_id', '')
        dis  = p.get('disease', '').lower()

        # ── Gender gates ──
        # Gynaecology department: only adult females (13+)
        if dept == 'D11':
            if g in ('male', 'other', 'prefer_not'):
                return True
            if g == 'female' and a is not None and a < 13:
                return True   # no gynaecology for girls under 13

        # Female-only diseases for non-females
        if dis in FEMALE_ONLY_DISEASES and g not in ('female', ''):
            return True

        # Male-only diseases for females
        if dis in MALE_ONLY_DISEASES and g == 'female':
            return True

        # Male-only diseases for under-18 (prostate/testicular diseases don't occur in children)
        PROSTATE_DISEASES = {'benign prostatic hyperplasia', 'prostate cancer', 'prostatitis'}
        if dis in PROSTATE_DISEASES and a is not None and a < 18:
            return True

        # ── Age gates ──
        if a is not None:
            # Paediatrics (D12): only under 18
            if dept == 'D12' and a >= 18:
                return True

            # Child-only diseases for adults
            if dis in CHILD_ONLY_DISEASES and a >= 18:
                return True

            # Infant-only diseases for children over 2
            if dis in INFANT_ONLY_DISEASES and a > 2:
                return True

            # Gestational: only females aged 13–55
            if dis in GESTATIONAL_DISEASES:
                if g != 'female' or a < 13 or a > 55:
                    return True

            # Adult-only diseases for under-18
            if dis in ADULT_ONLY_DISEASES and a < 18:
                return True

            # Menopausal syndrome: females 40+
            if dis == 'menopausal syndrome' and (g != 'female' or a < 40):
                return True

        return False

    filtered = [dict(p) for p in predictions if not _exclude(p)]

    # Safety net: if everything got filtered, fall back to original with only
    # the clearly impossible ones removed (avoid showing empty results)
    if not filtered:
        filtered = [dict(p) for p in predictions
                    if not (p.get('dept_id') == 'D11' and g in ('male', 'other', 'prefer_not'))]
        if not filtered:
            filtered = [dict(p) for p in predictions]

    # ─────────────────────────────────────────────────────────
    # STEP 2 — CONFIDENCE TUNING (boost / penalise by context)
    # ─────────────────────────────────────────────────────────
    for p in filtered:
        dept = p.get('dept_id', '')
        conf = float(p.get('confidence', 0))

        if a is not None:
            # Paediatrics boost for children
            if dept == 'D12' and a < 18:
                conf = min(conf * 1.20, 99)

            # Cardiology boost: higher risk at 40+
            if dept == 'D02':
                if a >= 60:   conf = min(conf * 1.20, 99)
                elif a >= 40: conf = min(conf * 1.10, 99)

            # Oncology: increases with age
            if dept == 'D16':
                if a >= 60:   conf = min(conf * 1.15, 99)
                elif a >= 40: conf = min(conf * 1.07, 99)

            # Rheumatology: more likely in adults
            if dept == 'D18' and a < 18:
                conf *= 0.6

            # Elderly typical depts boost for 65+
            if dept in ('D03', 'D08') and a >= 65:
                conf = min(conf * 1.10, 99)

            # Reduce adult chronic disease confidence for young patients (<25)
            if dept in ('D08', 'D09') and a < 25:
                conf *= 0.75

        # Gynaecology boost for reproductive-age females (18–45)
        if dept == 'D11' and g == 'female' and a is not None and 18 <= a <= 45:
            conf = min(conf * 1.10, 99)

        p['confidence'] = round(conf, 1)

    # ─────────────────────────────────────────────────────────
    # STEP 3 — RULE-BASED ENGINE
    # Symptom pattern combinations → forced department + disease
    # Rules are checked in priority order; first match wins.
    # ─────────────────────────────────────────────────────────

    def has(*terms):
        """True if ANY of the given terms appear in the symptom set."""
        return any(t in sym for t in terms)

    def has_all(*terms):
        """True if ALL of the given terms appear in the symptom set."""
        return all(t in sym for t in terms)

    rule_dept    = None
    rule_disease = None

    # ── EMERGENCY rules (highest priority) ───────────────────
    # Heart attack
    if has('chest pain', 'chest tightness', 'chest pressure') and \
       has('shortness of breath', 'breathlessness', 'sweating', 'left arm pain', 'jaw pain', 'nausea'):
        rule_dept = 'D02'; rule_disease = 'Acute Coronary Syndrome'

    # Stroke
    elif has('facial drooping', 'face drooping', 'sudden confusion', 'slurred speech') and \
         has('weakness', 'arm weakness', 'leg weakness', 'numbness', 'sudden headache'):
        rule_dept = 'D20'; rule_disease = 'Stroke — Emergency'

    # Anaphylaxis
    elif has('severe allergic reaction', 'throat swelling', 'tongue swelling') and \
         has('difficulty breathing', 'breathlessness', 'rash', 'hives'):
        rule_dept = 'D20'; rule_disease = 'Anaphylaxis — Emergency'

    # Head trauma
    elif has('head injury', 'head trauma') and \
         has('loss of consciousness', 'confusion', 'vomiting', 'seizure', 'convulsion'):
        rule_dept = 'D20'; rule_disease = 'Head Trauma — Emergency'

    # Diabetic emergency
    elif has('loss of consciousness', 'unconscious') and \
         has('high blood sugar', 'low blood sugar', 'diabetes', 'excessive thirst', 'frequent urination'):
        rule_dept = 'D08'; rule_disease = 'Diabetic Emergency'

    # ── PAEDIATRICS (age < 18) ────────────────────────────────
    elif a is not None and a < 18 and \
         has('fever', 'high fever') and \
         has('rash', 'convulsion', 'febrile seizure', 'fits', 'seizure'):
        rule_dept = 'D12'; rule_disease = 'Febrile Illness (Paediatric)'

    elif a is not None and a < 18 and \
         has('ear pain', 'ear discharge') and has('fever', 'irritability', 'hearing loss'):
        rule_dept = 'D12'; rule_disease = 'Otitis Media (Paediatric)'

    # ── GYNAECOLOGY (females only, 13+) ──────────────────────
    elif g == 'female' and (a is None or a >= 13) and \
         has('irregular periods', 'missed period', 'no period', 'heavy periods',
             'pelvic pain', 'lower abdominal pain', 'vaginal discharge',
             'vaginal bleeding', 'menstrual irregularity', 'menorrhagia',
             'dysmenorrhoea', 'period cramps'):
        rule_dept = 'D11'; rule_disease = 'Gynaecological Condition'

    elif g == 'female' and (a is None or a >= 13) and \
         has('pregnancy', 'pregnant', 'morning sickness', 'missed period') and \
         has('nausea', 'vomiting', 'breast tenderness', 'fatigue'):
        rule_dept = 'D11'; rule_disease = 'Pregnancy-Related Condition'

    # ── CARDIOLOGY ────────────────────────────────────────────
    elif has('chest pain', 'chest tightness') and \
         has('palpitations', 'irregular heartbeat', 'racing heart', 'heart pounding'):
        rule_dept = 'D02'; rule_disease = 'Cardiac Arrhythmia'

    elif has('breathlessness', 'shortness of breath') and \
         has('leg swelling', 'ankle swelling', 'fatigue') and \
         has('chest pain', 'orthopnoea', 'paroxysmal nocturnal dyspnoea'):
        rule_dept = 'D02'; rule_disease = 'Heart Failure'

    # ── NEUROLOGY ─────────────────────────────────────────────
    elif has('severe headache', 'worst headache', 'thunderclap headache') and \
         has('neck stiffness', 'photophobia', 'fever', 'vomiting'):
        rule_dept = 'D03'; rule_disease = 'Meningitis / Subarachnoid Haemorrhage'

    elif has('tremor', 'shaking', 'rigidity', 'slow movement', 'shuffling gait'):
        rule_dept = 'D03'; rule_disease = 'Parkinsonian Syndrome'

    elif has('seizure', 'convulsion', 'fits', 'epilepsy') and \
         not has('fever', 'high fever') and (a is None or a >= 18):
        rule_dept = 'D03'; rule_disease = 'Seizure Disorder'

    # ── PULMONOLOGY ───────────────────────────────────────────
    elif has('cough', 'persistent cough', 'chronic cough') and \
         has('blood in sputum', 'haemoptysis', 'coughing blood'):
        rule_dept = 'D07'; rule_disease = 'Haemoptysis — Pulmonology Referral'

    elif has('breathlessness', 'wheezing', 'shortness of breath') and \
         has('cough', 'chest tightness') and \
         not has('chest pain', 'palpitations'):
        rule_dept = 'D07'; rule_disease = 'Obstructive Airway Disease'

    # ── GASTROENTEROLOGY ──────────────────────────────────────
    elif has('blood in stool', 'rectal bleeding', 'black stool', 'melaena'):
        rule_dept = 'D06'; rule_disease = 'Lower GI Bleed — Gastroenterology'

    elif has('vomiting blood', 'haematemesis'):
        rule_dept = 'D06'; rule_disease = 'Upper GI Bleed — Gastroenterology'

    elif has('jaundice', 'yellow skin', 'yellow eyes', 'dark urine') and \
         has('abdominal pain', 'nausea', 'fatigue'):
        rule_dept = 'D06'; rule_disease = 'Hepatic / Biliary Condition'

    # ── ENDOCRINOLOGY ─────────────────────────────────────────
    elif has('excessive thirst', 'polydipsia') and \
         has('frequent urination', 'polyuria') and \
         has('weight loss', 'fatigue', 'blurred vision'):
        rule_dept = 'D08'; rule_disease = 'Diabetes Mellitus'

    elif has('weight gain', 'fatigue', 'cold intolerance', 'constipation', 'hair loss') and \
         has('dry skin', 'sluggishness', 'puffiness', 'slow heart rate'):
        rule_dept = 'D08'; rule_disease = 'Hypothyroidism'

    elif has('weight loss', 'heat intolerance', 'sweating', 'palpitations') and \
         has('tremor', 'anxiety', 'diarrhoea', 'bulging eyes', 'goitre'):
        rule_dept = 'D08'; rule_disease = 'Hyperthyroidism'

    # ── DERMATOLOGY ───────────────────────────────────────────
    elif has('rash', 'skin rash', 'itching', 'itchy skin', 'skin lesion',
             'hives', 'urticaria', 'eczema', 'psoriasis', 'skin blisters') and \
         not has('fever', 'chest pain', 'breathlessness', 'shortness of breath'):
        rule_dept = 'D05'; rule_disease = 'Dermatological Condition'

    # ── ENT ───────────────────────────────────────────────────
    elif has('ear pain', 'earache', 'hearing loss', 'tinnitus', 'ear discharge',
             'nasal congestion', 'runny nose', 'sneezing', 'sore throat',
             'throat pain', 'hoarseness', 'loss of smell', 'nosebleed') and \
         not has('fever', 'chest pain', 'breathlessness', 'rash'):
        rule_dept = 'D14'; rule_disease = 'ENT Condition'

    # ── OPHTHALMOLOGY ─────────────────────────────────────────
    elif has('eye pain', 'red eye', 'eye redness', 'blurred vision', 'double vision',
             'vision loss', 'eye discharge', 'floaters', 'flashes of light',
             'eye swelling', 'watery eyes'):
        rule_dept = 'D13'; rule_disease = 'Ophthalmic Condition'

    # ── UROLOGY ───────────────────────────────────────────────
    elif has('painful urination', 'burning urination', 'blood in urine',
             'haematuria', 'frequent urination', 'urinary urgency',
             'inability to urinate', 'urine retention') and \
         not has('excessive thirst', 'weight loss', 'diabetes'):
        rule_dept = 'D10'; rule_disease = 'Urological Condition'

    # ── ORTHOPAEDICS ──────────────────────────────────────────
    elif has('joint pain', 'knee pain', 'hip pain', 'bone pain',
             'fracture', 'sprain', 'back pain', 'neck pain',
             'muscle weakness', 'limping', 'swollen joint') and \
         not has('fever', 'rash', 'fatigue'):
        rule_dept = 'D04'; rule_disease = 'Musculoskeletal Condition'

    # ── PSYCHIATRY ────────────────────────────────────────────
    elif has('anxiety', 'panic attack', 'depression', 'low mood',
             'hallucination', 'delusion', 'suicidal thoughts', 'self-harm',
             'mood swings', 'psychosis', 'paranoia') and \
         not has('fever', 'chest pain', 'breathlessness', 'rash'):
        rule_dept = 'D15'; rule_disease = 'Mental Health Condition'

    elif has('insomnia', 'inability to sleep', 'sleep disorder', 'nightmares',
             'excessive sleepiness') and \
         not has('fever', 'cough', 'pain'):
        rule_dept = 'D15'; rule_disease = 'Sleep / Psychiatric Condition'

    # ── HAEMATOLOGY ───────────────────────────────────────────
    elif has('easy bruising', 'prolonged bleeding', 'spontaneous bleeding',
             'blood clots', 'swollen lymph nodes', 'pallor') and \
         has('fatigue', 'weakness', 'dizziness'):
        rule_dept = 'D17'; rule_disease = 'Haematological Condition'

    # ─────────────────────────────────────────────────────────
    # Apply rule override — inject at top
    # ─────────────────────────────────────────────────────────
    if rule_dept:
        # Remove duplicate of same dept so it doesn't appear twice
        filtered = [p for p in filtered if p.get('dept_id') != rule_dept]

        # Resolve department name from DB
        dept_name = rule_dept  # fallback
        try:
            conn = get_db()
            row  = conn.execute(
                "SELECT department_name FROM ml_departments WHERE dept_id=?",
                (rule_dept,)
            ).fetchone()
            if row:
                dept_name = row['department_name']
        except Exception:
            pass

        filtered.insert(0, {
            'disease':    rule_disease,
            'confidence': 95.0,
            'dept_id':    rule_dept,
            'department': dept_name,
            'rule_based': True,
        })

    # ─────────────────────────────────────────────────────────
    # STEP 4 — RE-RANK and return top 3
    # ─────────────────────────────────────────────────────────
    # Keep rule-based entry pinned at top; sort rest by confidence
    if filtered and filtered[0].get('rule_based'):
        top   = filtered[0]
        rest  = sorted(filtered[1:], key=lambda x: x.get('confidence', 0), reverse=True)
        return [top] + rest[:2]
    else:
        filtered.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return filtered[:3]


@app.route('/patient/doctors/<dept_id>')
def patient_dept_doctors(dept_id):
    data, email = _get_patient_or_abort()
    if not data:
        return redirect(url_for('patient_login'))

    disease       = request.args.get('disease', '')
    symptoms_json = request.args.get('symptoms', '[]')

    try:
        symptoms = json.loads(symptoms_json)
    except Exception:
        symptoms = []

    conn      = get_db()
    dept_row  = conn.execute(
        "SELECT department_name FROM ml_departments WHERE dept_id=?", (dept_id,)
    ).fetchone()
    dept_name = dept_row['department_name'] if dept_row else dept_id
    doctors   = get_available_doctors(dept_id)

    predictions = [{'disease': disease, 'department': dept_name,
                    'dept_id': dept_id, 'confidence': 0}]

    return render_template('p_results.html',
                           patient=to_obj(data),
                           symptoms=symptoms,
                           symptoms_json=symptoms_json,
                           predictions=predictions,
                           doctors=doctors)


@app.route('/patient/book-appointment', methods=['POST'])
def patient_book_appointment():
    data, email = _get_patient_or_abort()
    if not data:
        return redirect(url_for('patient_login'))

    doctor_id     = request.form.get('doctor_id', '').strip()
    disease       = request.form.get('disease', '').strip()
    department    = request.form.get('department', '').strip()
    dept_id       = request.form.get('dept_id', '').strip()
    symptoms_json = request.form.get('symptoms_json', '[]')

    if not doctor_id or not disease:
        flash('Invalid appointment request.', 'error')
        return redirect(url_for('patient_symptoms'))

    doctor = get_doctor(doctor_id)
    if not doctor:
        flash('Doctor not found.', 'error')
        return redirect(url_for('patient_symptoms'))

    appt_id = create_appointment(
        patient_email = email,
        doctor_id     = doctor_id,
        disease       = disease,
        department    = department,
        dept_id       = dept_id,
        symptoms_json = symptoms_json,
        source        = 'online'
    )

    # PRG — store confirmation in session then redirect
    session['last_appt'] = {
        'id':               appt_id,
        'doctor_name':      doctor['name'],
        'department':       department,
        'disease':          disease,
        'symptoms_json':    symptoms_json,
        'requested_at':     datetime.now().strftime('%d %b %Y, %I:%M %p'),
    }
    return redirect(url_for('patient_appointment_confirm'))


@app.route('/patient/appointment/confirm')
def patient_appointment_confirm():
    data, email = _get_patient_or_abort()
    if not data:
        return redirect(url_for('patient_login'))

    appt_data = session.pop('last_appt', None)
    if not appt_data:
        return redirect(url_for('patient_appointments'))

    try:
        syms = json.loads(appt_data.get('symptoms_json', '[]'))
    except Exception:
        syms = []
    appt_data['symptoms_display'] = ', '.join(syms) if syms else '—'

    return render_template('p_appointment_confirm.html',
                           patient=to_obj(data),
                           appointment=appt_data)


@app.route('/patient/appointments')
def patient_appointments():
    data, email = _get_patient_or_abort()
    if not data:
        flash('Your session has expired. Please sign in again.', 'error')
        return redirect(url_for('patient_login'))

    appointments = get_appointments_for_patient(email)
    return render_template('p_appointments.html',
                           patient=to_obj(data),
                           appointments=appointments)


@app.route('/patient/queue/<int:appt_id>')
def patient_queue_view(appt_id):
    data, email = _get_patient_or_abort()
    if not data:
        return redirect(url_for('patient_login'))

    appt = get_appointment_by_id(appt_id)
    if not appt or appt['patient_email'].lower() != email.lower():
        flash('Appointment not found.', 'error')
        return redirect(url_for('patient_appointments'))

    queue_entry = get_queue_entry_for_appointment(appt_id)
    full_queue  = []
    position    = None
    wait_label  = 'Unknown'
    wait_seconds = 0  # for JS countdown

    if queue_entry:
        full_queue = get_queue_for_doctor(appt['doctor_id'])
        for i, q in enumerate(full_queue):
            if q['appointment_id'] == appt_id:
                wait_label   = q.get('wait_label', 'Unknown')
                wait_seconds = q.get('wait_seconds', 0)
                position     = i + 1
                break

    doctor = get_doctor(appt['doctor_id'])

    return render_template('p_queue.html',
                           patient=to_obj(data),
                           appt=appt,
                           queue_entry=queue_entry,
                           full_queue=full_queue,
                           position=position,
                           wait_label=wait_label,
                           wait_seconds=wait_seconds,
                           doctor=doctor,
                           doctor_id=appt['doctor_id'],
                           my_email=email,
                           total_in_queue=len(full_queue))



# ══════════════════════════════════════════════════════════════
# API — Real-time queue data (polled by JS every 3s)
# ══════════════════════════════════════════════════════════════

@app.route('/api/queue/<doctor_id>')
def api_queue(doctor_id):
    """
    Return live queue JSON for a given doctor.
    Accessible by both patient and doctor pages via fetch().
    No auth required — patient names of others are anonymised.
    """
    queue = get_queue_for_doctor(doctor_id)
    public = []
    for idx, e in enumerate(queue):
        public.append({
            'queue_id':          e['id'],
            'appointment_id':    e['appointment_id'],
            'patient_email':     e['patient_email'],
            'status':            e['status'],
            'slot_minutes':      e.get('slot_minutes') or 15,
            'wait_seconds':      e.get('wait_seconds', 0),
            'remaining_seconds': e.get('remaining_seconds'),
            'wait_label':        e.get('wait_label', ''),
            'remaining_label':   e.get('remaining_label', ''),
            'disease':           e.get('disease', ''),
            'patient_name':      e.get('patient_name', ''),
            'position':          idx + 1,
        })
    return jsonify({'queue': public, 'total': len(public)})


@app.route('/api/queue/update-slot', methods=['POST'])
def api_update_slot():
    """AJAX endpoint — update slot_minutes for a specific queue entry."""
    doctor_id    = session.get('doctor_id', '')
    if not doctor_id:
        return jsonify({'ok': False, 'error': 'Not authenticated'}), 401
    queue_id     = request.json.get('queue_id') if request.is_json else request.form.get('queue_id', type=int)
    slot_minutes = request.json.get('slot_minutes') if request.is_json else request.form.get('slot_minutes', type=int)
    if not queue_id or not slot_minutes:
        return jsonify({'ok': False, 'error': 'Missing params'}), 400
    slot_minutes = max(1, min(int(slot_minutes), 240))
    conn = get_db()
    conn.execute(
        "UPDATE queue SET slot_minutes=? WHERE id=? AND doctor_id=?",
        (slot_minutes, queue_id, doctor_id)
    )
    conn.commit()
    queue = get_queue_for_doctor(doctor_id)
    public = [{'queue_id': e['id'], 'appointment_id': e['appointment_id'],
               'patient_email': e['patient_email'], 'status': e['status'],
               'slot_minutes': e.get('slot_minutes') or 15,
               'wait_seconds': e.get('wait_seconds', 0),
               'remaining_seconds': e.get('remaining_seconds'),
               'wait_label': e.get('wait_label', ''),
               'remaining_label': e.get('remaining_label', ''),
               'disease': e.get('disease', ''),
               'patient_name': e.get('patient_name', ''),
               'position': e.get('position') or 0} for e in queue]
    return jsonify({'ok': True, 'queue': public, 'total': len(public)})

# ══════════════════════════════════════════════════════════════
# ROUTES — DOCTOR
# ══════════════════════════════════════════════════════════════

@app.route('/doctor/login', methods=['GET', 'POST'])
def doctor_login():
    token = session.get('doctor_token')
    if token and validate_session_token(token, 'doctor'):
        return redirect(url_for('doctor_dashboard'))
    if request.method == 'POST':
        doctor_id = request.form.get('doctor_id', '').strip().lower()
        password  = request.form.get('password', '')
        doctor    = verify_doctor(doctor_id, password)
        if doctor:
            token = create_session_token('doctor', doctor_id)
            session.permanent = False
            session['doctor_token'] = token
            session['doctor_id']    = doctor_id
            return redirect(url_for('doctor_dashboard'))
        flash('Invalid Doctor ID or password.', 'error')
    return render_template('d_login.html')


def _get_doctor_or_abort():
    token = session.get('doctor_token')
    doctor_id = validate_session_token(token, 'doctor')
    if not doctor_id:
        session.clear()
        return None, None
    doc = get_doctor(doctor_id)
    if not doc:
        destroy_session_token(token)
        session.clear()
        return None, None
    return doc, doctor_id


@app.route('/doctor/dashboard')
def doctor_dashboard():
    doc, doctor_id = _get_doctor_or_abort()
    if not doc:
        flash('Your session has expired. Please sign in again.', 'error')
        return redirect(url_for('doctor_login'))

    all_appts = get_appointments_for_doctor(doctor_id)
    pending_appointments   = [a for a in all_appts if a['status'] == 'pending']
    completed_appointments = [a for a in all_appts if a['status'] == 'completed']
    rejected_appointments  = [a for a in all_appts if a['status'] == 'rejected']

    today = datetime.now().strftime('%Y-%m-%d')
    completed_today = sum(
        1 for a in completed_appointments
        if a.get('requested_at', '').startswith(today)
    )

    queue        = get_queue_for_doctor(doctor_id)
    slot_minutes = get_doctor_slot(doctor_id)

    # Disable "Start" if someone is already in-progress (one at a time rule)
    has_active_patient = any(e['status'] == 'in-progress' for e in queue)

    return render_template(
        'd_dashboard.html',
        doctor                 = to_obj(doc),
        doctor_id              = doctor_id,
        pending_appointments   = pending_appointments,
        completed_appointments = completed_appointments,
        rejected_appointments  = rejected_appointments,
        queue                  = queue,
        slot_minutes           = slot_minutes,
        pending_count          = len(pending_appointments),
        queue_count            = len(queue),
        completed_today        = completed_today,
        has_active_patient     = has_active_patient,
    )


@app.route('/doctor/appointments/accept', methods=['POST'])
def doctor_accept_appointment():
    _doc_tmp, _did_tmp = _get_doctor_or_abort()
    if not _doc_tmp:
        flash('Your session has expired. Please sign in again.', 'error')
        return redirect(url_for('doctor_login'))

    doctor_id    = session.get('doctor_id', '')
    appt_id      = request.form.get('appt_id', type=int)
    slot_minutes = get_doctor_slot(doctor_id)

    appt = get_appointment_by_id(appt_id)
    if not appt or appt['doctor_id'] != doctor_id:
        flash('Appointment not found.', 'error')
        return redirect(url_for('doctor_dashboard'))

    # Accept freely — patient joins waiting queue at next position
    update_appointment_status(appt_id, 'accepted')
    add_to_queue(appt_id, doctor_id, slot_minutes)
    flash('Appointment accepted — patient added to queue.', 'success')
    return redirect(url_for('doctor_dashboard'))


@app.route('/doctor/appointments/reject', methods=['POST'])
def doctor_reject_appointment():
    _doc_tmp, _did_tmp = _get_doctor_or_abort()
    if not _doc_tmp:
        flash('Your session has expired. Please sign in again.', 'error')
        return redirect(url_for('doctor_login'))

    appt_id = request.form.get('appt_id', type=int)
    notes   = request.form.get('notes', '').strip()

    appt = get_appointment_by_id(appt_id)
    if not appt or appt['doctor_id'] != session.get('doctor_id', ''):
        flash('Appointment not found.', 'error')
        return redirect(url_for('doctor_dashboard'))

    update_appointment_status(appt_id, 'rejected', notes or None)
    flash('Appointment rejected.', 'info')
    return redirect(url_for('doctor_dashboard'))


@app.route('/doctor/queue/start', methods=['POST'])
def doctor_queue_start():
    _doc_tmp, _did_tmp = _get_doctor_or_abort()
    if not _doc_tmp:
        flash('Your session has expired. Please sign in again.', 'error')
        return redirect(url_for('doctor_login'))

    doctor_id = session.get('doctor_id', '')
    queue_id  = request.form.get('queue_id', type=int)

    # Block if another patient is already in-progress
    conn        = get_db()
    in_progress = conn.execute(
        "SELECT COUNT(*) FROM queue WHERE doctor_id=? AND status='in-progress'",
        (doctor_id,)
    ).fetchone()[0]

    if in_progress > 0:
        flash('Complete the current patient consultation before starting the next one.', 'error')
        return redirect(url_for('doctor_dashboard'))

    start_queue_entry(queue_id)
    flash('Patient consultation started.', 'success')
    return redirect(url_for('doctor_dashboard'))


@app.route('/doctor/queue/done', methods=['POST'])
def doctor_queue_done():
    _doc_tmp, _did_tmp = _get_doctor_or_abort()
    if not _doc_tmp:
        flash('Your session has expired. Please sign in again.', 'error')
        return redirect(url_for('doctor_login'))
    queue_id = request.form.get('queue_id', type=int)
    complete_queue_entry(queue_id)
    flash('Consultation marked as complete. Next patient!', 'success')
    return redirect(url_for('doctor_dashboard'))


@app.route('/doctor/queue/remove', methods=['POST'])
def doctor_queue_remove():
    _doc_tmp, _did_tmp = _get_doctor_or_abort()
    if not _doc_tmp:
        flash('Your session has expired. Please sign in again.', 'error')
        return redirect(url_for('doctor_login'))
    queue_id = request.form.get('queue_id', type=int)
    remove_queue_entry(queue_id)
    flash('Patient removed from queue.', 'info')
    return redirect(url_for('doctor_dashboard'))


@app.route('/doctor/queue/set-slot', methods=['POST'])
def doctor_set_slot():
    _doc_tmp, _did_tmp = _get_doctor_or_abort()
    if not _doc_tmp:
        flash('Your session has expired. Please sign in again.', 'error')
        return redirect(url_for('doctor_login'))
    slot_minutes = request.form.get('slot_minutes', type=int, default=15)
    slot_minutes = max(1, min(slot_minutes, 240))
    set_doctor_slot(session.get('doctor_id', ''), slot_minutes)
    flash(f'Default slot updated to {slot_minutes} min.', 'success')
    return redirect(url_for('doctor_dashboard'))


@app.route('/doctor/queue/set-patient-slot', methods=['POST'])
def doctor_set_patient_slot():
    """Update slot_minutes for one specific queue entry (per-patient)."""
    _doc_tmp, _did_tmp = _get_doctor_or_abort()
    if not _doc_tmp:
        flash('Your session has expired. Please sign in again.', 'error')
        return redirect(url_for('doctor_login'))
    doctor_id    = session.get('doctor_id', '')
    queue_id     = request.form.get('queue_id', type=int)
    slot_minutes = request.form.get('slot_minutes', type=int, default=15)
    slot_minutes = max(1, min(slot_minutes, 240))
    conn = get_db()
    conn.execute(
        "UPDATE queue SET slot_minutes=? WHERE id=? AND doctor_id=?",
        (slot_minutes, queue_id, doctor_id)
    )
    conn.commit()
    flash(f'Slot set to {slot_minutes} min for this patient.', 'success')
    return redirect(url_for('doctor_dashboard'))


@app.route('/doctor/add-offline-patient', methods=['POST'])
def doctor_add_offline_patient():
    """Doctor adds a walk-in patient directly to the queue."""
    _doc_tmp, _did_tmp = _get_doctor_or_abort()
    if not _doc_tmp:
        flash('Your session has expired. Please sign in again.', 'error')
        return redirect(url_for('doctor_login'))

    patient_name = request.form.get('patient_name', '').strip()
    disease      = request.form.get('disease', '').strip()
    symptoms     = request.form.get('symptoms', '').strip()

    if not patient_name or not disease:
        flash('Patient name and condition are required.', 'error')
        return redirect(url_for('doctor_dashboard'))

    doc      = get_doctor(session.get('doctor_id', ''))
    dept_id  = 'D00'   # generic for offline
    # Try to look up dept_id from doctor's department
    conn = get_db()
    dept_row = conn.execute(
        "SELECT dept_id FROM ml_departments WHERE LOWER(department_name)=LOWER(?)",
        (doc['department'],)
    ).fetchone()
    if dept_row:
        dept_id = dept_row['dept_id']

    appt_id, _ = create_offline_appointment(
        doctor_id    = session.get('doctor_id', ''),
        patient_name = patient_name,
        disease      = disease,
        department   = doc['department'],
        dept_id      = dept_id,
        symptoms_text = symptoms
    )

    slot_minutes = get_doctor_slot(session.get('doctor_id', ''))
    add_to_queue(appt_id, session.get('doctor_id', ''), slot_minutes)
    flash(f'Walk-in patient "{patient_name}" added to queue.', 'success')
    return redirect(url_for('doctor_dashboard'))


@app.route('/doctor/logout')
def doctor_logout():
    token = session.get('doctor_token')
    destroy_session_token(token)
    session.clear()
    return redirect(url_for('doctor_login'))


@app.route('/doctor/forgot-password')
def doctor_forgot_password():
    return render_template('d_forgot.html')


# ══════════════════════════════════════════════════════════════
# ROUTES — ADMIN
# ══════════════════════════════════════════════════════════════

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    token = session.get('admin_token')
    if token and validate_session_token(token, 'admin'):
        return redirect(url_for('admin_dashboard'))
    if request.method == 'POST':
        admin_id = request.form.get('admin_id', '').strip().lower()
        password = request.form.get('password', '')
        admin    = verify_admin(admin_id, password)
        if admin:
            token = create_session_token('admin', admin_id)
            session.permanent = False
            session['admin_token'] = token
            session['admin_id']    = admin_id
            return redirect(url_for('admin_dashboard'))
        flash('Invalid Admin ID or password.', 'error')
    return render_template('a_login.html')


def _get_admin_or_abort():
    token    = session.get('admin_token')
    admin_id = validate_session_token(token, 'admin')
    if not admin_id:
        session.clear()
        return None, None
    admin = get_admin(admin_id)
    if not admin:
        destroy_session_token(token)
        session.clear()
        return None, None
    return admin, admin_id


@app.route('/admin/dashboard')
def admin_dashboard():
    _adm_tmp, _aid_tmp = _get_admin_or_abort()
    if not _adm_tmp:
        flash('Your session has expired. Please sign in again.', 'error')
        return redirect(url_for('admin_login'))
    admin_data = get_admin(session.get('admin_id', ''))
    if not admin_data:
        session.pop('admin_id', None)
        return redirect(url_for('admin_login'))
    return render_template(
        'a_dashboard.html',
        admin    = to_obj(admin_data),
        doctors  = get_all_doctors(),
        admins   = get_all_admins(),
        patients = get_all_patients(),
        appt_stats = get_appointment_stats(),
    )


@app.route('/admin/add-doctor', methods=['POST'])
def admin_add_doctor():
    _adm_tmp, _aid_tmp = _get_admin_or_abort()
    if not _adm_tmp:
        flash('Your session has expired. Please sign in again.', 'error')
        return redirect(url_for('admin_login'))
    name     = request.form.get('doc_name', '').strip()
    doc_id   = request.form.get('doc_id', '').strip().lower()
    dept     = request.form.get('doc_dept', '').strip()
    email    = request.form.get('doc_email', '').strip().lower()
    phone    = request.form.get('doc_phone', '').strip()
    password = request.form.get('doc_password', '')

    errors = []
    if not name:
        errors.append('Doctor name is required.')
    if not doc_id:
        errors.append('Doctor ID is required.')
    elif doctor_id_exists(doc_id):
        errors.append('Doctor ID "{}" already exists.'.format(doc_id))
    if not dept:
        errors.append('Department is required.')
    if not email or not EMAIL_RE.match(email):
        errors.append('Valid email is required.')
    if not PHONE_RE.match(phone) or PHONE_REPEAT_RE.match(phone) or phone in PHONE_SEQ:
        errors.append('Valid 10-digit phone number is required.')
    if not password or len(password) < 8:
        errors.append('Password must be at least 8 characters.')

    if errors:
        for e in errors:
            flash(e, 'error')
    else:
        create_doctor(doc_id, name, dept, email, phone, password)
        flash('{} has been added as a doctor successfully!'.format(name), 'success')
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/delete-doctor/<doc_id>', methods=['POST'])
def admin_delete_doctor(doc_id):
    _adm_tmp, _aid_tmp = _get_admin_or_abort()
    if not _adm_tmp:
        flash('Your session has expired. Please sign in again.', 'error')
        return redirect(url_for('admin_login'))
    doctor = get_doctor(doc_id)
    if doctor:
        delete_doctor(doc_id)
        flash('{} has been removed from the system.'.format(doctor['name']), 'success')
    else:
        flash('Doctor not found.', 'error')
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/add-admin', methods=['POST'])
def admin_add_admin():
    _adm_tmp, _aid_tmp = _get_admin_or_abort()
    if not _adm_tmp:
        flash('Your session has expired. Please sign in again.', 'error')
        return redirect(url_for('admin_login'))
    name     = request.form.get('new_admin_name', '').strip()
    admin_id = request.form.get('new_admin_id', '').strip().lower()
    email    = request.form.get('new_admin_email', '').strip().lower()
    password = request.form.get('new_admin_password', '')

    errors = []
    if not name:
        errors.append('Admin name is required.')
    if not admin_id:
        errors.append('Admin ID is required.')
    elif admin_id_exists(admin_id):
        errors.append('Admin ID "{}" already exists.'.format(admin_id))
    if not email or not EMAIL_RE.match(email):
        errors.append('Valid email is required.')
    if not password or len(password) < 8:
        errors.append('Password must be at least 8 characters.')

    if errors:
        for e in errors:
            flash(e, 'error')
    else:
        create_admin(admin_id, name, email, password)
        flash('{} added as admin successfully!'.format(name), 'success')
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/delete-patient/<path:patient_email>', methods=['POST'])
def admin_delete_patient(patient_email):
    _adm_tmp, _aid_tmp = _get_admin_or_abort()
    if not _adm_tmp:
        flash('Your session has expired. Please sign in again.', 'error')
        return redirect(url_for('admin_login'))
    patient = get_patient(patient_email)
    if patient:
        delete_patient(patient_email)
        flash('{} has been removed from the system.'.format(patient['full_name']), 'success')
    else:
        flash('Patient not found.', 'error')
    return redirect(url_for('admin_dashboard'))



@app.route('/admin/delete-admin/<target_admin_id>', methods=['POST'])
def admin_delete_admin(target_admin_id):
    _adm_tmp, _aid_tmp = _get_admin_or_abort()
    if not _adm_tmp:
        flash('Your session has expired. Please sign in again.', 'error')
        return redirect(url_for('admin_login'))
    # Prevent admin from deleting themselves
    if target_admin_id.lower() == session.get('admin_id', '').lower():
        flash('You cannot delete your own admin account.', 'error')
        return redirect(url_for('admin_dashboard'))
    target = get_admin(target_admin_id)
    if target:
        conn = get_db()
        conn.execute("DELETE FROM admins WHERE admin_id=?", (target_admin_id.lower(),))
        conn.commit()
        flash('{} has been removed as admin.'.format(target['name']), 'success')
    else:
        flash('Admin not found.', 'error')
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/logout')
def admin_logout():
    token = session.get('admin_token')
    destroy_session_token(token)
    session.clear()
    return redirect(url_for('admin_login'))


@app.route('/admin/forgot-password')
def admin_forgot_password():
    flash('Admin password reset requires direct database access. Contact your system administrator.', 'info')
    return redirect(url_for('admin_login'))


# ══════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ══════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500


# ══════════════════════════════════════════════════════════════
# AUTO-SEED — loads CSVs into DB if ml_symptoms table is empty
# ══════════════════════════════════════════════════════════════

def load_symptoms_cache():
    """Load all symptom names into a global cache at startup."""
    global SYMPTOMS_CACHE
    try:
        conn  = _raw_conn()
        rows  = conn.execute(
            "SELECT symptom_name FROM ml_symptoms ORDER BY symptom_name"
        ).fetchall()
        conn.close()
        SYMPTOMS_CACHE = [r['symptom_name'] for r in rows]
        print(f"[MedWay] Symptom cache loaded: {len(SYMPTOMS_CACHE)} symptoms.")
    except Exception as e:
        SYMPTOMS_CACHE = []
        print(f"[MedWay] WARNING: Could not load symptom cache: {e}")


def auto_seed():
    """
    Seed ML tables from CSVs at startup. Always re-seeds if any table is empty.
    No need to run seed_data.py manually.
    """
    import csv as _csv

    conn = _raw_conn()
    c    = conn.cursor()

    # Disable FK checks during seed so insert order doesn't matter
    c.execute("PRAGMA foreign_keys = OFF")

    # Ensure ML tables exist (no FK constraints — simpler, more robust)
    c.execute("""CREATE TABLE IF NOT EXISTS ml_departments (
                    dept_id TEXT PRIMARY KEY, department_name TEXT NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS ml_diseases (
                    disease_id TEXT PRIMARY KEY, disease_name TEXT NOT NULL, dept_id TEXT NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS ml_symptoms (
                    symptom_id TEXT PRIMARY KEY, symptom_name TEXT NOT NULL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS ml_symptom_disease (
                    disease_id TEXT NOT NULL, symptom_id TEXT NOT NULL,
                    PRIMARY KEY (disease_id, symptom_id))""")
    conn.commit()

    # Check counts
    n_sym  = c.execute("SELECT COUNT(*) FROM ml_symptoms").fetchone()[0]
    n_dept = c.execute("SELECT COUNT(*) FROM ml_departments").fetchone()[0]
    n_dis  = c.execute("SELECT COUNT(*) FROM ml_diseases").fetchone()[0]
    n_map  = c.execute("SELECT COUNT(*) FROM ml_symptom_disease").fetchone()[0]

    if n_sym > 0 and n_dept > 0 and n_dis > 0 and n_map > 0:
        print(f"[MedWay] DB already seeded: {n_dept} depts, {n_dis} diseases, "
              f"{n_sym} symptoms, {n_map} mappings.")
        c.execute("PRAGMA foreign_keys = ON")
        conn.close()
        return

    print(f"[MedWay] Seeding DB (depts={n_dept}, diseases={n_dis}, "
          f"symptoms={n_sym}, maps={n_map}) ...")

    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        print(f"[MedWay] ERROR: data/ folder not found at {DATA_DIR}")
        conn.close()
        return

    def read_csv(filename):
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"[MedWay] ERROR: {filename} not found in {DATA_DIR}")
            return []
        try:
            rows = []
            with open(path, newline='', encoding='utf-8-sig') as f:
                for row in _csv.DictReader(f):
                    rows.append({k.strip(): v.strip() for k, v in row.items()})
            print(f"[MedWay]   Read {len(rows)} rows from {filename}")
            return rows
        except Exception as e:
            print(f"[MedWay] ERROR reading {filename}: {e}")
            return []

    # Clear existing data before re-seeding
    c.execute("DELETE FROM ml_symptom_disease")
    c.execute("DELETE FROM ml_symptoms")
    c.execute("DELETE FROM ml_diseases")
    c.execute("DELETE FROM ml_departments")
    conn.commit()

    # Seed departments
    for r in read_csv('departments_master.csv'):
        try:
            c.execute("INSERT OR IGNORE INTO ml_departments VALUES (?,?)",
                      (r['dept_id'], r['department_name']))
        except Exception as e:
            print(f"[MedWay] dept insert error: {e} | row={r}")
    conn.commit()

    # Seed diseases
    for r in read_csv('diseases_master.csv'):
        try:
            c.execute("INSERT OR IGNORE INTO ml_diseases VALUES (?,?,?)",
                      (r['disease_id'], r['disease_name'], r['dept_id']))
        except Exception as e:
            print(f"[MedWay] disease insert error: {e} | row={r}")
    conn.commit()

    # Seed symptoms
    for r in read_csv('symptoms_master.csv'):
        try:
            c.execute("INSERT OR IGNORE INTO ml_symptoms VALUES (?,?)",
                      (r['symptom_id'], r['symptom_name']))
        except Exception as e:
            print(f"[MedWay] symptom insert error: {e} | row={r}")
    conn.commit()

    # Seed symptom-disease map
    # Note: CSV has columns: disease_id, disease_name, symptom_id, symptom_name
    for r in read_csv('symptom_disease_map.csv'):
        try:
            c.execute("INSERT OR IGNORE INTO ml_symptom_disease VALUES (?,?)",
                      (r['disease_id'], r['symptom_id']))
        except Exception as e:
            print(f"[MedWay] map insert error: {e} | row={r}")
    conn.commit()

    # Final counts
    n_dept = c.execute("SELECT COUNT(*) FROM ml_departments").fetchone()[0]
    n_dis  = c.execute("SELECT COUNT(*) FROM ml_diseases").fetchone()[0]
    n_sym  = c.execute("SELECT COUNT(*) FROM ml_symptoms").fetchone()[0]
    n_map  = c.execute("SELECT COUNT(*) FROM ml_symptom_disease").fetchone()[0]
    print(f"[MedWay] Seed complete: {n_dept} depts | {n_dis} diseases | "
          f"{n_sym} symptoms | {n_map} mappings")

    c.execute("PRAGMA foreign_keys = ON")
    conn.close()


# ══════════════════════════════════════════════════════════════
# AUTO-TRAIN — trains ML model if model.pkl is missing
# ══════════════════════════════════════════════════════════════

def auto_train():
    """
    Automatically train the ML model at startup if model.pkl is missing.
    No need to manually run ml/train_model.py.
    """
    ML_DIR     = os.path.join(BASE_DIR, 'ml')
    model_path = os.path.join(ML_DIR, 'model.pkl')

    if os.path.exists(model_path):
        print("[MedWay] ML model already trained. Skipping training.")
        return

    print("[MedWay] ML model not found. Starting auto-training (this may take ~30s)...")

    try:
        import numpy as np
        import sqlite3 as _sqlite3
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        import joblib
    except ImportError as e:
        print(f"[MedWay] WARNING: Cannot auto-train — missing library: {e}")
        print("[MedWay] Run: pip install scikit-learn numpy joblib pandas")
        return

    # Load data
    conn = _raw_conn()
    symptoms = conn.execute(
        "SELECT symptom_id, symptom_name FROM ml_symptoms ORDER BY symptom_id"
    ).fetchall()
    diseases = conn.execute(
        "SELECT d.disease_id, d.disease_name, d.dept_id, dep.department_name "
        "FROM ml_diseases d JOIN ml_departments dep ON d.dept_id = dep.dept_id "
        "ORDER BY d.disease_id"
    ).fetchall()
    mappings = conn.execute(
        "SELECT disease_id, symptom_id FROM ml_symptom_disease"
    ).fetchall()
    conn.close()

    if not symptoms or not diseases:
        print("[MedWay] WARNING: No data to train on. Skipping.")
        return

    print(f"[MedWay] Training on {len(symptoms)} symptoms × {len(diseases)} diseases...")

    symptom_index      = {row['symptom_name']: i for i, row in enumerate(symptoms)}
    symptom_id_to_name = {row['symptom_id']:   row['symptom_name'] for row in symptoms}
    disease_list       = [row['disease_name']  for row in diseases]
    disease_id_map     = {row['disease_id']:   row['disease_name'] for row in diseases}
    disease_dept       = {
        row['disease_name']: (row['dept_id'], row['department_name'])
        for row in diseases
    }

    n_diseases = len(diseases)
    n_symptoms = len(symptoms)
    X = np.zeros((n_diseases, n_symptoms), dtype=np.int8)
    disease_name_to_row = {name: i for i, name in enumerate(disease_list)}

    for m in mappings:
        dis_name = disease_id_map.get(m['disease_id'])
        sym_name = symptom_id_to_name.get(m['symptom_id'])
        if dis_name and sym_name and sym_name in symptom_index:
            X[disease_name_to_row[dis_name], symptom_index[sym_name]] = 1

    le    = LabelEncoder()
    y_enc = le.fit_transform(np.array(disease_list))

    # Augment
    rng  = np.random.default_rng(42)
    Xs, ys = [X], [y_enc]
    for _ in range(6):
        mask = rng.random(X.shape) > rng.uniform(0, 0.35, X.shape)
        Xs.append((X * mask).astype(np.int8))
        ys.append(y_enc)
    X_aug = np.vstack(Xs)
    y_aug = np.concatenate(ys)

    clf = RandomForestClassifier(
        n_estimators=150, max_features='sqrt',
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    clf.fit(X_aug, y_aug)

    os.makedirs(ML_DIR, exist_ok=True)
    joblib.dump(clf,           os.path.join(ML_DIR, 'model.pkl'))
    joblib.dump(symptom_index, os.path.join(ML_DIR, 'symptom_index.pkl'))
    joblib.dump(le,            os.path.join(ML_DIR, 'label_encoder.pkl'))
    joblib.dump(disease_dept,  os.path.join(ML_DIR, 'disease_dept.pkl'))

    print("[MedWay] Auto-training complete! Model saved.")

    # Reload ML module now that model exists
    global ML_AVAILABLE
    try:
        from ml.predict import match_symptoms as _ms, predict as _mp
        import ml.predict as _mlmod
        _mlmod._clf = None  # force reload
        ML_AVAILABLE = True
        print("[MedWay] ML module reloaded successfully.")
    except Exception as e:
        print(f"[MedWay] ML reload warning: {e}")


# ══════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════

with app.app_context():
    init_db()
    auto_seed()          # seed CSVs → DB (runs every startup if any table empty)
    auto_train()         # train ML model if model.pkl missing
    load_symptoms_cache()  # cache symptom names globally
    wipe_all_sessions()  # invalidate all sessions on restart

    # Re-import ML after auto-train
    try:
        from ml.predict import match_symptoms, predict as ml_predict
        ML_AVAILABLE = True
        print("[MedWay] ML engine loaded.")
    except Exception as e:
        ML_AVAILABLE = False
        print(f"[MedWay] ML engine not available: {e}")

if __name__ == '__main__':
    app.run(debug=True)