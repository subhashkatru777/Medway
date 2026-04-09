"""
MedWay – ml/predict.py
======================
Loads the trained model and exposes two public functions:

    match_symptoms(raw_input)  → list of matched symptom names
    predict(symptom_names)     → list of top-N predictions

Usage in app.py:
    from ml.predict import match_symptoms, predict

    matched  = match_symptoms("fever, headache, body ache")
    results  = predict(matched)
    # results = [
    #   {
    #     "disease":    "Malaria",
    #     "confidence": 78.4,
    #     "dept_id":    "D01",
    #     "department": "General Medicine"
    #   },
    #   ...
    # ]
"""

import os
import sys
import numpy as np

ML_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(ML_DIR)

# ── Lazy-load model artefacts (loaded once on first call) ─────
_clf           = None
_symptom_index = None   # {symptom_name: column_index}
_le            = None   # LabelEncoder
_disease_dept  = None   # {disease_name: (dept_id, dept_name)}
_symptom_names = None   # sorted list of all symptom names


def _load():
    global _clf, _symptom_index, _le, _disease_dept, _symptom_names
    if _clf is not None:
        return  # already loaded

    try:
        import joblib
    except ImportError:
        raise RuntimeError("joblib not installed. Run: pip install joblib")

    model_path = os.path.join(ML_DIR, 'model.pkl')
    if not os.path.exists(model_path):
        raise RuntimeError(
            "Model not trained yet.\n"
            "Run:  python ml/train_model.py"
        )

    _clf           = joblib.load(os.path.join(ML_DIR, 'model.pkl'))
    _symptom_index = joblib.load(os.path.join(ML_DIR, 'symptom_index.pkl'))
    _le            = joblib.load(os.path.join(ML_DIR, 'label_encoder.pkl'))
    _disease_dept  = joblib.load(os.path.join(ML_DIR, 'disease_dept.pkl'))
    _symptom_names = sorted(_symptom_index.keys())


# ── Symptom matching ──────────────────────────────────────────

def _normalise(text):
    """Lowercase, strip punctuation, collapse spaces."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def _fuzzy_match(query, threshold=75):
    """
    Match a single symptom query against all known symptoms.
    Uses rapidfuzz if available, falls back to difflib.
    Returns the best matching symptom name or None.
    """
    query = _normalise(query)
    if not query:
        return None

    # Exact match first (fastest)
    if query in _symptom_index:
        return query

    try:
        from rapidfuzz import process, fuzz
        result = process.extractOne(
            query,
            _symptom_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold
        )
        return result[0] if result else None

    except ImportError:
        # Fallback: difflib
        from difflib import get_close_matches
        matches = get_close_matches(query, _symptom_names, n=1, cutoff=threshold / 100)
        return matches[0] if matches else None


def match_symptoms(raw_input):
    """
    Parse a free-text or comma-separated symptom input and
    return a list of matched canonical symptom names.

    Args:
        raw_input (str | list): e.g. "fever, headache, body ache"
                                 or ["fever", "headache"]

    Returns:
        list[str]: matched symptom names from the dataset
    """
    _load()

    if isinstance(raw_input, str):
        # Split on comma or semicolon
        parts = [p.strip() for p in raw_input.replace(';', ',').split(',')]
    else:
        parts = [str(p).strip() for p in raw_input]

    matched = []
    for part in parts:
        if not part:
            continue
        m = _fuzzy_match(part)
        if m and m not in matched:
            matched.append(m)

    return matched


# ── Prediction ────────────────────────────────────────────────

def predict(symptom_names, top_n=3):
    """
    Given a list of matched symptom names, predict the most likely
    diseases and return top_n results with confidence scores.

    Args:
        symptom_names (list[str]): from match_symptoms()
        top_n (int): number of results to return (default 3)

    Returns:
        list[dict]: sorted by confidence descending, e.g.
        [
          {
            "disease":    "Malaria",
            "confidence": 78.4,      # percentage
            "dept_id":    "D01",
            "department": "General Medicine"
          },
          ...
        ]
        Returns empty list if no symptoms matched.
    """
    _load()

    if not symptom_names:
        return []

    # Build binary feature vector
    n_features = len(_symptom_index)
    vec = np.zeros((1, n_features), dtype=np.int8)
    for name in symptom_names:
        if name in _symptom_index:
            vec[0, _symptom_index[name]] = 1

    # Nothing matched in index
    if vec.sum() == 0:
        return []

    # Get probability distribution across all diseases
    proba   = _clf.predict_proba(vec)[0]   # shape: (n_classes,)
    top_idx = np.argsort(proba)[::-1][:top_n]

    results = []
    for idx in top_idx:
        if proba[idx] < 0.01:   # skip negligible probabilities
            continue
        disease_name = _le.inverse_transform([idx])[0]
        dept_info    = _disease_dept.get(disease_name, ('Unknown', 'Unknown'))
        results.append({
            'disease':    disease_name,
            'confidence': round(float(proba[idx]) * 100, 1),
            'dept_id':    dept_info[0],
            'department': dept_info[1],
        })

    return results


# ── Doctor matching (SQLite lookup) ──────────────────────────

def get_available_doctors(dept_id, db_path=None):
    """
    Return list of doctors in the given department from the DB.

    Args:
        dept_id (str): e.g. "D01"
        db_path (str): path to medway.db (auto-detected if None)

    Returns:
        list[dict]: doctor records with id, name, department
    """
    import sqlite3
    if db_path is None:
        db_path = os.path.join(BASE_DIR, 'medway.db')

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Match doctors whose department name matches the dept from ml_departments
    dept_name_row = conn.execute(
        "SELECT department_name FROM ml_departments WHERE dept_id = ?",
        (dept_id,)
    ).fetchone()

    if not dept_name_row:
        conn.close()
        return []

    dept_name = dept_name_row['department_name']

    # Doctors table uses department as plain text (added by admin)
    doctors = conn.execute(
        "SELECT doctor_id, name, department, email, phone "
        "FROM doctors WHERE LOWER(department) = LOWER(?)",
        (dept_name,)
    ).fetchall()

    conn.close()
    return [dict(d) for d in doctors]


# ── CLI test ──────────────────────────────────────────────────

if __name__ == '__main__':
    print("MedWay Symptom Predictor — interactive test")
    print("Type symptoms separated by commas. Type 'quit' to exit.\n")

    while True:
        raw = input("Enter symptoms: ").strip()
        if raw.lower() in ('quit', 'exit', 'q'):
            break
        if not raw:
            continue

        matched  = match_symptoms(raw)
        print(f"  Matched: {matched}")

        if not matched:
            print("  No symptoms matched. Try different terms.\n")
            continue

        results = predict(matched, top_n=3)
        print(f"\n  Top predictions:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['disease']:35s} "
                  f"{r['confidence']:5.1f}%  →  {r['department']}")
        print()
