"""
MedWay – ml/train_model.py
==========================
Builds the symptom-disease feature matrix from SQLite,
trains a Random Forest classifier, evaluates it, and saves
the model + supporting metadata to ml/model.pkl.

Run ONCE (after seed_data.py):

    python ml/train_model.py

Output:
    ml/model.pkl          ← trained RandomForest
    ml/symptom_index.pkl  ← {symptom_name: column_index} lookup
    ml/label_encoder.pkl  ← LabelEncoder (int ↔ disease_name)
    ml/disease_dept.pkl   ← {disease_name: (dept_id, dept_name)}
"""

import os
import sys
import sqlite3
import pickle
import numpy as np

# Allow running from any working directory
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH    = os.path.join(BASE_DIR, 'medway.db')
ML_DIR     = os.path.dirname(os.path.abspath(__file__))

try:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report
    import joblib
except ImportError:
    print("ERROR: Missing libraries. Run:")
    print("  pip install scikit-learn pandas numpy joblib")
    sys.exit(1)


# ── 1. Load data from SQLite ──────────────────────────────────

def load_data():
    print("[train] Connecting to database...")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # All symptoms (sorted for consistent column order)
    symptoms = conn.execute(
        "SELECT symptom_id, symptom_name FROM ml_symptoms ORDER BY symptom_id"
    ).fetchall()

    # All diseases
    diseases = conn.execute(
        "SELECT d.disease_id, d.disease_name, d.dept_id, dep.department_name "
        "FROM ml_diseases d "
        "JOIN ml_departments dep ON d.dept_id = dep.dept_id "
        "ORDER BY d.disease_id"
    ).fetchall()

    # Symptom-disease mapping
    mappings = conn.execute(
        "SELECT disease_id, symptom_id FROM ml_symptom_disease"
    ).fetchall()

    conn.close()
    print(f"[train] Loaded {len(symptoms)} symptoms, {len(diseases)} diseases, "
          f"{len(mappings)} mappings.")
    return symptoms, diseases, mappings


# ── 2. Build feature matrix ───────────────────────────────────

def build_matrix(symptoms, diseases, mappings):
    print("[train] Building feature matrix...")

    # symptom_name → column index
    symptom_index = {row['symptom_name']: i for i, row in enumerate(symptoms)}
    symptom_id_to_name = {row['symptom_id']: row['symptom_name'] for row in symptoms}

    # disease_id → row index
    disease_list  = [row['disease_name'] for row in diseases]
    disease_id_map = {row['disease_id']: row['disease_name'] for row in diseases}

    # disease_name → (dept_id, dept_name)
    disease_dept = {
        row['disease_name']: (row['dept_id'], row['department_name'])
        for row in diseases
    }

    n_diseases  = len(diseases)
    n_symptoms  = len(symptoms)

    # Binary matrix: rows = diseases, cols = symptoms
    X = np.zeros((n_diseases, n_symptoms), dtype=np.int8)
    disease_name_to_row = {name: i for i, name in enumerate(disease_list)}

    for m in mappings:
        dis_name = disease_id_map.get(m['disease_id'])
        sym_name = symptom_id_to_name.get(m['symptom_id'])
        if dis_name and sym_name and sym_name in symptom_index:
            row = disease_name_to_row[dis_name]
            col = symptom_index[sym_name]
            X[row, col] = 1

    y = np.array(disease_list)

    print(f"[train] Matrix shape: {X.shape}  "
          f"(diseases × symptoms)")
    print(f"[train] Avg symptoms per disease: "
          f"{X.sum(axis=1).mean():.1f}")

    return X, y, symptom_index, disease_dept


# ── 3. Train & evaluate ───────────────────────────────────────

def train(X, y):
    print("[train] Encoding labels...")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # With only 573 diseases and clean data, we use all for training.
    # We duplicate each sample slightly for robustness (simulate variations).
    print("[train] Augmenting data (symptom dropout simulation)...")
    X_aug, y_aug = augment(X, y_enc, copies=8)

    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y_aug, test_size=0.15, random_state=42, stratify=y_aug
    )
    print(f"[train] Train: {X_train.shape[0]} samples | "
          f"Test: {X_test.shape[0]} samples")

    print("[train] Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=200,       # 200 trees — good accuracy/speed balance
        max_depth=None,         # let trees grow fully
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',    # standard for classification
        class_weight='balanced',
        random_state=42,
        n_jobs=-1               # use all CPU cores
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n[train] Test Accuracy: {acc * 100:.2f}%")

    # Top-3 accuracy
    proba   = clf.predict_proba(X_test)
    top3    = np.argsort(proba, axis=1)[:, -3:]
    top3_acc = np.mean([y_test[i] in top3[i] for i in range(len(y_test))])
    print(f"[train] Top-3 Accuracy: {top3_acc * 100:.2f}%")

    return clf, le


def augment(X, y, copies=8):
    """
    Simulate real patient input by randomly dropping some symptoms.
    This makes the model robust to partial symptom entry.
    """
    rng = np.random.default_rng(42)
    Xs  = [X]
    ys  = [y]
    for _ in range(copies):
        # Randomly drop 0–3 symptoms from each disease row
        mask  = rng.random(X.shape) > rng.uniform(0, 0.4, X.shape)
        X_aug = (X * mask).astype(np.int8)
        Xs.append(X_aug)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


# ── 4. Save artefacts ─────────────────────────────────────────

def save(clf, le, symptom_index, disease_dept):
    os.makedirs(ML_DIR, exist_ok=True)

    joblib.dump(clf,           os.path.join(ML_DIR, 'model.pkl'))
    joblib.dump(symptom_index, os.path.join(ML_DIR, 'symptom_index.pkl'))
    joblib.dump(le,            os.path.join(ML_DIR, 'label_encoder.pkl'))
    joblib.dump(disease_dept,  os.path.join(ML_DIR, 'disease_dept.pkl'))

    print(f"\n[train] Saved to {ML_DIR}/")
    print("  model.pkl")
    print("  symptom_index.pkl")
    print("  label_encoder.pkl")
    print("  disease_dept.pkl")


# ── Main ──────────────────────────────────────────────────────

if __name__ == '__main__':
    symptoms, diseases, mappings = load_data()
    X, y, symptom_index, disease_dept = build_matrix(symptoms, diseases, mappings)
    clf, le = train(X, y)
    save(clf, le, symptom_index, disease_dept)
    print("\n[train] Phase 1 ML training complete!")
