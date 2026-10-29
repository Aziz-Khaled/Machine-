from pathlib import Path
import html
import re
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
LOS_MODEL_PATH = BASE_DIR / "modele_duree_sejour.joblib"      # page "Durée de séjour"
CHANGE_MODEL_PATH = BASE_DIR / "bo4_change_model.joblib"      # page "Médicaments status"
INSULIN_MODEL_PATH = BASE_DIR / "bo3_insulin_model.joblib"    # page "Traitement insuline"
READMIT_MODEL_PATH = BASE_DIR / "diabetic_readmission_pipeline.joblib"  # page "Risque de réadmission"

app = FastAPI(
    title="Hospital Stay Predictor",
    description="Prédictions expérimentales : durée d'hospitalisation, changement de traitement, insuline et réadmission"
)


# =========================================================
# MODÈLE 1 : DURÉE DE SÉJOUR (inchangé)
# =========================================================

LOS_MODEL = None
LOS_FEATURES = []
LOS_ERROR = ""

try:
    if not LOS_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {LOS_MODEL_PATH.name} "
            "(place-le dans le dossier de app.py)."
        )

    saved = joblib.load(LOS_MODEL_PATH)

    if isinstance(saved, dict):
        LOS_MODEL = saved.get("modele")
        feats = saved.get("features")
    else:
        LOS_MODEL = saved
        feats = None

    if LOS_MODEL is None:
        raise ValueError("Le modèle 'modele' est absent du fichier joblib.")

    if not feats:
        feats = getattr(LOS_MODEL, "feature_names_in_", None)

    if feats is None:
        raise ValueError("Impossible de récupérer les features du modèle.")

    LOS_FEATURES = list(feats)

except Exception as exc:
    LOS_MODEL = None
    LOS_ERROR = str(exc)
    print(f"[Durée de séjour] modèle non chargé : {exc}")


NUMERIC_FIELDS = {
    "num_medications": 10,
    "num_lab_procedures": 40,
    "num_procedures": 1,
    "number_diagnoses": 5,
    "number_inpatient": 0,
}

ID_FIELDS = {
    "admission_type_id",
    "admission_source_id",
    "discharge_disposition_id",
}

LABELS = {
    "num_medications": "Nombre de médicaments",
    "num_lab_procedures": "Nombre d'analyses de laboratoire",
    "num_procedures": "Nombre de procédures",
    "number_diagnoses": "Nombre de diagnostics",
    "number_inpatient": "Hospitalisations précédentes",
    "diag_1": "Diagnostic principal",
    "diag_2": "Diagnostic secondaire",
    "admission_type_id": "Type d'admission",
    "admission_source_id": "Origine de l'admission",
    "discharge_disposition_id": "Destination de sortie",
    "age": "Tranche d'âge",
    "medical_specialty": "Spécialité médicale",
    "max_glu_serum": "Résultat maximal de glycémie",
}

AGE_BINS = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
]

DEFAULT_CATEGORIES = {
    "age": AGE_BINS,
    "max_glu_serum": ["None", "Norm", ">200", ">300"],
}


def extract_categories(model, nested=False):
    """Lit les catégories apprises par le pipeline (pour fabriquer les listes).

    nested=False : comportement d'origine (transformeur 'cat' = encodeur direct)
    nested=True  : accepte aussi un sous-Pipeline (imputer -> OneHotEncoder)
    """
    categories = {}

    if model is None or not hasattr(model, "named_steps"):
        return categories

    prep = model.named_steps.get("prep") or model.named_steps.get("preprocessor")
    if prep is None or not hasattr(prep, "transformers_"):
        return categories

    for name, transformer, columns in prep.transformers_:
        if name != "cat" or isinstance(transformer, str):
            continue

        encoder = transformer
        if nested and hasattr(transformer, "named_steps"):
            encoder = list(transformer.named_steps.values())[-1]

        if not hasattr(encoder, "categories_"):
            continue

        try:
            for column, values in zip(columns, encoder.categories_):
                categories[str(column)] = [str(v) for v in values]
        except (TypeError, ValueError):
            pass

    return categories


LOS_CATEGORIES = extract_categories(LOS_MODEL, nested=False)


def los_categories(feature):
    return LOS_CATEGORIES.get(feature) or DEFAULT_CATEGORIES.get(feature, [])


def make_los_field(feature):
    label = LABELS.get(feature, feature.replace("_", " ").capitalize())
    safe_feature = html.escape(feature, quote=True)
    safe_label = html.escape(label)
    categories = los_categories(feature)

    if categories:
        options = "".join(
            f'<option value="{html.escape(v, quote=True)}">{html.escape(v)}</option>'
            for v in categories
        )
        control = (
            f'<select id="{safe_feature}" name="{safe_feature}" required>'
            f'<option value="">Choisir...</option>{options}</select>'
        )
    elif feature in NUMERIC_FIELDS:
        control = (
            f'<input type="number" id="{safe_feature}" name="{safe_feature}" '
            f'value="{NUMERIC_FIELDS[feature]}" min="0" step="1" required>'
        )
    elif feature in ID_FIELDS:
        control = (
            f'<input type="number" id="{safe_feature}" name="{safe_feature}" '
            f'min="1" step="1" value="1" required>'
        )
    else:
        control = (
            f'<input type="text" id="{safe_feature}" name="{safe_feature}" '
            f'placeholder="Saisir une valeur" required>'
        )

    return (
        '<div class="field">'
        f'<label for="{safe_feature}">{safe_label}</label>{control}</div>'
    )


# =========================================================
# MODÈLE 2 : CHANGEMENT DE TRAITEMENT (notebook BO4)
# =========================================================

CHANGE_MODEL = None
CHANGE_FEATURES = []
CHANGE_THRESHOLD = 0.5
CHANGE_ERROR = ""

try:
    if not CHANGE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {CHANGE_MODEL_PATH.name} "
            "(télécharge-le depuis Colab et place-le à côté de app.py)."
        )

    saved2 = joblib.load(CHANGE_MODEL_PATH)

    if not isinstance(saved2, dict) or "model" not in saved2:
        raise ValueError(
            "Format inattendu : le fichier doit contenir "
            "{'model', 'threshold', 'features'}."
        )

    CHANGE_MODEL = saved2["model"]
    CHANGE_THRESHOLD = float(saved2.get("threshold", 0.5))
    CHANGE_FEATURES = list(saved2.get("features") or [])

    if not CHANGE_FEATURES:
        CHANGE_FEATURES = list(getattr(CHANGE_MODEL, "feature_names_in_", []))

    if not CHANGE_FEATURES:
        raise ValueError("Impossible de récupérer les features du modèle.")

except Exception as exc:
    CHANGE_MODEL = None
    CHANGE_ERROR = str(exc)
    print(f"[Médicaments status] modèle non chargé : {exc}")


CHANGE_CATEGORIES = extract_categories(CHANGE_MODEL, nested=True)

# Champs du formulaire : (nom, libellé, type, valeur par défaut)
#   num = nombre entier >= 0 | cat = liste lue dans le modèle | age = tranche d'âge
CHANGE_FIELDS = [
    ("age", "Tranche d'âge", "age", None),
    ("gender", "Sexe", "cat", None),
    ("race", "Origine ethnique", "cat", None),
    ("time_in_hospital", "Durée du séjour (jours)", "num", 4),
    ("num_lab_procedures", "Nombre d'analyses de laboratoire", "num", 40),
    ("num_procedures", "Nombre de procédures", "num", 1),
    ("num_medications", "Nombre de médicaments", "num", 15),
    ("number_diagnoses", "Nombre de diagnostics", "num", 7),
    ("number_outpatient", "Visites ambulatoires (année passée)", "num", 0),
    ("number_emergency", "Passages aux urgences (année passée)", "num", 0),
    ("number_inpatient", "Hospitalisations précédentes", "num", 0),
    ("admission_type", "Type d'admission", "cat", None),
    ("admission_source", "Origine de l'admission", "cat", None),
    ("medical_specialty", "Spécialité médicale", "cat", None),
    ("diag_1_group", "Diagnostic principal (groupe)", "cat", None),
    ("diag_2_group", "Diagnostic secondaire (groupe)", "cat", None),
    ("diag_3_group", "Diagnostic additionnel (groupe)", "cat", None),
    ("A1Cresult", "Résultat HbA1c", "cat", None),
    ("max_glu_serum", "Résultat maximal de glycémie", "cat", None),
]

CHANGE_NUM_RAW = [n for n, _, k, _ in CHANGE_FIELDS if k == "num"]
CHANGE_CAT_RAW = [n for n, _, k, _ in CHANGE_FIELDS if k == "cat"]

CHANGE_DEFAULT_CATEGORIES = {
    "gender": ["Female", "Male"],
    "A1Cresult": ["None", "Norm", ">7", ">8"],
    "max_glu_serum": ["None", "Norm", ">200", ">300"],
}


def make_change_field(name, label, kind, default):
    safe_name = html.escape(name, quote=True)
    safe_label = html.escape(label)

    if kind == "age":
        values = AGE_BINS
    elif kind == "cat":
        values = CHANGE_CATEGORIES.get(name) or CHANGE_DEFAULT_CATEGORIES.get(name, [])
    else:
        values = None

    if kind == "num":
        control = (
            f'<input type="number" id="{safe_name}" name="{safe_name}" '
            f'value="{default}" min="0" step="1" required>'
        )
    elif values:
        options = "".join(
            f'<option value="{html.escape(v, quote=True)}">{html.escape(v)}</option>'
            for v in values
        )
        control = (
            f'<select id="{safe_name}" name="{safe_name}" required>'
            f'<option value="">Choisir...</option>{options}</select>'
        )
    else:
        control = (
            f'<input type="text" id="{safe_name}" name="{safe_name}" '
            f'placeholder="Saisir une valeur" required>'
        )

    return (
        '<div class="field">'
        f'<label for="{safe_name}">{safe_label}</label>{control}</div>'
    )


def age_to_num(age_bin):
    """'[60-70)' -> 65.0 (même calcul que le notebook : milieu de la tranche)."""
    m = re.fullmatch(r"\[(\d+)-(\d+)\)", age_bin.strip())
    if not m:
        raise ValueError(f"tranche d'âge invalide : {age_bin}")
    return (float(m.group(1)) + float(m.group(2))) / 2


# =========================================================
# MODÈLE 3 : TRAITEMENT INSULINE (notebook BO3)
# =========================================================

INSULIN_MODEL = None
INSULIN_FEATURES = []
INSULIN_ERROR = ""

# Correspondance classe numérique -> libellé.
# Encodage du notebook BO3 (section 3.3) : TARGET_MAP = {"No": 0, "Steady": 1, "Down": 2, "Up": 3}
INSULIN_CLASS_LABELS = {
    0: "No — pas d'insuline",
    1: "Steady — dose stable",
    2: "Down — dose diminuée",
    3: "Up — dose augmentée",
}

try:
    if not INSULIN_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {INSULIN_MODEL_PATH.name} "
            "(place-le dans le dossier de app.py)."
        )

    saved3 = joblib.load(INSULIN_MODEL_PATH)

    # Accepte un Pipeline nu ou un dict {'model', 'features', 'classes'}
    if isinstance(saved3, dict):
        INSULIN_MODEL = saved3.get("model") or saved3.get("modele")
        feats3 = saved3.get("features")
        labels3 = saved3.get("classes") or saved3.get("labels")
        if isinstance(labels3, dict):
            INSULIN_CLASS_LABELS = {int(k): str(v) for k, v in labels3.items()}
        elif labels3:
            INSULIN_CLASS_LABELS = {i: str(v) for i, v in enumerate(labels3)}
    else:
        INSULIN_MODEL = saved3
        feats3 = None

    if INSULIN_MODEL is None:
        raise ValueError("Aucun modèle trouvé dans le fichier joblib.")

    if not feats3:
        feats3 = getattr(INSULIN_MODEL, "feature_names_in_", None)
    if feats3 is None:
        raise ValueError("Impossible de récupérer les features du modèle.")

    INSULIN_FEATURES = list(feats3)

except Exception as exc:
    INSULIN_MODEL = None
    INSULIN_ERROR = str(exc)
    print(f"[Traitement insuline] modèle non chargé : {exc}")


INSULIN_CATEGORIES = extract_categories(INSULIN_MODEL, nested=True)

# (nom, libellé, type, valeur par défaut)
#   num = entier >= 0 | cat = liste lue dans le modèle | age = tranche d'âge
INSULIN_FIELDS = [
    ("age", "Tranche d'âge", "age", None),
    ("time_in_hospital", "Durée du séjour (jours)", "num", 4),
    ("num_lab_procedures", "Nombre d'analyses de laboratoire", "num", 40),
    ("num_procedures", "Nombre de procédures", "num", 1),
    ("num_medications", "Nombre de médicaments", "num", 15),
    ("number_diagnoses", "Nombre de diagnostics", "num", 7),
    ("prior_visits", "Visites antérieures (ambu. + urgences + hospit.)", "num", 0),
    ("n_other_diab_meds", "Autres antidiabétiques prescrits", "num", 0),
    ("diab_complication", "Complication diabétique (0 = non, 1 = oui)", "num", 0),
    ("A1Cresult", "Résultat HbA1c", "cat", None),
    ("max_glu_serum", "Résultat maximal de glycémie", "cat", None),
    ("diag_1_grp", "Diagnostic principal (groupe)", "cat", None),
    ("medical_specialty", "Spécialité médicale", "cat", None),
    ("admission_type", "Type d'admission", "cat", None),
    ("admission_source", "Origine de l'admission", "cat", None),
    ("discharge", "Destination de sortie", "cat", None),
]

INSULIN_NUM_RAW = [n for n, _, k, _ in INSULIN_FIELDS if k == "num"]
INSULIN_CAT_RAW = [n for n, _, k, _ in INSULIN_FIELDS if k == "cat"]

INSULIN_DEFAULT_CATEGORIES = {
    "A1Cresult": ["None", "Norm", ">7", ">8"],
    "max_glu_serum": ["None", "Norm", ">200", ">300"],
}


def make_insulin_field(name, label, kind, default):
    safe_name = html.escape(name, quote=True)
    safe_label = html.escape(label)

    if kind == "age":
        values = AGE_BINS
    elif kind == "cat":
        values = INSULIN_CATEGORIES.get(name) or INSULIN_DEFAULT_CATEGORIES.get(name, [])
    else:
        values = None

    if kind == "num":
        control = (
            f'<input type="number" id="{safe_name}" name="{safe_name}" '
            f'value="{default}" min="0" step="1" required>'
        )
    elif values:
        options = "".join(
            f'<option value="{html.escape(v, quote=True)}">{html.escape(v)}</option>'
            for v in values
        )
        control = (
            f'<select id="{safe_name}" name="{safe_name}" required>'
            f'<option value="">Choisir...</option>{options}</select>'
        )
    else:
        control = (
            f'<input type="text" id="{safe_name}" name="{safe_name}" '
            f'placeholder="Saisir une valeur" required>'
        )

    return (
        '<div class="field">'
        f'<label for="{safe_name}">{safe_label}</label>{control}</div>'
    )



# =========================================================
# MODÈLE 4 : RISQUE DE RÉADMISSION (notebook BO1)
# =========================================================

READMIT_MODEL = None
READMIT_FEATURES = []
READMIT_THRESHOLD = 0.5
READMIT_METRICS = {}
READMIT_ERROR = ""

try:
    if not READMIT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {READMIT_MODEL_PATH.name} "
            "(place-le dans le dossier de app.py)."
        )

    saved4 = joblib.load(READMIT_MODEL_PATH)

    # Format du notebook BO1 : {'pipeline', 'optimal_threshold', 'features_list', 'metrics'}
    if isinstance(saved4, dict):
        READMIT_MODEL = saved4.get("pipeline") or saved4.get("model")
        READMIT_THRESHOLD = float(saved4.get("optimal_threshold", saved4.get("threshold", 0.5)))
        READMIT_FEATURES = list(saved4.get("features_list") or saved4.get("features") or [])
        READMIT_METRICS = {k: float(v) for k, v in (saved4.get("metrics") or {}).items()}
    else:
        READMIT_MODEL = saved4

    if READMIT_MODEL is None:
        raise ValueError("Aucun pipeline trouvé dans le fichier joblib.")

    if not READMIT_FEATURES:
        READMIT_FEATURES = list(getattr(READMIT_MODEL, "feature_names_in_", []))
    if not READMIT_FEATURES:
        raise ValueError("Impossible de récupérer les features du modèle.")

except Exception as exc:
    READMIT_MODEL = None
    READMIT_ERROR = str(exc)
    print(f"[Risque de réadmission] modèle non chargé : {exc}")


READMIT_CATEGORIES = extract_categories(READMIT_MODEL, nested=True)

# (nom, libellé, type, valeur par défaut)
READMIT_FIELDS = [
    ("age", "Tranche d'âge", "age", None),
    ("time_in_hospital", "Durée du séjour (jours)", "num", 4),
    ("num_medications", "Nombre de médicaments", "num", 15),
    ("num_procedures", "Nombre de procédures", "num", 1),
    ("number_diagnoses", "Nombre de diagnostics", "num", 7),
    ("number_outpatient", "Visites ambulatoires (année passée)", "num", 0),
    ("number_emergency", "Passages aux urgences (année passée)", "num", 0),
    ("number_inpatient", "Hospitalisations précédentes (année passée)", "num", 0),
    # ⚠ À vérifier avec le notebook BO1 : définition exacte de ces deux variables
    ("n_sejours", "Nombre de séjours du patient", "num", 1),
    ("n_meds_used", "Nombre d'antidiabétiques utilisés", "num", 1),
    ("insulin", "Insuline pendant le séjour", "cat", None),
    ("change", "Changement de traitement (Ch = oui)", "cat", None),
]

READMIT_NUM_RAW = [n for n, _, k, _ in READMIT_FIELDS if k == "num"]
READMIT_CAT_RAW = [n for n, _, k, _ in READMIT_FIELDS if k == "cat"]

READMIT_DEFAULT_CATEGORIES = {
    "insulin": ["Down", "No", "Steady", "Up"],
    "change": ["Ch", "No"],
}


def make_readmit_field(name, label, kind, default):
    safe_name = html.escape(name, quote=True)
    safe_label = html.escape(label)

    if kind == "age":
        values = AGE_BINS
    elif kind == "cat":
        values = READMIT_CATEGORIES.get(name) or READMIT_DEFAULT_CATEGORIES.get(name, [])
    else:
        values = None

    if kind == "num":
        minimum = 1 if name == "n_sejours" else 0
        control = (
            f'<input type="number" id="{safe_name}" name="{safe_name}" '
            f'value="{default}" min="{minimum}" step="1" required>'
        )
    elif values:
        options = "".join(
            f'<option value="{html.escape(v, quote=True)}">{html.escape(v)}</option>'
            for v in values
        )
        control = (
            f'<select id="{safe_name}" name="{safe_name}" required>'
            f'<option value="">Choisir...</option>{options}</select>'
        )
    else:
        control = (
            f'<input type="text" id="{safe_name}" name="{safe_name}" '
            f'placeholder="Saisir une valeur" required>'
        )

    return (
        '<div class="field">'
        f'<label for="{safe_name}">{safe_label}</label>{control}</div>'
    )


def readmit_metrics_text():
    if not READMIT_METRICS:
        return "Probabilité prédite par le modèle"
    names = {"roc_auc": "AUC", "f1_score": "F1", "recall": "Rappel", "precision": "Précision"}
    parts = [f"{names.get(k, k)} {v:.2f}" for k, v in READMIT_METRICS.items()]
    return "Probabilité prédite par le modèle · test : " + " · ".join(parts)


# =========================================================
# PAGES HTML
# =========================================================

CSS = """
:root {
    --navy: #10243e;
    --blue: #2864dc;
    --bg: #f2f5fa;
    --text: #1c2b40;
    --muted: #68788d;
    --border: #e0e7f0;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; }
body { font-family: Inter, "Segoe UI", Arial, sans-serif; background: var(--bg); color: var(--text); }

.app { height: 100dvh; width: 100%; display: grid; grid-template-columns: 220px minmax(0, 1fr); overflow: hidden; }

.sidebar { background: var(--navy); color: white; padding: 26px 18px; display: flex; flex-direction: column; gap: 24px; min-width: 0; overflow: hidden; }
.brand { font-size: 18px; font-weight: 800; line-height: 1.4; }
.brand span { display: block; color: #9dbcfb; font-size: 12px; font-weight: 500; margin-top: 4px; }

.nav { display: flex; flex-direction: column; gap: 8px; }
.nav-item { display: block; padding: 12px; border-radius: 9px; background: transparent; color: #c5d3e5; font-size: 13px; text-decoration: none; transition: background .15s; }
.nav-item:hover { background: #1a3150; color: white; }
.nav-item.active { background: #203b5c; color: white; }

.sidebar-note { margin-top: auto; color: #c5d3e5; font-size: 11px; line-height: 1.6; }

.main { min-width: 0; min-height: 0; display: flex; flex-direction: column; padding: 22px 28px; gap: 16px; overflow: hidden; }
.topbar { flex: 0 0 auto; display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.topbar h1 { font-size: clamp(18px, 2vw, 25px); margin: 0; }
.status { font-size: 11px; color: #147348; background: #e1f5e9; border-radius: 20px; padding: 8px 12px; white-space: nowrap; }
.status.off { color: #b42318; background: #fde8e6; }

.intro { flex: 0 0 auto; color: var(--muted); font-size: 13px; line-height: 1.5; margin: 0; }
.alert { flex: 0 0 auto; background: #fff4e5; border: 1px solid #ffd9a8; color: #8a4b00; border-radius: 10px; padding: 10px 14px; font-size: 12px; line-height: 1.5; overflow-wrap: anywhere; }

.workspace { flex: 1 1 auto; min-height: 0; display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(220px, 0.8fr); gap: 18px; overflow: hidden; }
.panel { min-width: 0; min-height: 0; background: white; border: 1px solid var(--border); border-radius: 15px; padding: 20px; box-shadow: 0 4px 16px rgba(20, 40, 70, 0.04); overflow: hidden; }
.form-panel { display: flex; flex-direction: column; }
.panel h2 { margin: 0 0 6px; font-size: 17px; }
.panel-desc { color: var(--muted); font-size: 12px; line-height: 1.5; margin: 0 0 15px; }

.form-fields { flex: 1 1 auto; min-height: 0; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; align-content: start; overflow: auto; padding: 2px 5px 8px 0; }
.field { min-width: 0; }
.field label { display: block; font-size: 11px; font-weight: 700; margin-bottom: 6px; color: #42536a; }
.field input, .field select { width: 100%; min-width: 0; height: 38px; padding: 0 10px; border: 1px solid #dce4ef; border-radius: 8px; background: #fbfcfe; color: var(--text); font-size: 12px; outline: none; }
.field input:focus, .field select:focus { border-color: var(--blue); box-shadow: 0 0 0 3px rgba(40, 100, 220, 0.1); }

.actions { flex: 0 0 auto; padding-top: 12px; display: flex; align-items: center; gap: 10px; border-top: 1px solid #eef1f6; margin-top: 10px; }
button { min-height: 40px; padding: 0 16px; border: 0; border-radius: 8px; background: var(--blue); color: white; font-size: 12px; font-weight: 700; cursor: pointer; }
button:disabled { opacity: 0.65; cursor: wait; }
.reset { color: #52647a; background: #eef2f7; }

.result-panel { display: flex; flex-direction: column; justify-content: flex-start; overflow: auto; }
.result-card { background: #f2f6ff; border: 1px solid #e0e9ff; border-radius: 12px; padding: 18px; margin-top: 8px; }
.result-label { color: #526a8d; font-size: 12px; }
.result-number { font-size: clamp(30px, 4vw, 44px); font-weight: 800; color: var(--blue); margin: 12px 0 4px; }
.result-number small { font-size: 14px; font-weight: 600; }
.result-hint { color: var(--muted); font-size: 12px; line-height: 1.6; }

.verdict { display: inline-block; margin-top: 10px; padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: 700; background: #eef2f7; color: #52647a; }
.verdict.yes { background: #fff0dc; color: #a35200; }
.verdict.no { background: #e1f5e9; color: #147348; }

.meter { position: relative; height: 10px; background: #dfe7f5; border-radius: 6px; margin: 14px 0 6px; overflow: visible; }
.meter-fill { height: 100%; width: 0; background: var(--blue); border-radius: 6px; transition: width .35s; }
.meter-thr { position: absolute; top: -4px; width: 2px; height: 18px; background: #10243e; display: none; }
.meter-legend { display: flex; justify-content: space-between; font-size: 10px; color: var(--muted); }

.class-bars { margin-top: 14px; display: flex; flex-direction: column; gap: 8px; }
.class-row { display: grid; grid-template-columns: minmax(0, 1fr) 46px; gap: 4px 8px; align-items: center; font-size: 11px; color: #42536a; }
.class-row .bar { grid-column: 1 / -1; height: 6px; background: #dfe7f5; border-radius: 4px; overflow: hidden; }
.class-row .bar div { height: 100%; width: 0; background: #9dbcfb; border-radius: 4px; transition: width .35s; }
.class-row.top { color: var(--text); font-weight: 700; }
.class-row.top .bar div { background: var(--blue); }
.class-row .pct { text-align: right; }

.message { margin-top: 12px; font-size: 12px; line-height: 1.5; overflow-wrap: anywhere; }
.message.error { color: #b42318; }
.message.success { color: #147348; }
.footer { flex: 0 0 auto; color: #7b899b; font-size: 10px; line-height: 1.4; }

@media (max-width: 850px) {
    .app { grid-template-columns: 155px minmax(0, 1fr); }
    .main { padding: 16px; }
    .workspace { grid-template-columns: minmax(0, 1fr); grid-template-rows: minmax(0, 1fr) auto; }
    .result-panel { padding: 12px 16px; }
    .result-card { padding: 10px 14px; margin-top: 5px; }
    .result-number { margin: 4px 0; font-size: 28px; }
    .result-hint { display: none; }
}
@media (max-width: 560px) {
    .app { grid-template-columns: 1fr; }
    .sidebar { display: none; }
    .main { padding: 12px; gap: 10px; }
    .form-fields { grid-template-columns: 1fr; }
    .status { font-size: 10px; padding: 6px 8px; }
    .panel { padding: 14px; }
}
@media (max-height: 650px) {
    .main { padding-top: 10px; padding-bottom: 10px; gap: 8px; }
    .panel { padding: 12px; }
    .panel-desc { margin-bottom: 8px; }
    .form-fields { gap: 8px; }
    .field input, .field select { height: 32px; }
    .actions { padding-top: 8px; margin-top: 6px; }
}
"""

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>__CSS__</style>
</head>
<body>
<div class="app">

    <aside class="sidebar">
        <div class="brand">
            Hospital AI
            <span>Predictive Analytics</span>
        </div>

        <nav class="nav">
            <a class="nav-item __ACTIVE_LOS__" href="/">Durée de séjour</a>
            <a class="nav-item __ACTIVE_CHANGE__" href="/medicaments">Médicaments status</a>
            <a class="nav-item __ACTIVE_INSULIN__" href="/insuline">Traitement insuline</a>
            <a class="nav-item __ACTIVE_READMIT__" href="/readmission">Risque de réadmission</a>
        </nav>

        <div class="sidebar-note">
            Outil expérimental de prédiction par apprentissage automatique.
        </div>
    </aside>

    <main class="main">

        <header class="topbar">
            <h1>__H1__</h1>
            <div class="status __STATUS_CLASS__">__STATUS_TEXT__</div>
        </header>

        <p class="intro">__INTRO__</p>
        __ALERT__

        <section class="workspace">

            <div class="panel form-panel">
                <h2>Informations du patient</h2>
                <p class="panel-desc">
                    Complétez les champs ci-dessous puis lancez la prédiction.
                </p>

                <form id="predictionForm"
                      style="display:flex;flex:1;min-height:0;flex-direction:column;">

                    <div class="form-fields">
                        __FIELDS__
                    </div>

                    <div class="actions">
                        <button type="submit" id="predictButton">
                            Calculer la prédiction
                        </button>
                        <button type="reset" class="reset">
                            Réinitialiser
                        </button>
                    </div>

                </form>
            </div>

            <div class="panel result-panel">
                <h2>Résultat estimé</h2>
                <p class="panel-desc">__RESULT_DESC__</p>

                __RESULT_CARD__

                <div id="message" class="message" role="status"></div>

                <p class="result-hint" style="margin-top:18px;">
                    Cette estimation est indicative et ne remplace pas
                    l'évaluation d'un professionnel de santé.
                </p>
            </div>

        </section>

        <footer class="footer">
            Projet académique de Machine Learning · Résultats expérimentaux
        </footer>

    </main>
</div>

<script>
const form = document.getElementById("predictionForm");
const button = document.getElementById("predictButton");
const message = document.getElementById("message");

async function callApi(endpoint) {
    const data = {};

    for (const element of form.elements) {
        if (!element.name) continue;

        if (element.value.trim() === "") {
            throw new Error("Veuillez remplir tous les champs du formulaire.");
        }
        data[element.name] = element.value;
    }

    const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    if (!response.ok) {
        const detail = typeof result.detail === "string"
            ? result.detail
            : "Erreur de prédiction.";
        throw new Error(detail);
    }
    return result;
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    button.disabled = true;
    button.textContent = "Calcul en cours...";
    message.textContent = "";
    message.className = "message";

    try {
        const result = await callApi(ENDPOINT);
        showResult(result);
        message.textContent = "Prédiction calculée avec succès.";
        message.className = "message success";
    } catch (error) {
        message.textContent = error.message;
        message.className = "message error";
    } finally {
        button.disabled = false;
        button.textContent = "Calculer la prédiction";
    }
});

form.addEventListener("reset", () => {
    resetResult();
    message.textContent = "";
    message.className = "message";
});

__PAGE_SCRIPT__
</script>
</body>
</html>
"""

LOS_RESULT_CARD = """
<div class="result-card">
    <div class="result-label">Durée d'hospitalisation</div>
    <div class="result-number" id="resultNumber">-- <small>jours</small></div>
    <div class="result-hint" id="resultHint">
        Le résultat apparaîtra après validation du formulaire.
    </div>
</div>
"""

LOS_SCRIPT = """
const ENDPOINT = "/predict";
const resultNumber = document.getElementById("resultNumber");
const resultHint = document.getElementById("resultHint");

function showResult(result) {
    resultNumber.innerHTML = `${result.duree_arrondie} <small>jours</small>`;
    resultHint.textContent = "Estimation calculée à partir des informations saisies.";
}

function resetResult() {
    resultNumber.innerHTML = '-- <small>jours</small>';
    resultHint.textContent = "Le résultat apparaîtra après validation du formulaire.";
}
"""

CHANGE_RESULT_CARD = """
<div class="result-card">
    <div class="result-label">Probabilité de changement de traitement</div>
    <div class="result-number" id="resultNumber">-- <small>%</small></div>
    <div class="meter">
        <div class="meter-fill" id="meterFill"></div>
        <div class="meter-thr" id="meterThr" title="Seuil de décision"></div>
    </div>
    <div class="meter-legend"><span>0 %</span><span>100 %</span></div>
    <div><span class="verdict" id="verdict">En attente</span></div>
    <div class="result-hint" id="resultHint" style="margin-top:10px;">
        Le résultat apparaîtra après validation du formulaire.
    </div>
</div>
"""

CHANGE_SCRIPT = """
const ENDPOINT = "/predict-change";
const resultNumber = document.getElementById("resultNumber");
const resultHint = document.getElementById("resultHint");
const verdict = document.getElementById("verdict");
const meterFill = document.getElementById("meterFill");
const meterThr = document.getElementById("meterThr");

function showResult(r) {
    resultNumber.innerHTML = `${r.pourcentage} <small>%</small>`;
    meterFill.style.width = `${Math.min(100, r.pourcentage)}%`;
    meterThr.style.left = `${r.seuil * 100}%`;
    meterThr.style.display = "block";

    if (r.changement) {
        verdict.textContent = "Changement de traitement probable";
        verdict.className = "verdict yes";
    } else {
        verdict.textContent = "Changement peu probable";
        verdict.className = "verdict no";
    }

    resultHint.textContent =
        `Seuil de décision : ${(r.seuil * 100).toFixed(1)} % ` +
        `(trait noir, choisi pour maximiser le F1 sur l'entraînement).`;
}

function resetResult() {
    resultNumber.innerHTML = '-- <small>%</small>';
    meterFill.style.width = "0";
    meterThr.style.display = "none";
    verdict.textContent = "En attente";
    verdict.className = "verdict";
    resultHint.textContent = "Le résultat apparaîtra après validation du formulaire.";
}
"""


INSULIN_RESULT_CARD = """
<div class="result-card">
    <div class="result-label">Ajustement d'insuline prédit</div>
    <div class="result-number" id="resultNumber" style="font-size:clamp(22px,2.6vw,30px);">--</div>
    <div class="result-hint" id="resultConf">Confiance : -- %</div>
    <div class="class-bars" id="classBars"></div>
    <div class="result-hint" id="resultHint" style="margin-top:10px;">
        Le résultat apparaîtra après validation du formulaire.
    </div>
</div>
"""

INSULIN_SCRIPT = """
const ENDPOINT = "/predict-insulin";
const resultNumber = document.getElementById("resultNumber");
const resultConf = document.getElementById("resultConf");
const resultHint = document.getElementById("resultHint");
const classBars = document.getElementById("classBars");

function esc(s) {
    return String(s).replace(/[&<>"']/g, c => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[c]));
}

function showResult(r) {
    resultNumber.textContent = r.label;
    resultConf.textContent = `Confiance : ${r.confiance} %`;
    classBars.innerHTML = r.probabilites.map(p => `
        <div class="class-row ${p.classe === r.classe ? "top" : ""}">
            <span>${esc(p.label)}</span><span class="pct">${p.pourcentage} %</span>
            <div class="bar"><div style="width:${Math.min(100, p.pourcentage)}%"></div></div>
        </div>`).join("");
    resultHint.textContent = "Classe la plus probable selon la Random Forest.";
}

function resetResult() {
    resultNumber.textContent = "--";
    resultConf.textContent = "Confiance : -- %";
    classBars.innerHTML = "";
    resultHint.textContent = "Le résultat apparaîtra après validation du formulaire.";
}
"""


READMIT_RESULT_CARD = """
<div class="result-card">
    <div class="result-label">Probabilité de réadmission</div>
    <div class="result-number" id="resultNumber">-- <small>%</small></div>
    <div class="meter">
        <div class="meter-fill" id="meterFill"></div>
        <div class="meter-thr" id="meterThr" title="Seuil de décision"></div>
    </div>
    <div class="meter-legend"><span>0 %</span><span>100 %</span></div>
    <div><span class="verdict" id="verdict">En attente</span></div>
    <div class="result-hint" id="resultHint" style="margin-top:10px;">
        Le résultat apparaîtra après validation du formulaire.
    </div>
</div>
"""

READMIT_SCRIPT = """
const ENDPOINT = "/predict-readmission";
const resultNumber = document.getElementById("resultNumber");
const resultHint = document.getElementById("resultHint");
const verdict = document.getElementById("verdict");
const meterFill = document.getElementById("meterFill");
const meterThr = document.getElementById("meterThr");

function showResult(r) {
    resultNumber.innerHTML = `${r.pourcentage} <small>%</small>`;
    meterFill.style.width = `${Math.min(100, r.pourcentage)}%`;
    meterThr.style.left = `${r.seuil * 100}%`;
    meterThr.style.display = "block";

    if (r.readmission) {
        verdict.textContent = "Risque de réadmission élevé";
        verdict.className = "verdict yes";
    } else {
        verdict.textContent = "Risque de réadmission faible";
        verdict.className = "verdict no";
    }

    resultHint.textContent =
        `Seuil de décision : ${(r.seuil * 100).toFixed(1)} % ` +
        `(trait noir, seuil optimal enregistré avec le modèle).`;
}

function resetResult() {
    resultNumber.innerHTML = '-- <small>%</small>';
    meterFill.style.width = "0";
    meterThr.style.display = "none";
    verdict.textContent = "En attente";
    verdict.className = "verdict";
    resultHint.textContent = "Le résultat apparaîtra après validation du formulaire.";
}
"""


def render_page(*, active, title, h1, intro, fields, result_desc,
                result_card, script, model_ok, error=""):
    alert = ""
    if not model_ok:
        alert = (
            '<div class="alert"><strong>Modèle non chargé.</strong> '
            f'{html.escape(error)}</div>'
        )

    replacements = {
        "__TITLE__": title,
        "__CSS__": CSS,
        "__ACTIVE_LOS__": "active" if active == "los" else "",
        "__ACTIVE_CHANGE__": "active" if active == "change" else "",
        "__ACTIVE_INSULIN__": "active" if active == "insulin" else "",
        "__ACTIVE_READMIT__": "active" if active == "readmit" else "",
        "__H1__": h1,
        "__STATUS_CLASS__": "" if model_ok else "off",
        "__STATUS_TEXT__": "● Modèle chargé" if model_ok else "● Modèle indisponible",
        "__INTRO__": intro,
        "__ALERT__": alert,
        "__FIELDS__": fields,
        "__RESULT_DESC__": result_desc,
        "__RESULT_CARD__": result_card,
        "__PAGE_SCRIPT__": script,
    }

    page = PAGE_TEMPLATE
    for key, value in replacements.items():
        page = page.replace(key, value)
    return HTMLResponse(page)


@app.get("/", response_class=HTMLResponse)
def home():
    return render_page(
        active="los",
        title="Hospital Stay Predictor",
        h1="Prédiction hospitalière",
        intro=("Renseignez les informations du patient pour estimer la durée "
               "d'hospitalisation à partir du modèle entraîné."),
        fields="\n".join(make_los_field(f) for f in LOS_FEATURES),
        result_desc="Durée prédite par le modèle",
        result_card=LOS_RESULT_CARD,
        script=LOS_SCRIPT,
        model_ok=LOS_MODEL is not None,
        error=LOS_ERROR,
    )


@app.get("/medicaments", response_class=HTMLResponse)
def medicaments():
    return render_page(
        active="change",
        title="Médicaments status · Hospital AI",
        h1="Médicaments status",
        intro=("Estime la probabilité qu'un changement de traitement "
               "médicamenteux (dose ou molécule) soit effectué pendant le "
               "séjour, à partir du modèle Random Forest entraîné."),
        fields="\n".join(make_change_field(*f) for f in CHANGE_FIELDS),
        result_desc="Probabilité prédite par le modèle",
        result_card=CHANGE_RESULT_CARD,
        script=CHANGE_SCRIPT,
        model_ok=CHANGE_MODEL is not None,
        error=CHANGE_ERROR,
    )


@app.get("/insuline", response_class=HTMLResponse)
def insuline():
    return render_page(
        active="insulin",
        title="Traitement insuline · Hospital AI",
        h1="Traitement insuline",
        intro=("Prédit l'ajustement du traitement à l'insuline pendant le séjour "
               "(aucune, diminuée, stable ou augmentée) à partir du modèle "
               "Random Forest BO3."),
        fields="\n".join(make_insulin_field(*f) for f in INSULIN_FIELDS),
        result_desc="Classe prédite et probabilités par classe",
        result_card=INSULIN_RESULT_CARD,
        script=INSULIN_SCRIPT,
        model_ok=INSULIN_MODEL is not None,
        error=INSULIN_ERROR,
    )


@app.get("/readmission", response_class=HTMLResponse)
def readmission():
    return render_page(
        active="readmit",
        title="Risque de réadmission · Hospital AI",
        h1="Risque de réadmission",
        intro=("Estime la probabilité qu'un patient diabétique soit réadmis "
               "à l'hôpital après sa sortie, à partir du modèle Random Forest BO1."),
        fields="\n".join(make_readmit_field(*f) for f in READMIT_FIELDS),
        result_desc=html.escape(readmit_metrics_text()),
        result_card=READMIT_RESULT_CARD,
        script=READMIT_SCRIPT,
        model_ok=READMIT_MODEL is not None,
        error=READMIT_ERROR,
    )


# =========================================================
# API DE PRÉDICTION
# =========================================================

class PatientData(BaseModel):
    model_config = {"extra": "allow"}


@app.post("/predict")
def predict(patient: PatientData):
    if LOS_MODEL is None:
        raise HTTPException(status_code=503, detail=f"Modèle non chargé : {LOS_ERROR}")

    data = patient.model_dump()

    missing = [f for f in LOS_FEATURES if f not in data or data[f] is None]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Champs manquants : {', '.join(missing)}"
        )

    row = {}

    try:
        for feature in LOS_FEATURES:
            value = data[feature]

            if isinstance(value, str):
                value = value.strip()

            if feature in NUMERIC_FIELDS:
                row[feature] = float(value)
            elif feature in ID_FIELDS:
                row[feature] = int(value)
            else:
                row[feature] = str(value)

        input_df = pd.DataFrame([row], columns=LOS_FEATURES)
        prediction = float(LOS_MODEL.predict(input_df)[0])

        if not pd.notna(prediction):
            raise ValueError("Le modèle a retourné une valeur invalide.")

        return {
            "duree_jours": prediction,
            "duree_arrondie": round(prediction, 1)
        }

    except HTTPException:
        raise
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Valeur invalide : {exc}")
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la prédiction : {exc}"
        )


@app.post("/predict-change")
def predict_change(patient: PatientData):
    if CHANGE_MODEL is None:
        raise HTTPException(status_code=503, detail=f"Modèle non chargé : {CHANGE_ERROR}")

    data = patient.model_dump()

    required = ["age"] + CHANGE_NUM_RAW + CHANGE_CAT_RAW
    missing = [
        k for k in required
        if data.get(k) is None or str(data[k]).strip() == ""
    ]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Champs manquants : {', '.join(missing)}"
        )

    try:
        row = {}

        for name in CHANGE_NUM_RAW:
            value = float(data[name])
            if value < 0:
                raise ValueError(f"{name} ne peut pas être négatif")
            row[name] = value

        # Variables dérivées, calculées comme dans le notebook
        row["age_num"] = age_to_num(str(data["age"]))
        row["prior_visits_total"] = (
            row["number_outpatient"]
            + row["number_emergency"]
            + row["number_inpatient"]
        )

        for name in CHANGE_CAT_RAW:
            row[name] = str(data[name]).strip()

        absent = [f for f in CHANGE_FEATURES if f not in row]
        if absent:
            raise HTTPException(
                status_code=500,
                detail=("Le modèle attend des variables absentes du formulaire : "
                        + ", ".join(absent))
            )

        input_df = pd.DataFrame([row], columns=CHANGE_FEATURES)

        proba = float(CHANGE_MODEL.predict_proba(input_df)[0, 1])

        if not pd.notna(proba):
            raise ValueError("Le modèle a retourné une valeur invalide.")

        return {
            "probabilite": proba,
            "pourcentage": round(proba * 100, 1),
            "seuil": CHANGE_THRESHOLD,
            "changement": proba >= CHANGE_THRESHOLD,
        }

    except HTTPException:
        raise
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Valeur invalide : {exc}")
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la prédiction : {exc}"
        )

@app.post("/predict-insulin")
def predict_insulin(patient: PatientData):
    if INSULIN_MODEL is None:
        raise HTTPException(status_code=503, detail=f"Modèle non chargé : {INSULIN_ERROR}")

    data = patient.model_dump()

    required = ["age"] + INSULIN_NUM_RAW + INSULIN_CAT_RAW
    missing = [
        k for k in required
        if data.get(k) is None or str(data[k]).strip() == ""
    ]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Champs manquants : {', '.join(missing)}"
        )

    try:
        row = {}

        for name in INSULIN_NUM_RAW:
            value = float(data[name])
            if value < 0:
                raise ValueError(f"{name} ne peut pas être négatif")
            row[name] = value

        # Variables dérivées, calculées comme dans le notebook BO3
        row["age_num"] = age_to_num(str(data["age"]))
        # Notebook BO3 (section 3.2) : prior_visits passe par log1p avant l'entraînement
        row["prior_visits"] = float(np.log1p(row["prior_visits"]))

        for name in INSULIN_CAT_RAW:
            row[name] = str(data[name]).strip()

        absent = [f for f in INSULIN_FEATURES if f not in row]
        if absent:
            raise HTTPException(
                status_code=500,
                detail=("Le modèle attend des variables absentes du formulaire : "
                        + ", ".join(absent))
            )

        input_df = pd.DataFrame([row], columns=INSULIN_FEATURES)

        probas = INSULIN_MODEL.predict_proba(input_df)[0]
        classes = list(getattr(INSULIN_MODEL, "classes_", range(len(probas))))

        details = []
        for cls, p in zip(classes, probas):
            p = float(p)
            if not pd.notna(p):
                raise ValueError("Le modèle a retourné une valeur invalide.")
            key = int(cls) if str(cls).lstrip("-").isdigit() else cls
            details.append({
                "classe": str(cls),
                "label": INSULIN_CLASS_LABELS.get(key, str(cls)),
                "probabilite": p,
                "pourcentage": round(p * 100, 1),
            })

        best = max(details, key=lambda d: d["probabilite"])

        return {
            "classe": best["classe"],
            "label": best["label"],
            "confiance": best["pourcentage"],
            "probabilites": sorted(details, key=lambda d: -d["probabilite"]),
        }

    except HTTPException:
        raise
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Valeur invalide : {exc}")
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la prédiction : {exc}"
        )


@app.post("/predict-readmission")
def predict_readmission(patient: PatientData):
    if READMIT_MODEL is None:
        raise HTTPException(status_code=503, detail=f"Modèle non chargé : {READMIT_ERROR}")

    data = patient.model_dump()

    required = ["age"] + READMIT_NUM_RAW + READMIT_CAT_RAW
    missing = [
        k for k in required
        if data.get(k) is None or str(data[k]).strip() == ""
    ]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Champs manquants : {', '.join(missing)}"
        )

    try:
        row = {}

        for name in READMIT_NUM_RAW:
            value = float(data[name])
            if value < 0:
                raise ValueError(f"{name} ne peut pas être négatif")
            row[name] = value

        # Variables dérivées, calculées comme dans le notebook BO1
        row["age_num"] = age_to_num(str(data["age"]))
        row["total_prior_visits"] = (
            row["number_outpatient"]
            + row["number_emergency"]
            + row["number_inpatient"]
        )

        for name in READMIT_CAT_RAW:
            row[name] = str(data[name]).strip()

        absent = [f for f in READMIT_FEATURES if f not in row]
        if absent:
            raise HTTPException(
                status_code=500,
                detail=("Le modèle attend des variables absentes du formulaire : "
                        + ", ".join(absent))
            )

        input_df = pd.DataFrame([row], columns=READMIT_FEATURES)

        proba = float(READMIT_MODEL.predict_proba(input_df)[0, 1])

        if not pd.notna(proba):
            raise ValueError("Le modèle a retourné une valeur invalide.")

        return {
            "probabilite": proba,
            "pourcentage": round(proba * 100, 1),
            "seuil": READMIT_THRESHOLD,
            "readmission": proba >= READMIT_THRESHOLD,
        }

    except HTTPException:
        raise
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Valeur invalide : {exc}")
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la prédiction : {exc}"
        )