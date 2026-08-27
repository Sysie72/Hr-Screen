"""API FastAPI du démonstrateur HR-Screen.

L'API ne rend aucune décision RH : elle retourne une recommandation accompagnée
d'un rappel de contrôle humain obligatoire.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Any
from uuid import uuid4
import joblib
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.evaluation import comparison_report
from src.explainability import explain_advanced_prediction
from src.fairness_audit import audit_fairness
from src.job_matching import match_resume_to_job
from src.preprocessing import NUMERIC_COLUMN, TEXT_COLUMN, clean_text, load_and_split
from src.resume_extraction import MAX_FILES, SUPPORTED_EXTENSIONS, parse_uploaded_files
from src.train_advanced import MODEL_PATH as ADVANCED_PATH, train_advanced
from src.train_baseline import MODEL_PATH as BASELINE_PATH, influential_terms, train_baseline

MODELS: dict[str, Any] = {}
TEST_DATA: pd.DataFrame | None = None
JOB_STORE = Path("data/jobs.json")


def load_or_train_models() -> None:
    """Charge les artefacts; entraîne uniquement sur la partition d'entraînement si absents."""
    global TEST_DATA
    train_data, TEST_DATA = load_and_split()
    MODELS["baseline"] = joblib.load(BASELINE_PATH) if Path(BASELINE_PATH).exists() else train_baseline(train_data)
    MODELS["advanced"] = joblib.load(ADVANCED_PATH) if Path(ADVANCED_PATH).exists() else train_advanced(train_data)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_or_train_models()
    yield


app = FastAPI(title="HR-Screen API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class Candidate(BaseModel):
    resume_text: str = Field(min_length=10, max_length=20_000)
    years_experience: float = Field(ge=0, le=70)
    job_id: str


class JobCreate(BaseModel):
    title: str = Field(min_length=2, max_length=160)
    description: str = Field(min_length=10, max_length=10_000)
    required_skills: list[str] = Field(default_factory=list, max_length=30)
    preferred_skills: list[str] = Field(default_factory=list, max_length=30)
    min_years_experience: float = Field(default=0, ge=0, le=50)


def load_jobs() -> dict[str, dict[str, Any]]:
    if not JOB_STORE.exists():
        return {}
    try:
        return json.loads(JOB_STORE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_jobs(jobs: dict[str, dict[str, Any]]) -> None:
    JOB_STORE.parent.mkdir(parents=True, exist_ok=True)
    JOB_STORE.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")


def get_job_or_404(job_id: str) -> dict[str, Any]:
    job = load_jobs().get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Offre introuvable. Créez ou sélectionnez une offre avant l'analyse.")
    return job


def candidate_frame(candidate: Candidate) -> pd.DataFrame:
    return pd.DataFrame([{TEXT_COLUMN: clean_text(candidate.resume_text), NUMERIC_COLUMN: candidate.years_experience}])


def prediction(model: Any, data: pd.DataFrame) -> dict[str, Any]:
    probability = float(model.predict_proba(data)[0, 1])
    return {"decision": "Sélectionné" if probability >= 0.5 else "Rejeté", "probabilite_selection": round(probability, 4), "seuil": 0.5}


def batch_predictions(records: list[dict]) -> list[dict[str, Any]]:
    """Inférence vectorisée : une transformation par modèle pour tout le lot."""
    if not records:
        return []
    data = pd.DataFrame([
        {TEXT_COLUMN: clean_text(record["texte"]), NUMERIC_COLUMN: record["annees_experience"]}
        for record in records
    ])
    baseline_scores = MODELS["baseline"].predict_proba(data)[:, 1]
    advanced_scores = MODELS["advanced"].predict_proba(data)[:, 1]
    results = []
    for record, baseline_score, advanced_score in zip(records, baseline_scores, advanced_scores):
        advanced_probability = float(advanced_score)
        results.append({
            "fichier": record["fichier"],
            "annees_experience_extraites": record["annees_experience"],
            "competences_extraites": record["competences_extraites"],
            "baseline": {"decision": "Sélectionné" if baseline_score >= 0.5 else "Rejeté", "probabilite_selection": round(float(baseline_score), 4)},
            "advanced": {"decision": "Sélectionné" if advanced_probability >= 0.5 else "Rejeté", "probabilite_selection": round(advanced_probability, 4)},
        })
    return sorted(results, key=lambda item: item["advanced"]["probabilite_selection"], reverse=True)


def batch_job_matches(records: list[dict], job: dict[str, Any]) -> list[dict[str, Any]]:
    """Évalue chaque CV avec les critères de l'offre sélectionnée."""
    results = []
    for record in records:
        match = match_resume_to_job(record["texte"], record["annees_experience"], job)
        results.append({"fichier": record["fichier"], **match})
    return sorted(results, key=lambda item: item["score_adequation"], reverse=True)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "message": "Service de démonstration; revue humaine obligatoire."}


@app.get("/jobs")
def list_jobs() -> list[dict[str, Any]]:
    """Liste les offres créées localement par les RH."""
    return list(load_jobs().values())


@app.post("/jobs", status_code=201)
def create_job(job_input: JobCreate) -> dict[str, Any]:
    """Crée une offre qui devra être sélectionnée pour lancer une analyse."""
    job_id = str(uuid4())
    payload = job_input.model_dump() if hasattr(job_input, "model_dump") else job_input.dict()
    job = {"id": job_id, **payload}
    jobs = load_jobs()
    jobs[job_id] = job
    save_jobs(jobs)
    return job


@app.post("/predict")
def predict(candidate: Candidate) -> dict[str, Any]:
    job = get_job_or_404(candidate.job_id)
    matching = match_resume_to_job(candidate.resume_text, candidate.years_experience, job)
    return {"offre": {"id": job["id"], "titre": job["title"]}, "adequation": matching, "responsabilite_humaine": "L'IA propose, l'humain dispose. Le score évalue seulement l'adéquation aux critères déclarés de l'offre, pas la valeur d'une personne ni une décision d'embauche."}


@app.post("/predict-batch")
async def predict_batch(job_id: str = Form(...), files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Analyse jusqu'à 200 CV à partir de PDF, DOCX, TXT, CSV ou ZIP.

    Les résultats sont triés par score pour faciliter la revue, mais ne doivent
    jamais constituer le seul critère de sélection ou de rejet.
    """
    job = get_job_or_404(job_id)
    if not files:
        raise HTTPException(status_code=400, detail="Au moins un fichier est requis.")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_FILES} fichiers par envoi.")
    uploads = []
    for upload in files:
        filename = upload.filename or "cv_sans_nom.txt"
        if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Format non autorisé : {filename}")
        uploads.append((filename, await upload.read()))
    records, errors = parse_uploaded_files(uploads)
    return {
        "total_recu": len(records) + len(errors),
        "total_analyse": len(records),
        "offre": {"id": job["id"], "titre": job["title"]},
        "resultats": batch_job_matches(records, job),
        "erreurs_extraction": errors,
        "responsabilite_humaine": "L'IA propose, l'humain dispose. Le classement n'est qu'une aide à la revue RH et ne peut automatiser une décision.",
    }


@app.get("/evaluation")
def evaluation() -> dict[str, Any]:
    assert TEST_DATA is not None
    return comparison_report(MODELS["baseline"], MODELS["advanced"], TEST_DATA)


@app.get("/fairness-audit")
def fairness_audit() -> dict[str, Any]:
    assert TEST_DATA is not None
    return audit_fairness(MODELS["advanced"], TEST_DATA)


@app.get("/baseline-terms")
def baseline_terms() -> dict[str, Any]:
    return influential_terms(MODELS["baseline"])
