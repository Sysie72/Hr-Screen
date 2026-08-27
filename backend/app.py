"""API FastAPI du démonstrateur HR-Screen.

L'API ne rend aucune décision RH : elle retourne une recommandation accompagnée
d'un rappel de contrôle humain obligatoire.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.evaluation import comparison_report
from src.explainability import explain_advanced_prediction
from src.fairness_audit import audit_fairness
from src.preprocessing import NUMERIC_COLUMN, TEXT_COLUMN, clean_text, load_and_split
from src.train_advanced import MODEL_PATH as ADVANCED_PATH, train_advanced
from src.train_baseline import MODEL_PATH as BASELINE_PATH, influential_terms, train_baseline

MODELS: dict[str, Any] = {}
TEST_DATA: pd.DataFrame | None = None


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


def candidate_frame(candidate: Candidate) -> pd.DataFrame:
    return pd.DataFrame([{TEXT_COLUMN: clean_text(candidate.resume_text), NUMERIC_COLUMN: candidate.years_experience}])


def prediction(model: Any, data: pd.DataFrame) -> dict[str, Any]:
    probability = float(model.predict_proba(data)[0, 1])
    return {"decision": "Sélectionné" if probability >= 0.5 else "Rejeté", "probabilite_selection": round(probability, 4), "seuil": 0.5}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "message": "Service de démonstration; revue humaine obligatoire."}


@app.post("/predict")
def predict(candidate: Candidate) -> dict[str, Any]:
    data = candidate_frame(candidate)
    return {"baseline": prediction(MODELS["baseline"], data), "advanced": prediction(MODELS["advanced"], data), "explication": explain_advanced_prediction(MODELS["advanced"], data), "responsabilite_humaine": "L'IA propose, l'humain dispose. Aucune décision automatisée ne doit être prise à partir de cette sortie."}


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
