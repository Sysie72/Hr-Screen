"""Évaluation figée sur le jeu de test et contrôle de cas atypiques."""
from __future__ import annotations

from typing import Any
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.preprocessing import NUMERIC_COLUMN, TARGET_COLUMN, TEXT_COLUMN

EDGE_CASES = pd.DataFrame([
    {TEXT_COLUMN: "autodidacte github projets open source python data engineering", NUMERIC_COLUMN: 2, "description": "Autodidacte sans diplôme prestigieux"},
    {TEXT_COLUMN: "portfolio créatif compétences en mosaïque tableaux symboles ### python sql", NUMERIC_COLUMN: 3, "description": "CV à mise en page complexe"},
    {TEXT_COLUMN: "archéologue devenu développeur, scripts python et analyse de données", NUMERIC_COLUMN: 4, "description": "Profil atypique en reconversion"},
])


def evaluate_model(model: Any, test_data: pd.DataFrame) -> dict[str, Any]:
    """Retourne des valeurs sérialisables, dont les faux négatifs à surveiller."""
    x_test = test_data[[TEXT_COLUMN, NUMERIC_COLUMN]]
    y_true = test_data[TARGET_COLUMN]
    predictions = model.predict(x_test)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "confusion_matrix": matrix.tolist(),  # [[vrais négatifs, faux positifs], [faux négatifs, vrais positifs]]
        "false_negatives": int(matrix[1, 0]),
    }


def evaluate_edge_cases(model: Any) -> list[dict[str, Any]]:
    """Vérifie explicitement des profils que les données standards représentent mal."""
    features = EDGE_CASES[[TEXT_COLUMN, NUMERIC_COLUMN]]
    probabilities = model.predict_proba(features)[:, 1]
    predictions = model.predict(features)
    return [{"description": row["description"], "prediction": "Sélectionné" if int(prediction) else "Rejeté", "probabilite_selection": round(float(probability), 4)} for (_, row), prediction, probability in zip(EDGE_CASES.iterrows(), predictions, probabilities)]


def comparison_report(baseline: Any, advanced: Any, test_data: pd.DataFrame) -> dict[str, Any]:
    """Produit le rapport unique utilisé par l'API ou une exécution hors ligne."""
    return {"baseline": evaluate_model(baseline, test_data), "advanced": evaluate_model(advanced, test_data), "edge_cases": {"baseline": evaluate_edge_cases(baseline), "advanced": evaluate_edge_cases(advanced)}}
