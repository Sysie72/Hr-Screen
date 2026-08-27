"""Explication locale SHAP du modèle avancé, avec dégradation sûre si indisponible."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from src.preprocessing import NUMERIC_COLUMN, TEXT_COLUMN


def explain_advanced_prediction(model: Any, candidate: pd.DataFrame, top_n: int = 8) -> dict[str, Any]:
    """Calcule SHAP sur les dimensions latentes; elles restent des indicateurs, pas des motifs causaux."""
    features = candidate[[TEXT_COLUMN, NUMERIC_COLUMN]]
    transformed = model.named_steps["features"].transform(features)
    dense = model.named_steps["svd"].transform(transformed)
    probability = float(model.predict_proba(features)[0, 1])
    try:
        import shap
        explainer = shap.TreeExplainer(model.named_steps["classifier"])
        values = np.asarray(explainer.shap_values(dense))
        # Certaines versions renvoient (n, dimensions, classes).
        if values.ndim == 3:
            values = values[:, :, 1]
        local = values[0]
        ranked = np.argsort(np.abs(local))[::-1][:top_n]
        factors = [{"facteur": f"Dimension latente {int(index) + 1}", "impact": round(float(local[index]), 5), "sens": "favorise la sélection" if local[index] >= 0 else "défavorise la sélection"} for index in ranked]
        method = "SHAP TreeExplainer"
    except Exception as error:  # L'API reste disponible si SHAP est absent/incompatible.
        factors = [{"facteur": "Expérience déclarée", "impact": float(candidate.iloc[0][NUMERIC_COLUMN]), "sens": "information transmise au modèle"}]
        method = f"Repli sans SHAP ({type(error).__name__})"
    return {"probabilite_selection": round(probability, 4), "methode": method, "facteurs": factors, "avertissement": "Explication indicative : la décision finale relève obligatoirement d'un recruteur humain."}
