"""Audit de disparités : indicateur de vigilance, jamais preuve de neutralité."""
from __future__ import annotations

from typing import Any
import pandas as pd

from src.preprocessing import NUMERIC_COLUMN, TEXT_COLUMN


def audit_fairness(model: Any, test_data: pd.DataFrame, group_column: str = "audit_group", tolerance: float = 0.10) -> dict[str, Any]:
    """Compare les acceptations par groupe à niveau d'expérience comparable.

    Les groupes sont des proxys simulés : ne pas les employer en production pour
    prendre une décision individuelle. Ici ils servent exclusivement à l'audit.
    """
    if group_column not in test_data.columns:
        raise ValueError(f"Colonne d'audit absente : {group_column}")
    audited = test_data.copy()
    audited["prediction"] = model.predict(audited[[TEXT_COLUMN, NUMERIC_COLUMN]])
    audited["experience_bucket"] = pd.cut(audited[NUMERIC_COLUMN], bins=[-1, 1, 3, 6, float("inf")], labels=["0-1", "2-3", "4-6", "7+"])
    by_group = audited.groupby(group_column, observed=True)["prediction"].agg(acceptation="mean", effectif="count")
    rates = {str(group): {"taux_acceptation": round(float(values["acceptation"]), 4), "effectif": int(values["effectif"])} for group, values in by_group.iterrows()}
    gap = float(by_group["acceptation"].max() - by_group["acceptation"].min()) if len(by_group) > 1 else 0.0
    conditional_gaps = []
    for bucket, frame in audited.groupby("experience_bucket", observed=True):
        group_rates = frame.groupby(group_column)["prediction"].mean()
        if len(group_rates) > 1:
            conditional_gaps.append({"experience": str(bucket), "ecart": round(float(group_rates.max() - group_rates.min()), 4)})
    flagged = gap > tolerance or any(item["ecart"] > tolerance for item in conditional_gaps)
    text = ("ALERTE : un écart de taux d'acceptation supérieur au seuil a été observé. " if flagged else "Aucun écart supérieur au seuil observé sur cet échantillon. ") + "Ce résultat ne démontre pas l'absence de biais : documenter les données, contrôler les faux négatifs et imposer une revue humaine."
    return {"seuil": tolerance, "taux_par_groupe": rates, "ecart_global": round(gap, 4), "ecarts_a_competence_egale": conditional_gaps, "alerte": flagged, "rapport": text}
