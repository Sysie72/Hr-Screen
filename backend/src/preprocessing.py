"""Chargement, nettoyage et séparation strictement reproductible des CV."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

TEXT_COLUMN = "resume_text"
TARGET_COLUMN = "is_selected"
NUMERIC_COLUMN = "years_experience"

# Liste courte, embarquée pour éviter une dépendance réseau (NLTK) dans le prototype.
FRENCH_STOPWORDS = {"le", "la", "les", "de", "des", "du", "un", "une", "et", "ou", "a", "au", "aux", "en", "pour", "avec", "dans", "sur", "par", "ce", "cet", "cette", "mon", "ma", "mes", "je", "nous"}
STOPWORDS = set(ENGLISH_STOP_WORDS).union(FRENCH_STOPWORDS)


def clean_text(text: object) -> str:
    """Normalise le texte sans tenter d'inférer des attributs sensibles."""
    normalized = str(text or "").lower()
    normalized = re.sub(r"[^a-zà-ÿ0-9+#.\s]", " ", normalized)
    tokens = [token for token in normalized.split() if token not in STOPWORDS]
    return " ".join(tokens)


def _synthetic_dataset() -> pd.DataFrame:
    """Jeu pédagogique minimal; ne doit pas servir à une décision réelle."""
    rows = []
    accepted = [
        ("python machine learning sql data analysis cloud", 4),
        ("software engineer python docker api react", 5),
        ("data scientist statistics pandas sklearn tableau", 3),
        ("cybersecurity linux networks incident response", 6),
    ]
    rejected = [
        ("career transition motivated communication customer service", 0),
        ("student seeking first opportunity general office skills", 0),
        ("retail sales teamwork reliable available immediately", 1),
        ("administrative assistant scheduling filing reception", 1),
    ]
    # Les groupes sont artificiels et uniquement destinés au module d'audit.
    groups = ["groupe_a", "groupe_b"]
    for index in range(100):
        text, experience = accepted[index % len(accepted)]
        rows.append({TEXT_COLUMN: f"{text} project {index}", NUMERIC_COLUMN: experience + (index % 2), TARGET_COLUMN: 1, "audit_group": groups[index % 2]})
        text, experience = rejected[index % len(rejected)]
        rows.append({TEXT_COLUMN: f"{text} profile {index}", NUMERIC_COLUMN: experience, TARGET_COLUMN: 0, "audit_group": groups[(index + 1) % 2]})
    return pd.DataFrame(rows)


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Charge un CSV attendu ou retourne un jeu synthétique documenté."""
    if path and Path(path).exists():
        data = pd.read_csv(path)
    else:
        data = _synthetic_dataset()
    required = {TEXT_COLUMN, NUMERIC_COLUMN, TARGET_COLUMN}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Colonnes obligatoires absentes : {sorted(missing)}")
    data = data.copy()
    data[TEXT_COLUMN] = data[TEXT_COLUMN].map(clean_text)
    data[NUMERIC_COLUMN] = pd.to_numeric(data[NUMERIC_COLUMN], errors="coerce").fillna(0)
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="raise").astype(int)
    if not set(data[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("is_selected doit être binaire (0 ou 1).")
    return data


def split_dataset(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Effectue une unique séparation stratifiée; le test reste réservé à l'évaluation."""
    return train_test_split(data, test_size=test_size, random_state=random_state, stratify=data[TARGET_COLUMN])


def load_and_split(path: str | Path | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Point d'entrée commun garantissant le même découpage pour les deux modèles."""
    return split_dataset(load_dataset(path))
