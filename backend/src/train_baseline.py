"""Entraînement du modèle transparent TF-IDF + régression logistique."""
from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import NUMERIC_COLUMN, TARGET_COLUMN, TEXT_COLUMN, load_and_split

MODEL_PATH = Path("models/baseline.joblib")


def build_baseline() -> Pipeline:
    """Construit une chaîne interprétable, sans fuite des données de test."""
    features = ColumnTransformer([
        ("texte", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5_000), TEXT_COLUMN),
        ("experience", StandardScaler(), [NUMERIC_COLUMN]),
    ])
    return Pipeline([("features", features), ("classifier", LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=42))])


def train_baseline(train_data: pd.DataFrame, output_path: Path = MODEL_PATH) -> Pipeline:
    model = build_baseline()
    model.fit(train_data[[TEXT_COLUMN, NUMERIC_COLUMN]], train_data[TARGET_COLUMN])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return model


def influential_terms(model: Pipeline, top_n: int = 12) -> dict[str, list[tuple[str, float]]]:
    """Expose les termes les plus favorables/défavorables de la régression."""
    vectorizer = model.named_steps["features"].named_transformers_["texte"]
    names = list(vectorizer.get_feature_names_out()) + ["years_experience"]
    coefficients = model.named_steps["classifier"].coef_[0]
    ranked = sorted(zip(names, coefficients), key=lambda pair: pair[1])
    return {"favorables": [(word, round(float(weight), 4)) for word, weight in ranked[-top_n:][::-1]], "defavorables": [(word, round(float(weight), 4)) for word, weight in ranked[:top_n]]}


if __name__ == "__main__":
    train, _ = load_and_split()
    train_baseline(train)
    print(f"Modèle baseline enregistré dans {MODEL_PATH}")
