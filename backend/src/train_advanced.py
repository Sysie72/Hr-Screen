"""Modèle dense plus expressif : TF-IDF, SVD et gradient boosting."""
from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import NUMERIC_COLUMN, TARGET_COLUMN, TEXT_COLUMN, load_and_split

MODEL_PATH = Path("models/advanced.joblib")


def build_advanced() -> Pipeline:
    """Réduit le texte en représentations denses, puis apprend des interactions non linéaires."""
    features = ColumnTransformer([
        ("texte", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=8_000), TEXT_COLUMN),
        ("experience", StandardScaler(), [NUMERIC_COLUMN]),
    ])
    return Pipeline([
        ("features", features),
        ("svd", TruncatedSVD(n_components=30, random_state=42)),
        ("classifier", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, random_state=42)),
    ])


def train_advanced(train_data: pd.DataFrame, output_path: Path = MODEL_PATH) -> Pipeline:
    model = build_advanced()
    model.fit(train_data[[TEXT_COLUMN, NUMERIC_COLUMN]], train_data[TARGET_COLUMN])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return model


if __name__ == "__main__":
    train, _ = load_and_split()
    train_advanced(train)
    print(f"Modèle avancé enregistré dans {MODEL_PATH}")
