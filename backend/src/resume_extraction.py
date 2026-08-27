"""Extraction robuste de CV pour le traitement par lot.

Les règles sont explicites et auditables : aucune inférence d'attribut sensible
(genre, origine, âge, photo, etc.) n'est effectuée.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
import zipfile

import pandas as pd

MAX_FILES = 200
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 Mo par document, limite anti-abus.
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".zip"}

SKILL_PATTERNS = {
    "Python": r"\bpython\b", "SQL": r"\bsql\b", "Machine learning": r"\b(machine learning|apprentissage automatique|scikit[- ]learn)\b",
    "Data analysis": r"\b(data analysis|analyse de donn[eé]es|pandas)\b", "Docker": r"\bdocker\b",
    "Cloud": r"\b(aws|azure|gcp|cloud)\b", "React": r"\breact(?:js)?\b", "Java": r"\bjava\b",
    "Excel": r"\bexcel\b", "Power BI": r"\bpower\s?bi\b",
}


def extract_years_experience(text: str) -> float:
    """Extrait la durée déclarée, jamais l'âge de la personne."""
    normalized = text.lower()
    matches = re.findall(r"\b(\d{1,2}(?:[,.]\d+)?)\s*(?:ans?|ann(?:é|e)es?)\s+d['’ ]?(?:exp[eé]rience|exp)\b", normalized)
    if matches:
        return min(max(float(value.replace(",", ".")) for value in matches), 70.0)
    # Exemple : "2018 - 2024". On conserve l'intervalle le plus long plausible.
    intervals = re.findall(r"\b(19\d{2}|20\d{2})\s*[-–]\s*(19\d{2}|20\d{2}|pr[eé]sent|today)\b", normalized)
    durations = []
    for start, end in intervals:
        end_year = 2026 if end in {"présent", "present", "today"} else int(end)
        duration = end_year - int(start)
        if 0 <= duration <= 50:
            durations.append(duration)
    return float(max(durations, default=0))


def extract_skills(text: str) -> list[str]:
    """Retourne des compétences trouvées par des motifs Regex versionnés."""
    return [skill for skill, pattern in SKILL_PATTERNS.items() if re.search(pattern, text, flags=re.IGNORECASE)]


def _text_from_pdf(content: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _text_from_docx(content: bytes) -> str:
    from docx import Document
    document = Document(BytesIO(content))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def extract_text(filename: str, content: bytes) -> str:
    """Lit un CV texte, PDF ou DOCX; CSV et ZIP sont traités plus haut."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _text_from_pdf(content)
    if suffix == ".docx":
        return _text_from_docx(content)
    if suffix == ".txt":
        return content.decode("utf-8", errors="replace")
    raise ValueError(f"Format non pris en charge : {suffix}")


def _record(filename: str, text: str, years: float | None = None) -> dict:
    return {"fichier": filename, "texte": text.strip(), "annees_experience": extract_years_experience(text) if years is None else max(float(years), 0), "competences_extraites": extract_skills(text)}


def parse_uploaded_files(files: list[tuple[str, bytes]]) -> tuple[list[dict], list[dict]]:
    """Transforme des fichiers téléversés en CV, avec erreurs isolées par document."""
    records, errors = [], []
    expanded: list[tuple[str, bytes]] = []
    for filename, content in files:
        suffix = Path(filename).suffix.lower()
        if suffix == ".zip":
            try:
                with zipfile.ZipFile(BytesIO(content)) as archive:
                    for entry in archive.infolist():
                        if entry.is_dir() or Path(entry.filename).suffix.lower() not in {".pdf", ".docx", ".txt"}:
                            continue
                        if entry.file_size <= MAX_FILE_SIZE:
                            expanded.append((Path(entry.filename).name, archive.read(entry)))
            except zipfile.BadZipFile:
                errors.append({"fichier": filename, "erreur": "Archive ZIP invalide."})
        else:
            expanded.append((filename, content))
    for filename, content in expanded[:MAX_FILES]:
        try:
            if len(content) > MAX_FILE_SIZE:
                raise ValueError("Fichier trop volumineux (maximum 10 Mo).")
            if Path(filename).suffix.lower() == ".csv":
                table = pd.read_csv(BytesIO(content))
                if "resume_text" not in table.columns:
                    raise ValueError("Le CSV doit contenir la colonne resume_text.")
                for index, row in table.head(MAX_FILES - len(records)).iterrows():
                    records.append(_record(str(row.get("filename", f"{Path(filename).stem}_{index}")), str(row["resume_text"]), row.get("years_experience") if pd.notna(row.get("years_experience")) else None))
            else:
                records.append(_record(filename, extract_text(filename, content)))
        except Exception as error:
            errors.append({"fichier": filename, "erreur": str(error)})
    return records, errors
