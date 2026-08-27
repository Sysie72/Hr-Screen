"""Moteur transparent de correspondance entre un CV et une offre.

Ce score n'est pas une probabilité d'embauche : il représente uniquement les
critères déclarés par le recruteur pour une offre donnée.
"""
from __future__ import annotations

import re
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import clean_text
from src.resume_extraction import extract_skills


def _normalise_skills(skills: list[str]) -> list[str]:
    return [skill.strip() for skill in skills if skill and skill.strip()]


def _is_present(skill: str, resume: str) -> bool:
    """Cherche un terme demandé sans exploiter de donnée personnelle."""
    words = clean_text(skill).split()
    if not words:
        return False
    pattern = r"\b" + r"\s+".join(re.escape(word) for word in words) + r"\b"
    return bool(re.search(pattern, clean_text(resume), flags=re.IGNORECASE))


def _semantic_similarity(resume: str, job_description: str) -> float:
    if not resume.strip() or not job_description.strip():
        return 0.0
    matrix = TfidfVectorizer(ngram_range=(1, 2)).fit_transform([clean_text(resume), clean_text(job_description)])
    return float(cosine_similarity(matrix[0], matrix[1])[0, 0])


def match_resume_to_job(resume: str, years_experience: float, job: dict[str, Any]) -> dict[str, Any]:
    """Retourne un score de correspondance explicable sur 100.

    Pondération : 50 % compétences obligatoires, 20 % compétences souhaitées,
    15 % expérience, 15 % proximité textuelle avec la description.
    """
    required = _normalise_skills(job.get("required_skills", []))
    preferred = _normalise_skills(job.get("preferred_skills", []))
    found_required = [skill for skill in required if _is_present(skill, resume)]
    missing_required = [skill for skill in required if skill not in found_required]
    found_preferred = [skill for skill in preferred if _is_present(skill, resume)]
    minimum_years = float(job.get("min_years_experience", 0))
    required_coverage = len(found_required) / len(required) if required else 1.0
    preferred_coverage = len(found_preferred) / len(preferred) if preferred else 1.0
    experience_coverage = min(max(float(years_experience), 0) / minimum_years, 1.0) if minimum_years > 0 else 1.0
    similarity = _semantic_similarity(resume, job.get("description", ""))
    score = 100 * (0.50 * required_coverage + 0.20 * preferred_coverage + 0.15 * experience_coverage + 0.15 * similarity)
    return {
        "score_adequation": round(score, 1),
        "nature_score": "Score de correspondance à l'offre — ce n'est pas une probabilité d'embauche.",
        "competences_obligatoires_trouvees": found_required,
        "competences_obligatoires_manquantes": missing_required,
        "competences_souhaitees_trouvees": found_preferred,
        "experience_cv": round(float(years_experience), 1),
        "experience_minimale_offre": minimum_years,
        "experience_suffisante": float(years_experience) >= minimum_years,
        "proximite_description": round(similarity * 100, 1),
        "competences_detectees": extract_skills(resume),
        "ponderation": {"competences_obligatoires": "50%", "competences_souhaitees": "20%", "experience": "15%", "proximite_textuelle": "15%"},
    }
