"""Explainability helpers for recruiter-facing output."""

from __future__ import annotations

from typing import Dict, List, Set

import pandas as pd


def _split_skills(value: str) -> Set[str]:
    if not isinstance(value, str) or not value.strip():
        return set()
    return {s.strip().lower() for s in value.split(";") if s.strip()}


def build_explanation(row: pd.Series) -> Dict[str, str]:
    candidate_skills = _split_skills(row["candidate_skills"])
    required = _split_skills(row["required_skills"])
    optional = _split_skills(row["optional_skills"])

    matched_required = sorted(candidate_skills & required)
    missing_required = sorted(required - candidate_skills)
    matched_optional = sorted(candidate_skills & optional)

    return {
        "matched_required_skills": ", ".join(matched_required[:8]) if matched_required else "none",
        "missing_required_skills": ", ".join(missing_required[:8]) if missing_required else "none",
        "matched_optional_skills": ", ".join(matched_optional[:8]) if matched_optional else "none",
        "experience_summary": f"{int(row['candidate_years'])}y candidate vs {int(row['job_min_years'])}y required",
    }
