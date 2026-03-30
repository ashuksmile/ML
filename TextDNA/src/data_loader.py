"""Data loading and dataset generation utilities for HireSense AI."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

SKILL_POOL = [
    "python", "sql", "machine learning", "deep learning", "nlp", "aws", "azure",
    "docker", "kubernetes", "tableau", "power bi", "statistics", "etl", "spark",
    "communication", "leadership", "project management", "fastapi", "streamlit", "git",
]

EDUCATION_LEVELS = ["high_school", "bachelor", "master", "phd"]

CANDIDATE_ALIASES = {
    "candidate_id": "candidate_id",
    "id": "candidate_id",
    "name": "name",
    "full_name": "name",
    "years_experience": "years_experience",
    "experience_years": "years_experience",
    "education_level": "education_level",
    "education": "education_level",
    "skills": "skills",
    "skill_set": "skills",
    "resume_text": "resume_text",
    "resume": "resume_text",
    "summary": "resume_text",
}

JOB_ALIASES = {
    "job_id": "job_id",
    "id": "job_id",
    "job_title": "job_title",
    "title": "job_title",
    "min_years": "min_years",
    "minimum_years": "min_years",
    "min_education": "min_education",
    "education": "min_education",
    "required_skills": "required_skills",
    "required": "required_skills",
    "optional_skills": "optional_skills",
    "optional": "optional_skills",
    "job_text": "job_text",
    "description": "job_text",
    "job_description": "job_text",
}

PAIRS_ALIASES = {
    "candidate_id": "candidate_id",
    "job_id": "job_id",
    "match_label": "match_label",
    "label": "match_label",
}


def load_config(config_path: str = "config/config.yaml") -> dict:
    project_root = Path(__file__).parent.parent
    with open(project_root / config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dirs(config: dict) -> Dict[str, Path]:
    root = Path(__file__).parent.parent
    dirs = {
        "raw": root / config["paths"]["raw_data"],
        "processed": root / config["paths"]["processed_data"],
        "models": root / config["paths"]["models"],
        "results": root / config["paths"]["results"],
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _normalize_skill_text(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    if ";" in value:
        parts = value.split(";")
    elif "," in value:
        parts = value.split(",")
    else:
        parts = [value]
    clean = sorted({p.strip().lower() for p in parts if p.strip()})
    return ";".join(clean)


def _edu_guess(value: str) -> Optional[str]:
    v = str(value).strip().lower()
    if "phd" in v or "doctor" in v:
        return "phd"
    if "master" in v or "msc" in v or "mba" in v:
        return "master"
    if "bachelor" in v or "btech" in v or "be" == v:
        return "bachelor"
    if "high" in v or "school" in v or "diploma" in v:
        return "high_school"
    return None


def _edu_normalize(value: str) -> str:
    guessed = _edu_guess(value)
    return guessed if guessed is not None else "bachelor"


def _rename_using_aliases(df: pd.DataFrame, aliases: Dict[str, str]) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in aliases:
            rename_map[col] = aliases[key]
    return df.rename(columns=rename_map)


def _candidate_column_aliases() -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for alias, canonical in CANDIDATE_ALIASES.items():
        grouped.setdefault(canonical, []).append(alias)
    return grouped


def _job_column_aliases() -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for alias, canonical in JOB_ALIASES.items():
        grouped.setdefault(canonical, []).append(alias)
    return grouped


def _require_alias_column_presence(raw_df: pd.DataFrame, required: List[str], grouped_aliases: Dict[str, List[str]]) -> List[str]:
    lower_cols = {str(c).strip().lower() for c in raw_df.columns}
    errors: List[str] = []
    for canonical in required:
        aliases = grouped_aliases.get(canonical, [canonical])
        if not any(alias in lower_cols for alias in aliases):
            errors.append(
                f"missing column for '{canonical}'. accepted aliases: {aliases}"
            )
    return errors


def _first_present_alias_column(raw_df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    alias_set = {a.lower() for a in aliases}
    for col in raw_df.columns:
        if str(col).strip().lower() in alias_set:
            return col
    return None


def _validate_non_negative_numeric_raw(raw_df: pd.DataFrame, aliases: List[str], canonical: str) -> List[str]:
    errors: List[str] = []
    source_col = _first_present_alias_column(raw_df, aliases)
    if source_col is None:
        return errors

    vals = pd.to_numeric(raw_df[source_col], errors="coerce")
    bad_numeric = vals.isna()
    bad_negative = vals < 0

    bad_numeric_idx = bad_numeric[bad_numeric].index.tolist()[:10]
    bad_negative_idx = bad_negative[bad_negative].index.tolist()[:10]

    if bad_numeric_idx:
        errors.append(f"column '{canonical}' has non-numeric values at rows {bad_numeric_idx}")
    if bad_negative_idx:
        errors.append(f"column '{canonical}' has negative values at rows {bad_negative_idx}")
    return errors


def _validate_education_values_raw(raw_df: pd.DataFrame, aliases: List[str], canonical: str) -> List[str]:
    errors: List[str] = []
    source_col = _first_present_alias_column(raw_df, aliases)
    if source_col is None:
        return errors

    bad_rows: List[int] = []
    for idx, value in raw_df[source_col].items():
        if _edu_guess(str(value)) is None:
            bad_rows.append(int(idx))
        if len(bad_rows) >= 10:
            break
    if bad_rows:
        errors.append(
            f"column '{canonical}' has unsupported values at rows {bad_rows}. use: high_school, bachelor, master, phd"
        )
    return errors


def validate_candidates_input(raw_candidates_df: pd.DataFrame) -> None:
    errors: List[str] = []
    grouped = _candidate_column_aliases()
    errors.extend(
        _require_alias_column_presence(
            raw_candidates_df,
            required=["skills", "years_experience", "education_level"],
            grouped_aliases=grouped,
        )
    )
    if errors:
        raise ValueError("Candidates CSV schema errors:\n- " + "\n- ".join(errors))

    normalized = _base_candidate_schema(raw_candidates_df)
    errors.extend(
        _validate_non_negative_numeric_raw(
            raw_candidates_df,
            grouped["years_experience"],
            "years_experience",
        )
    )
    errors.extend(
        _validate_education_values_raw(
            raw_candidates_df,
            grouped["education_level"],
            "education_level",
        )
    )

    if normalized["skills"].eq("").all():
        errors.append("all rows have empty skills. provide at least one skill per candidate")
    if normalized["resume_text"].astype(str).str.strip().eq("").all():
        errors.append("all rows have empty resume text")

    if errors:
        raise ValueError("Candidates CSV value errors:\n- " + "\n- ".join(errors))


def validate_jobs_input(raw_jobs_df: pd.DataFrame) -> None:
    errors: List[str] = []
    grouped = _job_column_aliases()
    errors.extend(
        _require_alias_column_presence(
            raw_jobs_df,
            required=["required_skills", "min_years", "min_education"],
            grouped_aliases=grouped,
        )
    )
    if errors:
        raise ValueError("Jobs CSV schema errors:\n- " + "\n- ".join(errors))

    normalized = _base_job_schema(raw_jobs_df)
    errors.extend(
        _validate_non_negative_numeric_raw(
            raw_jobs_df,
            grouped["min_years"],
            "min_years",
        )
    )
    errors.extend(
        _validate_education_values_raw(
            raw_jobs_df,
            grouped["min_education"],
            "min_education",
        )
    )

    if normalized["required_skills"].eq("").all():
        errors.append("all rows have empty required skills")
    if normalized["job_text"].astype(str).str.strip().eq("").all():
        errors.append("all rows have empty job text")

    if errors:
        raise ValueError("Jobs CSV value errors:\n- " + "\n- ".join(errors))


def _base_candidate_schema(candidates_df: pd.DataFrame) -> pd.DataFrame:
    df = _rename_using_aliases(candidates_df.copy(), CANDIDATE_ALIASES)
    if "candidate_id" not in df.columns:
        df["candidate_id"] = [f"C{i+1:04d}" for i in range(len(df))]
    if "name" not in df.columns:
        df["name"] = [f"Candidate_{i+1}" for i in range(len(df))]
    if "years_experience" not in df.columns:
        df["years_experience"] = 0
    if "education_level" not in df.columns:
        df["education_level"] = "bachelor"
    if "skills" not in df.columns:
        df["skills"] = ""
    if "resume_text" not in df.columns:
        df["resume_text"] = (
            "Candidate " + df["name"].astype(str) + " with skills " + df["skills"].astype(str)
        )

    df["skills"] = df["skills"].astype(str).map(_normalize_skill_text)
    df["education_level"] = df["education_level"].map(_edu_normalize)
    df["years_experience"] = pd.to_numeric(df["years_experience"], errors="coerce").fillna(0).clip(lower=0)
    return df[["candidate_id", "name", "years_experience", "education_level", "skills", "resume_text"]]


def _base_job_schema(jobs_df: pd.DataFrame) -> pd.DataFrame:
    df = _rename_using_aliases(jobs_df.copy(), JOB_ALIASES)
    if "job_id" not in df.columns:
        df["job_id"] = [f"J{i+1:03d}" for i in range(len(df))]
    if "job_title" not in df.columns:
        df["job_title"] = "Role"
    if "min_years" not in df.columns:
        df["min_years"] = 0
    if "min_education" not in df.columns:
        df["min_education"] = "bachelor"
    if "required_skills" not in df.columns:
        df["required_skills"] = ""
    if "optional_skills" not in df.columns:
        df["optional_skills"] = ""
    if "job_text" not in df.columns:
        df["job_text"] = (
            "Role " + df["job_title"].astype(str) + " requires " + df["required_skills"].astype(str)
        )

    df["required_skills"] = df["required_skills"].astype(str).map(_normalize_skill_text)
    df["optional_skills"] = df["optional_skills"].astype(str).map(_normalize_skill_text)
    df["min_education"] = df["min_education"].map(_edu_normalize)
    df["min_years"] = pd.to_numeric(df["min_years"], errors="coerce").fillna(0).clip(lower=0)
    return df[["job_id", "job_title", "min_years", "min_education", "required_skills", "optional_skills", "job_text"]]


def _score_pair_for_label(row: pd.Series) -> float:
    education_rank = {"high_school": 0, "bachelor": 1, "master": 2, "phd": 3}
    c_skills = set(str(row["candidate_skills"]).split(";")) if row["candidate_skills"] else set()
    req = set(str(row["required_skills"]).split(";")) if row["required_skills"] else set()
    opt = set(str(row["optional_skills"]).split(";")) if row["optional_skills"] else set()

    req_overlap = len(c_skills & req) / max(1, len(req))
    opt_overlap = len(c_skills & opt) / max(1, len(opt))
    exp_fit = min(1.0, float(row["candidate_years"]) / max(1.0, float(row["job_min_years"])))
    edu_fit = 1.0 if education_rank.get(str(row["candidate_education"]), 1) >= education_rank.get(str(row["job_min_education"]), 1) else 0.0
    return 0.55 * req_overlap + 0.20 * opt_overlap + 0.20 * exp_fit + 0.05 * edu_fit


def build_pairs_from_candidates_jobs(
    candidates_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    n_pairs: int,
    random_state: int = 42,
    label_threshold: float = 0.62,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    rows = []
    for _ in range(n_pairs):
        c = candidates_df.iloc[int(rng.integers(0, len(candidates_df)))]
        j = jobs_df.iloc[int(rng.integers(0, len(jobs_df)))]

        row = {
            "candidate_id": c["candidate_id"],
            "job_id": j["job_id"],
            "resume_text": c["resume_text"],
            "job_text": j["job_text"],
            "candidate_skills": c["skills"],
            "required_skills": j["required_skills"],
            "optional_skills": j["optional_skills"],
            "candidate_years": c["years_experience"],
            "job_min_years": j["min_years"],
            "candidate_education": c["education_level"],
            "job_min_education": j["min_education"],
        }
        score = _score_pair_for_label(pd.Series(row)) + float(rng.normal(0, 0.05))
        row["match_label"] = int(score >= label_threshold)
        rows.append(row)

    return pd.DataFrame(rows)


def _random_skill_set(min_n: int = 4, max_n: int = 9) -> List[str]:
    n = random.randint(min_n, max_n)
    return random.sample(SKILL_POOL, n)


def _build_resume_text(name: str, years: int, skills: List[str], education: str) -> str:
    skill_text = ", ".join(skills)
    return (
        f"Candidate {name} has {years} years of experience. "
        f"Primary skills include {skill_text}. "
        f"Education level: {education}."
    )


def _build_job_text(title: str, required: List[str], optional: List[str], min_years: int, education: str) -> str:
    req = ", ".join(required)
    opt = ", ".join(optional)
    return (
        f"Role {title} requires at least {min_years} years experience. "
        f"Required skills: {req}. Preferred skills: {opt}. "
        f"Minimum education: {education}."
    )


def generate_sample_data(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create realistic synthetic candidates, jobs, and labeled match pairs."""
    random.seed(config["data"]["random_state"])
    np.random.seed(config["data"]["random_state"])

    n_candidates = config["data"]["sample_candidates"]
    n_jobs = config["data"]["sample_jobs"]
    n_pairs = config["data"]["sample_pairs"]

    candidates = []
    for i in range(n_candidates):
        cid = f"C{i+1:04d}"
        years = int(np.clip(np.random.normal(4.5, 2.2), 0, 15))
        skills = _random_skill_set()
        education = random.choice(EDUCATION_LEVELS)
        candidates.append(
            {
                "candidate_id": cid,
                "name": f"Candidate_{i+1}",
                "years_experience": years,
                "education_level": education,
                "skills": ";".join(skills),
                "resume_text": _build_resume_text(f"Candidate_{i+1}", years, skills, education),
            }
        )

    role_names = [
        "ML Engineer", "Data Analyst", "Data Scientist", "Backend Engineer", "BI Engineer",
        "AI Product Analyst", "MLOps Engineer", "NLP Engineer", "Analytics Lead",
    ]
    jobs = []
    for i in range(n_jobs):
        jid = f"J{i+1:03d}"
        required = _random_skill_set(3, 6)
        optional = [s for s in _random_skill_set(2, 4) if s not in required]
        min_years = random.randint(1, 8)
        education = random.choice(EDUCATION_LEVELS[1:])
        title = random.choice(role_names)
        jobs.append(
            {
                "job_id": jid,
                "job_title": title,
                "min_years": min_years,
                "min_education": education,
                "required_skills": ";".join(required),
                "optional_skills": ";".join(optional),
                "job_text": _build_job_text(title, required, optional, min_years, education),
            }
        )

    candidates_df = pd.DataFrame(candidates)
    jobs_df = pd.DataFrame(jobs)

    education_rank = {"high_school": 0, "bachelor": 1, "master": 2, "phd": 3}

    pairs = []
    for _ in range(n_pairs):
        c = candidates_df.sample(1).iloc[0]
        j = jobs_df.sample(1).iloc[0]

        c_skills = set(c["skills"].split(";"))
        req = set(j["required_skills"].split(";"))
        opt = set(j["optional_skills"].split(";")) if j["optional_skills"] else set()

        req_overlap = len(c_skills & req) / max(1, len(req))
        opt_overlap = len(c_skills & opt) / max(1, len(opt))
        exp_fit = min(1.0, c["years_experience"] / max(1, j["min_years"]))
        edu_fit = 1.0 if education_rank[c["education_level"]] >= education_rank[j["min_education"]] else 0.0

        noisy_score = 0.55 * req_overlap + 0.20 * opt_overlap + 0.20 * exp_fit + 0.05 * edu_fit
        noisy_score += np.random.normal(0, 0.06)

        pairs.append(
            {
                "candidate_id": c["candidate_id"],
                "job_id": j["job_id"],
                "resume_text": c["resume_text"],
                "job_text": j["job_text"],
                "candidate_skills": c["skills"],
                "required_skills": j["required_skills"],
                "optional_skills": j["optional_skills"],
                "candidate_years": c["years_experience"],
                "job_min_years": j["min_years"],
                "candidate_education": c["education_level"],
                "job_min_education": j["min_education"],
                "match_label": int(noisy_score >= 0.62),
            }
        )

    pairs_df = pd.DataFrame(pairs)
    return candidates_df, jobs_df, pairs_df


def import_real_data(
    config: dict,
    candidates_path: str,
    jobs_path: str,
    pairs_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Import user-provided candidate and job CSVs and persist standardized raw files."""
    raw_candidates_df = pd.read_csv(candidates_path)
    raw_jobs_df = pd.read_csv(jobs_path)
    validate_candidates_input(raw_candidates_df)
    validate_jobs_input(raw_jobs_df)

    candidates_df = _base_candidate_schema(raw_candidates_df)
    jobs_df = _base_job_schema(raw_jobs_df)

    if pairs_path:
        p_raw = _rename_using_aliases(pd.read_csv(pairs_path), PAIRS_ALIASES)
        required_cols = {"candidate_id", "job_id", "match_label"}
        missing = required_cols.difference(set(p_raw.columns))
        if missing:
            raise ValueError(f"pairs file missing columns: {sorted(missing)}")

        merge_base = candidates_df.merge(jobs_df, how="cross")
        merge_base = merge_base.rename(
            columns={
                "skills": "candidate_skills",
                "required_skills": "required_skills",
                "optional_skills": "optional_skills",
                "years_experience": "candidate_years",
                "min_years": "job_min_years",
                "education_level": "candidate_education",
                "min_education": "job_min_education",
            }
        )
        pairs_df = merge_base.merge(
            p_raw[["candidate_id", "job_id", "match_label"]],
            on=["candidate_id", "job_id"],
            how="inner",
        )
    else:
        pairs_df = build_pairs_from_candidates_jobs(
            candidates_df,
            jobs_df,
            n_pairs=config["data"].get("sample_pairs", 2000),
            random_state=config["data"]["random_state"],
            label_threshold=0.62,
        )

    save_raw_data(config, candidates_df, jobs_df, pairs_df)
    return candidates_df, jobs_df, pairs_df


def save_raw_data(config: dict, candidates_df: pd.DataFrame, jobs_df: pd.DataFrame, pairs_df: pd.DataFrame) -> None:
    dirs = _ensure_dirs(config)
    candidates_df.to_csv(dirs["raw"] / "candidates.csv", index=False)
    jobs_df.to_csv(dirs["raw"] / "jobs.csv", index=False)
    pairs_df.to_csv(dirs["raw"] / "labeled_pairs.csv", index=False)


def load_raw_data(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dirs = _ensure_dirs(config)
    candidates = pd.read_csv(dirs["raw"] / "candidates.csv")
    jobs = pd.read_csv(dirs["raw"] / "jobs.csv")
    pairs = pd.read_csv(dirs["raw"] / "labeled_pairs.csv")
    return candidates, jobs, pairs


def load_or_create_training_data(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Load existing raw data if available, otherwise generate synthetic data.

    Returns source type as one of: real, synthetic, existing.
    """
    dirs = _ensure_dirs(config)
    cand_path = dirs["raw"] / "candidates.csv"
    job_path = dirs["raw"] / "jobs.csv"
    pair_path = dirs["raw"] / "labeled_pairs.csv"

    if cand_path.exists() and job_path.exists() and pair_path.exists():
        c, j, p = load_raw_data(config)
        return c, j, p, "existing"

    if cand_path.exists() and job_path.exists():
        candidates_df = _base_candidate_schema(pd.read_csv(cand_path))
        jobs_df = _base_job_schema(pd.read_csv(job_path))
        pairs_df = build_pairs_from_candidates_jobs(
            candidates_df,
            jobs_df,
            n_pairs=config["data"].get("sample_pairs", 2000),
            random_state=config["data"]["random_state"],
            label_threshold=0.62,
        )
        save_raw_data(config, candidates_df, jobs_df, pairs_df)
        return candidates_df, jobs_df, pairs_df, "real"

    candidates_df, jobs_df, pairs_df = generate_sample_data(config)
    save_raw_data(config, candidates_df, jobs_df, pairs_df)
    return candidates_df, jobs_df, pairs_df, "synthetic"
