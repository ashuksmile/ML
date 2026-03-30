"""Feature engineering for resume-job matching."""

from __future__ import annotations

from typing import Dict, List, Set

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EDU_RANK = {"high_school": 0, "bachelor": 1, "master": 2, "phd": 3}


def _split_skills(value: str) -> Set[str]:
    if not isinstance(value, str) or not value.strip():
        return set()
    return {s.strip().lower() for s in value.split(";") if s.strip()}


def build_similarity_feature(df: pd.DataFrame, max_features: int = 3500) -> np.ndarray:
    corpus = pd.concat([df["resume_text"], df["job_text"]], axis=0).astype(str).tolist()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features)
    vec.fit(corpus)

    resume_vec = vec.transform(df["resume_text"].astype(str).tolist())
    job_vec = vec.transform(df["job_text"].astype(str).tolist())
    sim = cosine_similarity(resume_vec, job_vec).diagonal()
    return sim


def build_structured_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    req_weight = config["matching"]["required_skill_weight"]
    opt_weight = config["matching"]["optional_skill_weight"]

    rows: List[Dict[str, float]] = []
    for _, r in df.iterrows():
        c_skills = _split_skills(r["candidate_skills"])
        req = _split_skills(r["required_skills"])
        opt = _split_skills(r["optional_skills"])

        req_overlap = len(c_skills & req) / max(1, len(req))
        opt_overlap = len(c_skills & opt) / max(1, len(opt))
        weighted_skill_fit = req_weight * req_overlap + opt_weight * opt_overlap

        exp_ratio = float(r["candidate_years"]) / max(1.0, float(r["job_min_years"]))
        exp_fit = min(1.5, exp_ratio)

        c_edu = EDU_RANK.get(str(r["candidate_education"]).lower(), 0)
        j_edu = EDU_RANK.get(str(r["job_min_education"]).lower(), 0)
        edu_gap = float(c_edu - j_edu)
        edu_fit = 1.0 if c_edu >= j_edu else 0.0

        rows.append(
            {
                "required_overlap": req_overlap,
                "optional_overlap": opt_overlap,
                "weighted_skill_fit": weighted_skill_fit,
                "experience_fit": exp_fit,
                "education_fit": edu_fit,
                "education_gap": edu_gap,
            }
        )

    return pd.DataFrame(rows)


def build_feature_table(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    struct_df = build_structured_features(df, config)
    struct_df["semantic_similarity"] = build_similarity_feature(df)
    return struct_df
