"""Streamlit app for recruiter-facing ranking view."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_config


st.set_page_config(page_title="HireSense AI", page_icon="📄", layout="wide")

st.title("HireSense AI")
st.caption("Resume-to-Job Match Engine for Recruiters")

config = load_config()
rankings_path = project_root / config["paths"]["results"] / "sample_rankings.csv"

if not rankings_path.exists():
    st.warning("No rankings found. Run: python run_project.py")
else:
    df = pd.read_csv(rankings_path)
    st.subheader("Top Candidate Recommendations")

    min_score = st.slider("Minimum match score", 0, 100, 65)
    filtered = df[df["match_score"] >= min_score].copy()

    st.dataframe(filtered, use_container_width=True)

    st.subheader("Top 3 Candidates")
    top3 = filtered.head(3)
    for _, row in top3.iterrows():
        st.markdown(f"- {row['candidate_id']} score={row['match_score']}")
        st.markdown(f"  matched required: {row['matched_required_skills']}")
        st.markdown(f"  missing required: {row['missing_required_skills']}")
