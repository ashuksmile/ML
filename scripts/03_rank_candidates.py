"""Generate ranked candidate recommendations for a selected job."""

import argparse
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_config, load_raw_data
from src.explain import build_explanation
from src.feature_builder import build_feature_table
from src.matcher import load_model_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=str, default=None, help="Job ID to rank candidates for")
    parser.add_argument("--top-k", type=int, default=10, help="Top K candidates")
    args = parser.parse_args()

    config = load_config()
    candidates_df, jobs_df, _ = load_raw_data(config)

    job_id = args.job_id or jobs_df.sample(1).iloc[0]["job_id"]
    job_row = jobs_df[jobs_df["job_id"] == job_id]
    if job_row.empty:
        raise ValueError(f"job_id not found: {job_id}")
    job_row = job_row.iloc[0]

    rows = []
    for _, c in candidates_df.iterrows():
        rows.append(
            {
                "candidate_id": c["candidate_id"],
                "job_id": job_id,
                "resume_text": c["resume_text"],
                "job_text": job_row["job_text"],
                "candidate_skills": c["skills"],
                "required_skills": job_row["required_skills"],
                "optional_skills": job_row["optional_skills"],
                "candidate_years": c["years_experience"],
                "job_min_years": job_row["min_years"],
                "candidate_education": c["education_level"],
                "job_min_education": job_row["min_education"],
            }
        )

    infer_df = pd.DataFrame(rows)
    feature_df = build_feature_table(infer_df, config)

    artifact = load_model_artifacts(config)
    model = artifact["model"]

    probs = model.predict_proba(feature_df.values)[:, 1] if hasattr(model, "predict_proba") else model.predict(feature_df.values)
    infer_df["match_score"] = (probs * 100.0).round(2)

    explanations = infer_df.apply(build_explanation, axis=1, result_type="expand")
    final = pd.concat([infer_df[["candidate_id", "job_id", "match_score"]], explanations], axis=1)
    final = final.sort_values("match_score", ascending=False).head(args.top_k)

    out_path = project_root / config["paths"]["results"] / "sample_rankings.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_path, index=False)

    print(f"Ranked candidates for {job_id}")
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
