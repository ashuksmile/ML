import pandas as pd

from src.feature_builder import build_structured_features


def test_structured_features_columns_present():
    df = pd.DataFrame(
        [
            {
                "candidate_skills": "python;sql;machine learning",
                "required_skills": "python;sql",
                "optional_skills": "docker",
                "candidate_years": 4,
                "job_min_years": 2,
                "candidate_education": "master",
                "job_min_education": "bachelor",
            }
        ]
    )
    config = {"matching": {"required_skill_weight": 0.6, "optional_skill_weight": 0.4}}

    out = build_structured_features(df, config)

    expected = {
        "required_overlap",
        "optional_overlap",
        "weighted_skill_fit",
        "experience_fit",
        "education_fit",
        "education_gap",
    }
    assert expected.issubset(set(out.columns))
    assert out.shape[0] == 1
