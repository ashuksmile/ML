# HireSense AI: Resume-to-Job Match Engine

HireSense AI is a practical ML project for recruiters. It ranks candidates for a job role using a combination of semantic similarity and structured fit signals, then explains why each candidate got the score.

## Why this project is useful

- Reduces manual screening effort by ranking top-fit candidates.
- Makes decisions transparent with matched skills and missing skills.
- Works as a base for ATS-style shortlist generation.

## Core capabilities

- Parse resumes and job descriptions from structured CSV data.
- Build feature vectors using:
  - TF-IDF semantic similarity
  - Skill overlap
  - Experience alignment
  - Education alignment
- Train a supervised match classifier.
- Convert model output to recruiter-friendly score (0-100).
- Produce explanations for each candidate-job pair.

## Tech stack

- Python
- Pandas, NumPy
- Scikit-learn
- PyYAML
- Joblib
- Streamlit

## Quickstart

```bash
pip install -r requirements.txt
python run_project.py
```

## Use Your Own Data (Recommended)

Import real recruiter datasets:

```bash
python scripts/00_import_real_data.py --candidates path/to/candidates.csv --jobs path/to/jobs.csv
```

Optional labeled pairs file (for supervised ground truth):

```bash
python scripts/00_import_real_data.py --candidates path/to/candidates.csv --jobs path/to/jobs.csv --pairs path/to/pairs.csv
```

Then train and rank:

```bash
python run_project.py
```

If only candidates and jobs are provided, the project auto-creates weak labels for training.
The import step now performs strict validation and reports row-level errors for invalid numeric or education fields.

### Expected Columns

Candidates CSV supports (with alias mapping):
- candidate_id, name, years_experience, education_level, skills, resume_text

Jobs CSV supports (with alias mapping):
- job_id, job_title, min_years, min_education, required_skills, optional_skills, job_text

Pairs CSV (optional):
- candidate_id, job_id, match_label

Then launch dashboard:

```bash
streamlit run app/streamlit_app.py
```

## Pipeline outputs

- `data/processed/train_pairs.csv`
- `models/match_model.joblib`
- `results/metrics_summary.json`
- `results/sample_rankings.csv`

## Project structure

```text
TextDNA/
  app/
    streamlit_app.py
  config/
    config.yaml
  data/
    raw/
    processed/
  models/
  results/
  scripts/
    00_import_real_data.py
    01_generate_sample_data.py
    02_train_match_engine.py
    03_rank_candidates.py
  src/
    __init__.py
    data_loader.py
    feature_builder.py
    matcher.py
    explain.py
  run_project.py
```

## Resume-ready summary

- Built a resume-to-job match engine to rank candidates using NLP and ML.
- Combined semantic similarity with structured signals (skills, experience, education).
- Implemented explainable scoring with matched and missing skill insights.
- Trained and evaluated a classifier for shortlist quality.
- Delivered an interactive dashboard for recruiter-facing candidate ranking.
