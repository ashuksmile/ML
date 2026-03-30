"""Train the resume-job match model and persist artifacts."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_config, load_or_create_training_data
from src.feature_builder import build_feature_table
from src.matcher import save_model_artifacts, train_match_model


def main() -> None:
    config = load_config()
    _, _, pairs_df, source = load_or_create_training_data(config)

    feature_df = build_feature_table(pairs_df, config)
    labels = pairs_df["match_label"].values

    processed_dir = project_root / config["paths"]["processed_data"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / "train_pairs.csv"

    training_df = pairs_df.copy()
    for col in feature_df.columns:
        training_df[col] = feature_df[col]
    training_df.to_csv(processed_path, index=False)

    model, metrics = train_match_model(feature_df, labels, config)
    save_model_artifacts(model, feature_df.columns, config, metrics)

    print(f"Data source: {source}")
    print("Model training complete")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
