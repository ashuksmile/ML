"""Import real candidate/job CSVs and standardize to project schema."""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import import_real_data, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates CSV")
    parser.add_argument("--jobs", required=True, help="Path to jobs CSV")
    parser.add_argument("--pairs", default=None, help="Optional labeled pairs CSV")
    args = parser.parse_args()

    config = load_config()
    try:
        c_df, j_df, p_df = import_real_data(
            config,
            candidates_path=args.candidates,
            jobs_path=args.jobs,
            pairs_path=args.pairs,
        )
    except ValueError as exc:
        print("Import validation failed:")
        print(str(exc))
        raise SystemExit(1)

    print("Imported real data and saved standardized files.")
    print(f"  candidates: {len(c_df)}")
    print(f"  jobs: {len(j_df)}")
    print(f"  labeled pairs: {len(p_df)}")


if __name__ == "__main__":
    main()
