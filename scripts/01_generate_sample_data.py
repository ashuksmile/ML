"""Generate synthetic candidate, job, and labeled pair data."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import generate_sample_data, load_config, save_raw_data


def main() -> None:
    config = load_config()
    candidates, jobs, pairs = generate_sample_data(config)
    save_raw_data(config, candidates, jobs, pairs)

    print("Generated sample data:")
    print(f"  candidates: {len(candidates)}")
    print(f"  jobs: {len(jobs)}")
    print(f"  pairs: {len(pairs)}")


if __name__ == "__main__":
    main()
