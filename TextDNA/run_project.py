"""Run full HireSense AI pipeline."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    project_root = Path(__file__).parent
    steps = [
        project_root / "scripts" / "02_train_match_engine.py",
        project_root / "scripts" / "03_rank_candidates.py",
    ]

    for step in steps:
        runpy.run_path(str(step), run_name="__main__")
