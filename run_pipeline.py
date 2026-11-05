"""Entry point: reproduce every table and figure in results/.

    python run_pipeline.py          # full experiment
    python run_pipeline.py --quick  # fast smoke test (writes to results_quick/)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from spamdetect.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
