"""The README results block must equal what the committed result files render."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import report


def test_readme_results_match_result_files():
    assert report.current() == report.render()
