"""Command-line entry for the StreamLoop churn tuning study.

The python-hello boilerplate started as ``print("Hello World")`` and told
the reader to replace this file. This is that entry point.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from streamloop.data import load_telco
from streamloop.report import render_summary, write_deliverables
from streamloop.study import PRODUCTION_CONFIG, run_study


def main():
    try:
        frame, meta = load_telco()
    except Exception as exc:
        print(f"error: could not load the official dataset: {exc}", file=sys.stderr)
        return 1
    result = run_study(frame, config=PRODUCTION_CONFIG, dataset_meta=meta)
    write_deliverables(
        result,
        ROOT,
        full_metrics_path=Path("/opt/cursor/artifacts/metrics_full.json"),
    )
    print(render_summary(result), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
