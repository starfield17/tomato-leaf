from __future__ import annotations

import argparse
from pathlib import Path

from report_utils import ExperimentRecord, generate_report_artifacts, load_experiment_record
from utils import ensure_dir

DEFAULT_EXPERIMENTS = (
    Path("outputs/simplecnn_color_e20"),
    Path("outputs/fastcnn_color_e20"),
    Path("outputs/resnet18_color_e20"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate experiment comparison charts and Markdown report.")
    parser.add_argument("--experiments", type=Path, nargs="*", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiment_results"))
    return parser.parse_args()


def load_records(*, experiment_dirs: list[Path]) -> list[ExperimentRecord]:
    return [load_experiment_record(experiment_dir=path) for path in experiment_dirs]


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(path=args.output_dir)
    records = load_records(experiment_dirs=list(args.experiments))
    generate_report_artifacts(records=records, output_dir=output_dir)
    print(f"Generated report artifacts in {output_dir}")


if __name__ == "__main__":
    main()
