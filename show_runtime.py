from __future__ import annotations

import argparse
import pickle
import statistics
from pathlib import Path

import optuna


def load_study_list(result_dir: Path, label: str) -> list[optuna.Study]:
    path = result_dir / f"{label}.pickle"
    with path.open("rb") as f:
        study_list = pickle.load(f)
    if not isinstance(study_list, list):
        raise TypeError(f"{path} does not contain a list of studies")
    return study_list


def get_study_runtime(study: optuna.Study) -> float:
    cumtimes = [
        float(trial.user_attrs["cumtime"])
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and "cumtime" in trial.user_attrs
    ]
    # print("cumtimes:", cumtimes[-10:])
    if not cumtimes:
        raise ValueError("No COMPLETE trials with 'cumtime' found")
    return max(cumtimes)


def find_labels(result_dir: Path) -> list[str]:
    return sorted(path.stem for path in result_dir.glob("*.pickle"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute final runtime (max cumtime) per pickled study and print averages by label."
    )
    parser.add_argument(
        "--result-dir",
        default="./gp_simulator_results_n5_dataset2",
        help="Directory containing pickled study lists.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Target labels (pickle basename). If omitted, all *.pickle in --result-dir are used.",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    labels = args.labels or find_labels(result_dir)

    if not labels:
        raise FileNotFoundError(f"No pickle files found in {result_dir}")

    for label in labels:
        study_list = load_study_list(result_dir, label)
        runtimes = [get_study_runtime(study) for study in study_list]

        mean_runtime = statistics.mean(runtimes)
        min_runtime = min(runtimes)
        max_runtime = max(runtimes)

        print(
            f"{label}: mean={mean_runtime:.3f} sec "
            f"(n={len(runtimes)}, min={min_runtime:.3f}, max={max_runtime:.3f})"
        )


if __name__ == "__main__":
    main()
