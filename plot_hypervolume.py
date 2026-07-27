from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import optunahub
from optuna.visualization._hypervolume_history import _get_hypervolume_history_info

wfg = optunahub.load_module("benchmarks/wfg")


def load_study_list(result_dir: Path, label: str) -> list[optuna.Study]:
    path = result_dir / f"{label}.pickle"
    with path.open("rb") as f:
        study_list = pickle.load(f)
    return study_list


def get_style(label: str) -> dict[str, str]:
    if label in {"qlogei", "qLogEI", "fatplus", "fatmax", "qLogEHVI-3-128"}:
        return {
            "color": "red",
            "marker": "*",
            "ls": "dotted",
            "plot_label": "fatmax" if label == "fatmax" else label,
        }
    if label in {"master", "softplus", "max"}:
        return {
            "color": "blue",
            "marker": "s",
            "ls": "dashed",
            "plot_label": "max" if label == "max" else label,
        }
    if label in {"qlogei-128", "relu"}:
        return {
            "color": "green",
            "marker": "D",
            "ls": "dashdot",
            "plot_label": "relu" if label == "relu" else label,
        }
    if label == "qlogei-32":
        return {
            "color": "orange",
            "marker": "P",
            "ls": "dashdot",
            "plot_label": label,
        }
    if label == "qlogei-64":
        return {
            "color": "purple",
            "marker": "X",
            "ls": "dashdot",
            "plot_label": label,
        }
    return {
        "color": "black",
        "marker": "o",
        "ls": "solid",
        "plot_label": label,
    }


def build_plot_title(result_dir: Path) -> str:
    name = result_dir.name.lower()
    parts: list[str] = []

    if "wfg" in name:
        function_match = re.search(r"_function(\d+)", name)
        objective_match = re.search(r"_n(\d+)", name)
        dimension_match = re.search(r"_d(\d+)", name)
        k_match = re.search(r"_k(\d+)", name)

        benchmark = "WFG"
        if function_match:
            benchmark += f" Function {function_match.group(1)}"

        details = []
        if objective_match:
            details.append(f"{objective_match.group(1)} objectives")
        if dimension_match:
            details.append(f"dim={dimension_match.group(1)}")
        if k_match:
            details.append(f"k={k_match.group(1)}")

        parts.append(benchmark)
        if details:
            parts.append(f"({', '.join(details)})")
    else:
        dataset_match = re.search(r"dataset(\d+)", name)
        dataset_id = dataset_match.group(1) if dataset_match else "unknown"

        if "bbo" in name or "bbob" in name:
            parts.append(f"BBOB Dataset {dataset_id}")
        else:
            parts.append(f"HPOBench Dataset {dataset_id}")

    worker_match = re.search(r"(?:without_evaltime_|results_)n(\d+)(?:_|$)", name)
    if worker_match:
        parts.append(f"{worker_match.group(1)} workers")

    if "without_evaltime" in name:
        parts.append("without evaluation time")

    if not parts:
        return result_dir.name

    return " | ".join(parts)


def create_problem(
    function_id: int,
    n_objectives: int,
    dimension: int,
    k: int,
):
    return wfg.Problem(
        function_id=function_id,
        n_objectives=n_objectives,
        dimension=dimension,
        k=k,
    )


def get_reference_point(study_list: list[optuna.Study], args: argparse.Namespace) -> np.ndarray:
    attrs = study_list[0].user_attrs

    if attrs.get("problem_module") == "wfg":
        problem = create_problem(
            function_id=int(attrs["function_id"]),
            n_objectives=int(attrs["n_objectives"]),
            dimension=int(attrs["dimension"]),
            k=int(attrs["k"]),
        )
        return np.asarray(problem.reference_point, dtype=np.float64)

    if None not in (args.function_id, args.n_objectives, args.dimension, args.k):
        problem = create_problem(
            function_id=args.function_id,
            n_objectives=args.n_objectives,
            dimension=args.dimension,
            k=args.k,
        )
        return np.asarray(problem.reference_point, dtype=np.float64)

    raise ValueError(
        "Could not determine the WFG problem configuration from the study. "
        "Please regenerate the pickle with the updated simulator or pass "
        "--function-id, --n-objectives, --dimension, and --k."
    )


def get_hypervolume_history(
    study: optuna.Study, reference_point: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not study._is_multi_objective():
        raise ValueError(
            "Hypervolume is only available for multi-objective studies, "
            f"but got {len(study.directions)} objective(s)."
        )

    info = _get_hypervolume_history_info(study, reference_point)
    return np.asarray(info.trial_numbers, dtype=int), np.asarray(info.values, dtype=float)


def aggregate_histories(
    study_list: list[optuna.Study], reference_point: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    histories = [get_hypervolume_history(study, reference_point) for study in study_list]
    all_trial_numbers = sorted({int(trial) for trials, _ in histories for trial in trials})
    trial_to_index = {trial_number: i for i, trial_number in enumerate(all_trial_numbers)}

    values = np.full((len(histories), len(all_trial_numbers)), np.nan, dtype=float)
    for row, (trial_numbers, hv_values) in enumerate(histories):
        for trial_number, hv_value in zip(trial_numbers, hv_values):
            values[row, trial_to_index[int(trial_number)]] = hv_value

    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    return np.asarray(all_trial_numbers, dtype=int), mean, std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        default="./gp_simulator_results_without_evaltime_n5_bbo_dataset10",
        help="Directory containing pickled study lists.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Labels to load, e.g. method-a method-b",
    )
    parser.add_argument(
        "--function-id",
        type=int,
        help="WFG function id. Used only when the pickle does not include problem metadata.",
    )
    parser.add_argument(
        "--n-objectives",
        type=int,
        help="Number of WFG objectives. Used only when the pickle does not include problem metadata.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        help="WFG search-space dimension. Used only when the pickle does not include problem metadata.",
    )
    parser.add_argument(
        "--k",
        type=int,
        help="WFG position parameter k. Used only when the pickle does not include problem metadata.",
    )
    parser.add_argument(
        "--output",
        default="hypervolume_history.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    _, ax = plt.subplots()

    for label in args.labels:
        study_list = load_study_list(result_dir, label)
        reference_point = get_reference_point(study_list, args)
        trial_numbers, mean, std = aggregate_histories(study_list, reference_point)
        style = get_style(label)

        ax.plot(
            trial_numbers,
            mean,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["ls"],
            markevery=max(1, len(trial_numbers) // 10),
            label=style["plot_label"],
        )
        ax.fill_between(
            trial_numbers,
            mean - std,
            mean + std,
            color=style["color"],
            alpha=0.2,
        )

    ax.grid(True, which="major", alpha=0.5)
    ax.grid(True, which="minor", alpha=0.2)
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Hypervolume")
    ax.set_title(build_plot_title(result_dir))
    ax.legend()

    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
