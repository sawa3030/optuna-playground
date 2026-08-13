from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
import optunahub
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._hypervolume_history import _get_hypervolume_history_info

wfg = optunahub.load_module("benchmarks/wfg")
dtlz_constrained = optunahub.load_module("benchmarks/dtlz_constrained")


def load_study_list(result_dir: Path, label: str) -> list[optuna.Study]:
    path = result_dir / f"{label}.pickle"
    with path.open("rb") as f:
        study_list = pickle.load(f)
    return study_list


def get_style(label: str) -> dict[str, str]:
    if label in {"qlogei", "qLogEI", "fatplus", "fatmax", "qConstrainedLogEHVI-with-hvi"}:
        return {
            "color": "red",
            "marker": "*",
            "ls": "dotted",
            "plot_label": "qConstrainedLogEHVI",
        }
    if label in {"master", "softplus", "max"}:
        return {
            "color": "blue",
            "marker": "s",
            "ls": "dashed",
            "plot_label": "max" if label == "max" else label,
        }
    if label in {"qlogei-128", "relu", "qConstrainedLogEHVI", }:
        return {
            "color": "green",
            "marker": "D",
            "ls": "dashdot",
            "plot_label": "relu" if label == "relu" else label,
        }
    if label in {"qlogei-32", "qConstrainedLogEHVI-without-feasible-check"}:
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

    if "dtlz" in name:
        function_match = re.search(r"_function(\d+)", name)
        objective_match = re.search(r"_n(\d+)", name)
        dimension_match = re.search(r"_d(\d+)", name)
        constraint_match = re.search(r"(?:^|_)c(\d+)(?:_|$)", name)

        benchmark = "DTLZ"
        if function_match:
            benchmark = f"{benchmark}{function_match.group(1)}"
        if constraint_match:
            benchmark = f"C{constraint_match.group(1)}-{benchmark}"

        details = []
        if objective_match:
            details.append(f"{objective_match.group(1)} objectives")
        if dimension_match:
            details.append(f"dim={dimension_match.group(1)}")

        parts.append(benchmark)
        if details:
            parts.append(f"({', '.join(details)})")
    elif "wfg" in name:
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
    problem_module: str,
    function_id: int,
    n_objectives: int,
    dimension: int,
    k: int | None,
    constraint_type: int | None,
):
    if problem_module == "wfg":
        return wfg.Problem(
            function_id=function_id,
            n_objectives=n_objectives,
            dimension=dimension,
            k=k,
        )
    if problem_module == "dtlz_constrained":
        if constraint_type is None:
            raise ValueError("--constraint-type is required for dtlz_constrained.")
        return dtlz_constrained.Problem(
            function_id=function_id,
            n_objectives=n_objectives,
            constraint_type=constraint_type,
            dimension=dimension,
        )
    raise ValueError(f"Unsupported problem_module: {problem_module}")


def _get_problem_attr(
    attrs: dict[str, Any],
    arg_value: Any,
    key: str,
) -> Any:
    if key in attrs:
        return attrs[key]
    return arg_value


def create_problem_from_study(study: optuna.Study, args: argparse.Namespace) -> Any:
    attrs = study.user_attrs
    problem_module = _get_problem_attr(attrs, args.problem_module, "problem_module")
    if problem_module is None:
        problem_module = "wfg"

    function_id = _get_problem_attr(attrs, args.function_id, "function_id")
    n_objectives = _get_problem_attr(attrs, args.n_objectives, "n_objectives")
    dimension = _get_problem_attr(attrs, args.dimension, "dimension")
    k = _get_problem_attr(attrs, args.k, "k")
    constraint_type = _get_problem_attr(attrs, args.constraint_type, "constraint_type")

    if None in (function_id, n_objectives, dimension):
        raise ValueError(
            "Could not determine the problem configuration from the study. "
            "Please regenerate the pickle with the updated simulator or pass "
            "the required CLI arguments."
        )

    return create_problem(
        problem_module=problem_module,
        function_id=int(function_id),
        n_objectives=int(n_objectives),
        dimension=int(dimension),
        k=None if k is None else int(k),
        constraint_type=None if constraint_type is None else int(constraint_type),
    )


def get_hypervolume_history(
    study: optuna.Study,
    reference_point: np.ndarray,
    problem: Any | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not study._is_multi_objective():
        raise ValueError(
            "Hypervolume is only available for multi-objective studies, "
            f"but got {len(study.directions)} objective(s)."
        )

    populate_constraints(study, problem)
    assert_trials_within_reference_point(study, reference_point, problem)
    info = _get_hypervolume_history_info(study, reference_point)
    return np.asarray(info.trial_numbers, dtype=int), np.asarray(info.values, dtype=float)


def is_feasible(trial: FrozenTrial, problem: Any | None) -> bool:
    if len(trial.constraints) > 0:
        return all(x <= 0.0 for x in trial.constraints.values())
    if problem is not None and hasattr(problem, "evaluate_constraints"):
        return all(x <= 0.0 for x in problem.evaluate_constraints(trial.params.copy()))
    return True


def is_dtlz_constrained_problem(problem: Any | None) -> bool:
    return problem is not None and hasattr(problem, "evaluate_constraints")


def populate_constraints(study: optuna.Study, problem: Any | None) -> None:
    if problem is None or not hasattr(problem, "evaluate_constraints"):
        return

    for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
        if len(trial.constraints) > 0:
            continue
        for i, constraint_value in enumerate(problem.evaluate_constraints(trial.params.copy())):
            trial.set_constraint(str(i), float(constraint_value))


def assert_trials_within_reference_point(
    study: optuna.Study,
    reference_point: np.ndarray,
    problem: Any | None,
) -> None:
    if not is_dtlz_constrained_problem(problem):
        return

    for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
        if not is_feasible(trial, problem):
            continue
        values = np.asarray(trial.values, dtype=np.float64)
        if (values > reference_point).any():
            raise AssertionError(
                "Feasible C-DTLZ trial exceeds the fixed reference point [1.0] * M: "
                f"trial={trial.number}, values={trial.values}, "
                f"reference_point={reference_point.tolist()}"
            )


def get_reference_point(
    study_lists: list[list[optuna.Study]],
    problems: list[Any],
) -> np.ndarray:
    first_problem = problems[0]
    if is_dtlz_constrained_problem(first_problem):
        return np.ones(len(study_lists[0][0].directions), dtype=np.float64)*1000

    reference_point = getattr(first_problem, "reference_point", None)
    if reference_point is not None:
        return np.asarray(reference_point, dtype=np.float64)

    raise ValueError(
        "Could not determine the reference point from the problem. "
        "Please store problem metadata with the result."
    )


def aggregate_histories(
    study_list: list[optuna.Study],
    reference_point: np.ndarray,
    problem: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    histories = [get_hypervolume_history(study, reference_point, problem) for study in study_list]
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
        help="Function id. Used only when the pickle does not include problem metadata.",
    )
    parser.add_argument(
        "--n-objectives",
        type=int,
        help="Number of objectives. Used only when the pickle does not include problem metadata.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        help="Search-space dimension. Used only when the pickle does not include problem metadata.",
    )
    parser.add_argument(
        "--k",
        type=int,
        help="WFG position parameter k. Used only when the pickle does not include problem metadata.",
    )
    parser.add_argument(
        "--problem-module",
        choices=("wfg", "dtlz_constrained"),
        help="Problem module. Used only when the pickle does not include problem metadata.",
    )
    parser.add_argument(
        "--constraint-type",
        type=int,
        help="Constraint type for C-DTLZ. Used only when the pickle does not include problem metadata.",
    )
    parser.add_argument(
        "--output",
        default="hypervolume_history.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    _, ax = plt.subplots()
    study_lists = [load_study_list(result_dir, label) for label in args.labels]
    problems = [create_problem_from_study(study_list[0], args) for study_list in study_lists]
    reference_point = get_reference_point(study_lists, problems)

    for label, study_list, problem in zip(args.labels, study_lists, problems):
        trial_numbers, mean, std = aggregate_histories(study_list, reference_point, problem)
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
