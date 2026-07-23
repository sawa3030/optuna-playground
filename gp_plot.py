from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import re

import matplotlib.pyplot as plt
import optuna
import optunahub


plot_target_over_time = optunahub.load_local_module(
    package = "visualization/plot_target_over_time",
    registry_root = "/home/eri/pfn/optunahub-registry/package",
    # "visualization/plot_target_over_time"
).plot_target_over_time


def load_study_list(result_dir: Path, label: str) -> list[optuna.Study]:
    path = result_dir / f"{label}.pickle"
    with path.open("rb") as f:
        study_list = pickle.load(f)
    return study_list


def trim_to_same_length(study_list: list[optuna.Study]) -> list[optuna.Study]:
    min_len = min(len(study.trials) for study in study_list)

    trimmed = []
    for study in study_list:
        new_study = optuna.create_study(directions=study.directions)
        new_study.add_trials(study.trials[:min_len])
        trimmed.append(new_study)

    return trimmed


def get_style(label: str) -> dict[str, str]:
    if label == "qlogei" or label == "qLogEI" or label == "fatplus" or label == "fatmax":
        return {
            "color": "red",
            "marker": "*",
            "ls": "dotted",
            # "plot_label": "qLogEI (n_qmc_samples=512)",
            "plot_label": "fatplus",
        }
    elif label == "master" or label == "softplus" or label == "max":
        return {
            "color": "blue",
            "marker": "s",
            "ls": "dashed",
            # "plot_label": "master",
            "plot_label": "softplus",
        }
    elif label == "qlogei-128" or label == "relu":
        return {
            "color": "green",
            "marker": "D",
            "ls": "dashdot",
            # "plot_label": "qLogEI (n_qmc_samples=128)",
            "plot_label": "relu",
        }
    elif label == "qlogei-32":
        return {
            "color": "orange",
            "marker": "P",
            "ls": "dashdot",
            "plot_label": "qLogEI (n_qmc_samples=32)",
        }
    elif label == "qlogei-64":
        return {
            "color": "purple",
            "marker": "X",
            "ls": "dashdot",
            "plot_label": "qLogEI (n_qmc_samples=64)",
        }
    else:
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
        help="Labels to load, e.g. v4.7.0 add-qlogei master",
    )
    parser.add_argument(
        "--output",
        default="async-bench-example_without_evaltime_n5_bbo_dataset10.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)

    _, ax = plt.subplots()

    for label in args.labels:
        study_list = load_study_list(result_dir, label)
        study_list = trim_to_same_length(study_list)

        style = get_style(label)

        plot_target_over_time(
            study_list,
            color=style["color"],
            ax=ax,
            cumtime_func=lambda t: max(t.user_attrs["cumtime"], 1e-12),
            label=style["plot_label"],
            marker=style["marker"],
            ls=style["ls"],
            markevery=10,
        )

    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.5)
    ax.grid(True, which="minor", alpha=0.2)
    ax.set_xlabel("Simulated cumulative time")
    ax.set_ylabel("Best value so far")
    ax.set_title(build_plot_title(result_dir))
    ax.legend()

    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
