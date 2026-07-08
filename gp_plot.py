from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import re

import matplotlib.pyplot as plt
import optuna
import optunahub
import random


plot_target_over_time = optunahub.load_module(
    "visualization/plot_target_over_time"
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
    if label == "master":
        return {
            "color": "blue",
            "marker": "s",
            "ls": "dashed",
            "plot_label": "master",
        }
    elif label == "qConstrainedLogEI":
        return {
            "color": "green",
            "marker": "D",
            "ls": "dashdot",
            # "plot_label": "qLogEI (n_qmc_samples=128)",
            "plot_label": "qConstrainedLogEI",
        }
    elif label == "qConstrainedLogEI-tau0-01":
        return {
            "color": "red",
            "marker": "P",
            "ls": "dashdot",
            "plot_label": "qConstrainedLogEI (tau=0.01)",
        }
    elif label == "qConstrainedLogEI-tau0-1":
        return {
            "color": "purple",
            "marker": "X",
            "ls": "dashdot",
            "plot_label": "qConstrainedLogEI (tau=0.1)",
        }
    elif label == "qConstrainedLogEI-tau1":
        return {
            "color": "brown",
            "marker": "D",
            "ls": "dashdot",
            "plot_label": "qConstrainedLogEI (tau=1.0)",
        }
    elif label == "qConstrainedLogEI-tau10":
        return {
            "color": "black",
            "marker": "s",
            "ls": "dashdot",
            "plot_label": "qConstrainedLogEI (tau=10.0)",
        }
    elif label == "qConstrainedLogEI-tau100":
        return {
            "color": "orange",
            "marker": "*",
            "ls": "dashdot",
            "plot_label": "qConstrainedLogEI (tau=100.0)",
        }
    elif label == "qConstrainedLogEI-tau0-001":
        return {
            "color": "orange",
            "marker": "o",
            "ls": "dashdot",
            "plot_label": "qConstrainedLogEI (tau=0.001)",
        }
    else:
        color = random.choice(["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"])
        return {
            "color": color,
            "marker": "o",
            "ls": "solid",
            "plot_label": label,
        }


def build_plot_title(result_dir: Path) -> str:
    # return f"objective: cos(2*x)*cos(y) + sin(x), constraint: cos(x)*cos(y) - sin(x)*sin(y) - 0.5 <= 0"
    return f"objective: sin(x) + y, constraint: sin(x)*sin(y) + 0.95 <= 0"


def get_trial_count_for_plot(trial: optuna.trial.FrozenTrial) -> float:
    return trial.user_attrs.get("trial_num")


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
            cumtime_func=get_trial_count_for_plot,
            label=style["plot_label"],
            marker=style["marker"],
            ls=style["ls"],
            markevery=10,
        )

    ax.grid(True, which="major", alpha=0.5)
    ax.grid(True, which="minor", alpha=0.2)
    ax.set_xlabel("Number of trials")
    ax.set_ylabel("Best value so far")
    ax.set_title(build_plot_title(result_dir))
    ax.legend()

    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
