from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.stats import mannwhitneyu


JSONL_PATH = Path("benchmark_results/details.jsonl")

N_TRIALS_LIST = [25, 50, 75, 100]
BATCH_SIZES = [5, 10, 50]
ALPHA = 0.05

def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON at line {line_no}: {e}") from e


def one_sided_mwu_j_better(
    xj: np.ndarray,
    xk: np.ndarray,
    *,
    direction: str,
    alpha: float,
) -> bool:
    if len(xj) != 100 or len(xk) != 100:
        raise RuntimeError("Expected 100 samples for each group")

    # SciPy: mannwhitneyu(x, y, alternative='less') tests whether x is stochastically smaller than y.
    if direction == "min":
        alt = "less"     # j < k なら j が良い
    elif direction == "max":
        alt = "greater"  # j > k なら j が良い

    stat = mannwhitneyu(xj, xk, alternative=alt)
    return stat.pvalue < alpha


def main() -> None:
    # group[(type, n_trials, batch_size)][liar] = list of best_values
    group: Dict[Tuple[str, int, int], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for rec in iter_jsonl(JSONL_PATH):
        btype = rec["benchmark"]
        if not btype.startswith("bbob:") and not btype.startswith("hpobench"):
            continue

        n_trials = int(rec["n_trials"])
        batch_size = int(rec["batch_size"])
        if n_trials not in N_TRIALS_LIST or batch_size not in BATCH_SIZES:
            continue

        liar = "none" if rec.get("constant_liar") is None else str(rec.get("constant_liar"))
        best = float(rec["best_value"])
        group[(btype, n_trials, batch_size)][liar].append(best)

    # For each key, compute wins matrix among liars
    # wins[(type, n_trials, batch_size)][liar_j] = wins_count
    results: Dict[Tuple[str, int, int], Dict[str, int]] = {}

    for key, liar_to_vals in sorted(group.items()):
        btype, n_trials, batch_size = key
        direction = "min" if btype.startswith("bbob:") else "max"

        liars = sorted(liar_to_vals.keys())
        wins = {lj: 0 for lj in liars}

        for j in liars:
            xj = np.asarray(liar_to_vals[j], dtype=float)
            for k in liars:
                if j == k:
                    continue
                xk = np.asarray(liar_to_vals[k], dtype=float)
                if one_sided_mwu_j_better(xj, xk, direction=direction, alpha=ALPHA):
                    if j == "none":
                        print(f"[DEBUG] {key}: 'none' better than '{k}'")
                    # if k != "none":
                    #     continue
                    wins[j] += 1

        results[key] = wins

    # Pretty print as tables
    # Table rows: (type, n_trials, batch_size), cols: liar strategy
    all_liars = sorted({lj for wins in results.values() for lj in wins.keys()})

    header = ["benchmark_type", "n_trials", "batch_size"] + [f"wins:{lj}" for lj in all_liars]
    print("\t".join(header))



    for (btype, n_trials, batch_size), wins in results.items():
        row = [btype, str(n_trials), str(batch_size)]
        for lj in all_liars:
            row.append(str(wins.get(lj, 0)))
        print("\t".join(row))

    # もしCSV保存したいなら：
    out_csv = JSONL_PATH.parent / "wins_mwu.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for (btype, n_trials, batch_size), wins in results.items():
            row = [btype, str(n_trials), str(batch_size)] + [str(wins.get(lj, 0)) for lj in all_liars]
            f.write(",".join(row) + "\n")
    print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    main()
