#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
READ_LOGS_PY="${SCRIPT_DIR}/read_logs.py"
OPTUNA_DIR="${SCRIPT_DIR}/../optuna"

mkdir -p "${RESULTS_DIR}"
outfile="${RESULTS_DIR}/apply_logs_master.csv"
echo "length,run,time" > "${outfile}"

outfile="${RESULTS_DIR}/apply_logs_pr.csv"
echo "length,run,time" > "${outfile}"

for run in {1..5}; do
    for branch in master pr; do
        cd "${OPTUNA_DIR}"
        git checkout "${branch}"
        cd "${SCRIPT_DIR}"
        for k in {1..17}; do
            length=$((1<<k))
            result=$(PYTHONPATH="${OPTUNA_DIR}:${PYTHONPATH:-}" python "${READ_LOGS_PY}" --log_length "${length}")
            outfile="${RESULTS_DIR}/apply_logs_${branch}.csv"
            echo "$length,$run,$result" >> "${outfile}"
        done
    done
done
