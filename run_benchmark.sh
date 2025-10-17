#!/bin/bash

branch=pr
# branch=$(git rev-parse --abbrev-ref HEAD | tr '/' '_')
mkdir -p results
outfile="results/apply_logs_master.csv"
echo "length,run,time" > $outfile

outfile="results/apply_logs_pr.csv"
echo "length,run,time" > $outfile

for run in {1..5}; do
    for branch in master pr; do
        cd ../optuna
        git checkout $branch
        cd ../examples
        for k in {1..17}; do
            length=$((1<<k))
            result=$(python read_logs.py --log_length $length)
            outfile="results/apply_logs_${branch}.csv"
            echo "$length,$run,$result" >> $outfile
        done
    done
done