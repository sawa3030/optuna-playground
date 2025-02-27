mkdir master_only_waiting_trials_1000
mkdir fix_only_waiting_trials_1000
for i in {1..10}
do
    cd ~/pfn/optuna
    git switch master
    cd ~/pfn/examples
    python optuna_simple.py 1000 1 master_only_waiting_trials_1000 $i
    cd ~/pfn/optuna
    git switch fix/speed-up-pop-waiting-trial-id-1
    cd ~/pfn/examples
    python optuna_simple.py 1000 1 fix_only_waiting_trials_1000 $i
done