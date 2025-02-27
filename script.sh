mkdir master_without_waiting_trials_1000
mkdir fix_without_waiting_trials_1000
for i in {1..10}
do
    cd ~/pfn/optuna
    git switch master
    cd ~/pfn/examples
    python optuna_simple.py 0 1000 master_without_waiting_trials_1000 $i
    cd ~/pfn/optuna
    git switch fix/speed-up-pop-waiting-trial-id-1
    cd ~/pfn/examples
    python optuna_simple.py 0 1000 fix_without_waiting_trials_1000 $i
done