mkdir master_with_waiting_trials_100
mkdir fix_with_waiting_trials_100
for i in {1..10}
do
    cd ~/pfn/optuna
    git switch master
    cd ~/pfn/examples
    python optuna_simple.py 100 1 master_with_waiting_trials_100 $i
    cd ~/pfn/optuna
    git switch fix/speed-up-pop-waiting-trial-id-1
    cd ~/pfn/examples
    python optuna_simple.py 100 1 fix_with_waiting_trials_100 $i
done