mkdir master_without_waiting_trials_100
mkdir fix_without_waiting_trials_100
for i in {1..10}
do
    cd ~/pfn/optuna
    git switch master
    cd ~/pfn/examples
    python optuna_simple.py 0 100 master_without_waiting_trials_100 $i
    cd ~/pfn/optuna
    git switch fix/speed-up-pop-waiting-trial-id-1
    cd ~/pfn/examples
    python optuna_simple.py 0 100 fix_without_waiting_trials_100 $i
done