mkdir master_5000
mkdir fix_5000

for i in {1..10}
do
    # cd ~/pfn/optuna
    # git switch master
    # cd ~/pfn/examples
    python test_cmaes.py master_5000 $i
    # cd ~/pfn/optuna
    # git switch avoid-deepcopy
    # cd ~/pfn/examples
    python test_cmaes.py fix_5000 $i
done