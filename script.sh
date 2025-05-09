mkdir master_500
mkdir fix_500

for i in {1..10}
do
    cd ~/pfn/optuna
    git switch master
    cd ~/pfn/examples
    python tell_with_warning.py master_500 $i
    cd ~/pfn/optuna
    git switch avoid-deepcopy
    cd ~/pfn/examples
    python tell_with_warning.py fix_500 $i
done