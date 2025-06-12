mkdir master_1000
mkdir fix_1000

for i in {1..5}
do
    cd ~/pfn/optuna
    git switch master
    cd ~/pfn/examples
    python tell_with_warning.py master_1000 $i
    cd ~/pfn/optuna
    git switch remove-the-last-get-trial-in-tell-with-warning
    cd ~/pfn/examples
    python tell_with_warning.py fix_1000 $i
done