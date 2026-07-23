cd ~/pfn/optuna
git switch master

source ~/pfn/optuna/venv/bin/
cd ~/pfn/optuna-playground
# python gp_simulate_for_non_evaluation_time.py --label master --out-dir ./qlogehvi_wfg_function1_n2_d3_k1 --function-id 1 --n-objectives 2 --dimension 3 --k 1
# python gp_simulate_for_non_evaluation_time.py --label master --out-dir ./qlogehvi_wfg_function2_n2_d3_k1 --function-id 2 --n-objectives 2 --dimension 3 --k 1
# python gp_simulate_for_non_evaluation_time.py --label master --out-dir ./qlogehvi_wfg_function3_n2_d3_k1 --function-id 3 --n-objectives 2 --dimension 3 --k 1
# python gp_simulate_for_non_evaluation_time.py --label master --out-dir ./qlogehvi_wfg_function4_n2_d3_k1 --function-id 4 --n-objectives 2 --dimension 3 --k 1

# cd ~/pfn/optuna
# git switch add-qlogehvi-4

# source ~/pfn/optuna/venv/bin/
# cd ~/pfn/optuna-playground
# python gp_simulate_for_non_evaluation_time.py --label qLogEHVI-4-128 --out-dir ./qlogehvi_wfg_function1_n2_d3_k1 --function-id 1 --n-objectives 2 --dimension 3 --k 1
# python gp_simulate_for_non_evaluation_time.py --label qLogEHVI-4-128 --out-dir ./qlogehvi_wfg_function2_n2_d3_k1 --function-id 2 --n-objectives 2 --dimension 3 --k 1
# python gp_simulate_for_non_evaluation_time.py --label qLogEHVI-4-128 --out-dir ./qlogehvi_wfg_function3_n2_d3_k1 --function-id 3 --n-objectives 2 --dimension 3 --k 1
# python gp_simulate_for_non_evaluation_time.py --label qLogEHVI-4-128 --out-dir ./qlogehvi_wfg_function4_n2_d3_k1 --function-id 4 --n-objectives 2 --dimension 3 --k 1


source ~/pfn/optuna/venv/bin/
cd ~/pfn/optuna-playground
python plot_hypervolume.py --result-dir ./qlogehvi_wfg_function1_n2_d3_k1 --labels master qLogEHVI-4-128 --output qlogehvi_wfg_function1_n2_d3_k1.png
python plot_hypervolume.py --result-dir ./qlogehvi_wfg_function2_n2_d3_k1 --labels master qLogEHVI-4-128 --output qlogehvi_wfg_function2_n2_d3_k1.png
python plot_hypervolume.py --result-dir ./qlogehvi_wfg_function3_n2_d3_k1 --labels master qLogEHVI-4-128 --output qlogehvi_wfg_function3_n2_d3_k1.png
python plot_hypervolume.py --result-dir ./qlogehvi_wfg_function4_n2_d3_k1 --labels master qLogEHVI-4-128 --output qlogehvi_wfg_function4_n2_d3_k1.png

