# cd ~/pfn/optuna
# git switch master

# source ~/pfn/optuna/venv/bin/activate
# cd ~/pfn/optuna-playground
# python gp_simulate_for_non_evaluation_time.py --label master --out-dir ./q_constrained_logehvi_c1_dtlz_function1_n2_d3 --problem-module dtlz_constrained --function-id 1 --constraint-type 1 --n-objectives 2 --dimension 3
# python gp_simulate_for_non_evaluation_time.py --label master --out-dir ./q_constrained_logehvi_c1_dtlz_function3_n2_d3 --problem-module dtlz_constrained --function-id 3 --constraint-type 1 --n-objectives 2 --dimension 3
# python gp_simulate_for_non_evaluation_time.py --label master --out-dir ./q_constrained_logehvi_c2_dtlz_function2_n2_d3 --problem-module dtlz_constrained --function-id 2 --constraint-type 2 --n-objectives 2 --dimension 3
# python gp_simulate_for_non_evaluation_time.py --label master --out-dir ./q_constrained_logehvi_c3_dtlz_function1_n2_d3 --problem-module dtlz_constrained --function-id 1 --constraint-type 3 --n-objectives 2 --dimension 3
# python gp_simulate_for_non_evaluation_time.py --label master --out-dir ./q_constrained_logehvi_c3_dtlz_function4_n2_d3 --problem-module dtlz_constrained --function-id 4 --constraint-type 3 --n-objectives 2 --dimension 3

cd ~/pfn/optuna
git switch add-constrained-qlogehvi
source ~/pfn/optuna/venv/bin/activate
cd ~/pfn/optuna-playground
python gp_simulate_for_non_evaluation_time.py --label qConstrainedLogEHVI-without-feasible-check --out-dir ./q_constrained_logehvi_c1_dtlz_function1_n2_d3 --problem-module dtlz_constrained --function-id 1 --constraint-type 1 --n-objectives 2 --dimension 3
python gp_simulate_for_non_evaluation_time.py --label qConstrainedLogEHVI-without-feasible-check --out-dir ./q_constrained_logehvi_c1_dtlz_function3_n2_d3 --problem-module dtlz_constrained --function-id 3 --constraint-type 1 --n-objectives 2 --dimension 3
python gp_simulate_for_non_evaluation_time.py --label qConstrainedLogEHVI-without-feasible-check --out-dir ./q_constrained_logehvi_c2_dtlz_function2_n2_d3 --problem-module dtlz_constrained --function-id 2 --constraint-type 2 --n-objectives 2 --dimension 3
python gp_simulate_for_non_evaluation_time.py --label qConstrainedLogEHVI-without-feasible-check --out-dir ./q_constrained_logehvi_c3_dtlz_function1_n2_d3 --problem-module dtlz_constrained --function-id 1 --constraint-type 3 --n-objectives 2 --dimension 3
python gp_simulate_for_non_evaluation_time.py --label qConstrainedLogEHVI-without-feasible-check --out-dir ./q_constrained_logehvi_c3_dtlz_function4_n2_d3 --problem-module dtlz_constrained --function-id 4 --constraint-type 3 --n-objectives 2 --dimension 3

# python plot_hypervolume.py --result-dir ./q_constrained_logehvi_c1_dtlz_function1_n2_d3 --labels master qConstrainedLogEHVI qConstrainedLogEHVI-without-feasible-check --output qlogehvi_c1_dtlz_function1_n2_d3.png
# python plot_hypervolume.py --result-dir ./q_constrained_logehvi_c1_dtlz_function3_n2_d3 --labels master qConstrainedLogEHVI qConstrainedLogEHVI-without-feasible-check --output qlogehvi_c1_dtlz_function3_n2_d3.png
# python plot_hypervolume.py --result-dir ./q_constrained_logehvi_c2_dtlz_function2_n2_d3 --labels master qConstrainedLogEHVI qConstrainedLogEHVI-without-feasible-check --output qlogehvi_c2_dtlz_function2_n2_d3.png
# python plot_hypervolume.py --result-dir ./q_constrained_logehvi_c3_dtlz_function1_n2_d3 --labels master qConstrainedLogEHVI qConstrainedLogEHVI-without-feasible-check --output qlogehvi_c3_dtlz_function1_n2_d3.png
# python plot_hypervolume.py --result-dir ./q_constrained_logehvi_c3_dtlz_function4_n2_d3 --labels master qConstrainedLogEHVI qConstrainedLogEHVI-without-feasible-check --output qlogehvi_c3_dtlz_function4_n2_d3.png



# source ~/pfn/optuna/venv/bin/
# cd ~/pfn/optuna-playground
# python plot_hypervolume.py --result-dir ./qlogehvi_wfg_function1_n2_d3_k1 --labels master qLogEHVI-3-128 --output qlogehvi_wfg_function1_n4_d5_k3.png
# python plot_hypervolume.py --result-dir ./qlogehvi_wfg_function2_n2_d3_k1 --labels master qLogEHVI-4-128 qLogEHVI-3-128 --output qlogehvi_wfg_function2_n2_d3_k1.png
# python plot_hypervolume.py --result-dir ./qlogehvi_wfg_function3_n2_d3_k1 --labels master qLogEHVI-4-128 qLogEHVI-3-128 --output qlogehvi_wfg_function3_n2_d3_k1.png
# python plot_hypervolume.py --result-dir ./qlogehvi_wfg_function4_n2_d3_k1 --labels master qLogEHVI-4-128 qLogEHVI-3-128 --output qlogehvi_wfg_function4_n2_d3_k1.png
