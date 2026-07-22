source ~/pfn/optuna/venv/bin/
cd ~/pfn/optuna-playground

# python gp_simulate_for_non_evaluation_time.py --label master --out-dir ./qlogehvi_wfg_function4_n2_d3_k1
python gp_simulate_for_non_evaluation_time.py --label qLogEHVI-4-2048 --out-dir ./qlogehvi_wfg_function4_n2_d3_k1
# python gp_simulate_for_non_evaluation_time.py --label master --out-dir ./constrained_trial_num_2
# python gp_simulate_for_non_evaluation_time.py --label master --out-dir ./constrained_trial_num_1
# python gp_simulate_for_non_evaluation_time.py --label qConstrainedLogEI-tau0-001 --out-dir ./constrained_trial_num_1 --tau 0.001 --use_qmc True
# python gp_simulate_for_non_evaluation_time.py --label qConstrainedLogEI-tau0-01 --out-dir ./constrained_trial_num_1 --tau 0.01 --use_qmc True
# python gp_simulate_for_non_evaluation_time.py --label qConstrainedLogEI-tau0-1 --out-dir ./constrained_trial_num_1 --tau 0.1 --use_qmc True
# python gp_simulate_for_non_evaluation_time.py --label qConstrainedLogEI-tau1 --out-dir ./constrained_trial_num_1 --tau 1.0 --use_qmc True
# python gp_simulate_for_non_evaluation_time.py --label qConstrainedLogEI-tau10 --out-dir ./constrained_trial_num_1 --tau 10.0 --use_qmc True
# python gp_simulate_for_non_evaluation_time.py --label qConstrainedLogEI-tau100 --out-dir ./constrained_trial_num_1 --tau 100 --use_qmc True


# python gp_simulate_for_non_evaluation_time.py --label max --out-dir ./maxoperator_without_evaltime_n5_dataset0 --dataset-id 0
# python gp_simulate_for_non_evaluation_time.py --label max --out-dir ./maxoperator_without_evaltime_n5_dataset1 --dataset-id 1
# python gp_simulate_for_non_evaluation_time.py --label max --out-dir ./maxoperator_without_evaltime_n5_dataset2 --dataset-id 2

# python gp_simulate_for_non_evaluation_time.py --label qLogEHVI --out-dir ./maxoperator_without_evaltime_n5_bbo_dataset1 --dataset-id 1
# python gp_simulate_for_non_evaluation_time.py --label KB --out-dir ./multi_without_evaltime_n5_wfg_dataset10 --dataset-id 10
# python gp_simulate_for_non_evaluation_time.py --label qLogEHVI --out-dir ./maxoperator_without_evaltime_n5_bbo_dataset10 --dataset-id 10

# python gp_simulator.py --label qlogei-128 --out-dir ./gp_simulator_results_without_evaltime_n5_dataset0 --dataset-id 0
# python gp_simulator.py --label qlogei-128 --out-dir ./gp_simulator_results_without_evaltime_n5_dataset1 --dataset-id 1
# python gp_simulator.py --label qlogei --out-dir ./gp_simulator_results_without_evaltime_n5_dataset2 --dataset-id 2

# python gp_plot.py --labels qlogei-64 qlogei-128 qlogei master --result-dir ./gp_simulator_results_without_evaltime_n5_bbo_dataset15 --output async-bench-example_without_evaltime_n5_bbo_dataset15.png
# python gp_plot.py --labels qlogei-64 qlogei-128 qlogei master --result-dir ./gp_simulator_results_without_evaltime_n5_bbo_dataset10 --output async-bench-example_without_evaltime_n5_bbo_dataset10.png
# python gp_plot.py --labels qlogei-64 qlogei-128 qlogei master --result-dir ./gp_simulator_results_without_evaltime_n5_bbo_dataset6 --output async-bench-example_without_evaltime_n5_bbo_dataset6.png
# python gp_plot.py --labels qlogei-64 qlogei-128 qlogei master --result-dir ./gp_simulator_results_without_evaltime_n5_bbo_dataset1 --output async-bench-example_without_evaltime_n5_bbo_dataset1.png

# python gp_plot.py --labels fatmax max --result-dir ./maxoperator_without_evaltime_n5_dataset0 --output maxoperator_without_evaltime_n5_dataset0.png
# python gp_plot.py --labels fatmax max --result-dir ./maxoperator_without_evaltime_n5_dataset1 --output maxoperator_without_evaltime_n5_dataset1.png
# python gp_plot.py --labels fatmax max --result-dir ./maxoperator_without_evaltime_n5_dataset2 --output maxoperator_without_evaltime_n5_dataset2.png

# python gp_plot.py --labels fatmax max --result-dir ./maxoperator_without_evaltime_n5_bbo_dataset1 --output maxoperator_without_evaltime_n5_bbo_dataset1.png
# python gp_plot.py --labels qLogEHVI KB --result-dir ./multi_without_evaltime_n5_wfg_dataset10 --output multi_without_evaltime_n5_wfg_dataset10.png
# python gp_plot.py --labels fatmax max --result-dir ./maxoperator_without_evaltime_n5_bbo_dataset10 --output maxoperator_without_evaltime_n5_bbo_dataset10.png

# python gp_plot.py --labels qlogei-64 qlogei-128 qlogei master --result-dir ./gp_simulator_results_without_evaltime_n5_dataset0 --output async-bench-example_without_evaltime_n5_dataset0.png
# python gp_plot.py --labels qlogei-64 qlogei-128 qlogei master --result-dir ./gp_simulator_results_without_evaltime_n5_dataset1 --output async-bench-example_without_evaltime_n5_dataset1.png



