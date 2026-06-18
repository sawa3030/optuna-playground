source ~/pfn/optuna/venv/bin/
cd ~/pfn/optuna-playground

# python gp_simulate_for_non_evaluation_time.py --label max --out-dir ./maxoperator_without_evaltime_n5_dataset0 --dataset-id 0
# python gp_simulate_for_non_evaluation_time.py --label max --out-dir ./maxoperator_without_evaltime_n5_dataset1 --dataset-id 1
# python gp_simulate_for_non_evaluation_time.py --label max --out-dir ./maxoperator_without_evaltime_n5_dataset2 --dataset-id 2

# python gp_simulate_for_non_evaluation_time.py --label fatmax --out-dir ./maxoperator_without_evaltime_n5_bbo_dataset1 --dataset-id 1
# python gp_simulate_for_non_evaluation_time.py --label fatmax --out-dir ./maxoperator_without_evaltime_n5_bbo_dataset6 --dataset-id 6
# python gp_simulate_for_non_evaluation_time.py --label fatmax --out-dir ./maxoperator_without_evaltime_n5_bbo_dataset10 --dataset-id 10

# python gp_simulator.py --label qlogei-128 --out-dir ./gp_simulator_results_without_evaltime_n5_dataset0 --dataset-id 0
# python gp_simulator.py --label qlogei-128 --out-dir ./gp_simulator_results_without_evaltime_n5_dataset1 --dataset-id 1
# python gp_simulator.py --label qlogei --out-dir ./gp_simulator_results_without_evaltime_n5_dataset2 --dataset-id 2

# python gp_plot.py --labels qlogei-64 qlogei-128 qlogei master --result-dir ./gp_simulator_results_without_evaltime_n5_bbo_dataset15 --output async-bench-example_without_evaltime_n5_bbo_dataset15.png
# python gp_plot.py --labels qlogei-64 qlogei-128 qlogei master --result-dir ./gp_simulator_results_without_evaltime_n5_bbo_dataset10 --output async-bench-example_without_evaltime_n5_bbo_dataset10.png
# python gp_plot.py --labels qlogei-64 qlogei-128 qlogei master --result-dir ./gp_simulator_results_without_evaltime_n5_bbo_dataset6 --output async-bench-example_without_evaltime_n5_bbo_dataset6.png
# python gp_plot.py --labels qlogei-64 qlogei-128 qlogei master --result-dir ./gp_simulator_results_without_evaltime_n5_bbo_dataset1 --output async-bench-example_without_evaltime_n5_bbo_dataset1.png

python gp_plot.py --labels fatmax max --result-dir ./maxoperator_without_evaltime_n5_dataset0 --output maxoperator_without_evaltime_n5_dataset0.png
python gp_plot.py --labels fatmax max --result-dir ./maxoperator_without_evaltime_n5_dataset1 --output maxoperator_without_evaltime_n5_dataset1.png
python gp_plot.py --labels fatmax max --result-dir ./maxoperator_without_evaltime_n5_dataset2 --output maxoperator_without_evaltime_n5_dataset2.png

python gp_plot.py --labels fatmax max --result-dir ./maxoperator_without_evaltime_n5_bbo_dataset1 --output maxoperator_without_evaltime_n5_bbo_dataset1.png
python gp_plot.py --labels fatmax max --result-dir ./maxoperator_without_evaltime_n5_bbo_dataset6 --output maxoperator_without_evaltime_n5_bbo_dataset6.png
python gp_plot.py --labels fatmax max --result-dir ./maxoperator_without_evaltime_n5_bbo_dataset10 --output maxoperator_without_evaltime_n5_bbo_dataset10.png

# python gp_plot.py --labels qlogei-64 qlogei-128 qlogei master --result-dir ./gp_simulator_results_without_evaltime_n5_dataset0 --output async-bench-example_without_evaltime_n5_dataset0.png
# python gp_plot.py --labels qlogei-64 qlogei-128 qlogei master --result-dir ./gp_simulator_results_without_evaltime_n5_dataset1 --output async-bench-example_without_evaltime_n5_dataset1.png



