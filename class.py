import numpy as np
import matplotlib.pyplot as plt
import optuna

parents_params = np.array([[1, 1], [3, 3]])
search_space_bounds = np.array([[0, 4], [0, 4]])
study = optuna.create_study(directions=["minimize"])

eta = 1
# crossover = optuna.samplers.nsgaii.SBXCrossover(eta, use_child_gene_prob=1.0, uniform_crossover_prob=0.1)
crossover = optuna.samplers.nsgaii.VSBXCrossover(eta)


children_params = []
for i in range(1000):
    rng = np.random.RandomState(i)
    child_params = crossover.crossover(
        parents_params=parents_params,
        rng=rng,
        study=study,
        search_space_bounds=search_space_bounds,
    )

    children_params.append(child_params)
ch = np.array(children_params)

plt.figure(figsize=(8, 6))
plt.scatter(ch[:, 0], ch[:, 1], label="Children", s=10, c="#ff7f0e")
plt.scatter(
    parents_params[:, 0], parents_params[:, 1], label="Parents", marker="*", s=200, c="#1f77b4"
)

plt.title(f"SBX Crossover eta={eta}")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.xlim(-0.5, 4.5)
plt.ylim(-0.5, 4.5)

plt.show()
