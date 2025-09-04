import numpy as np

# from optuna._hypervolume.wfg import _compute_hv
from optuna.study._multi_objective import _is_pareto_front_2d

def master_is_pareto_front_nd(unique_lexsorted_loss_values: np.ndarray) -> np.ndarray:
    loss_values = unique_lexsorted_loss_values[:, 1:]
    n_trials = loss_values.shape[0]
    on_front = np.zeros(n_trials, dtype=bool)
    nondominated_indices: np.ndarray[tuple[int, ...], np.dtype[np.signedinteger]] = np.arange(
        n_trials
    )
    while len(loss_values):
        nondominated_and_not_top = np.any(loss_values < loss_values[0], axis=1)
        on_front[nondominated_indices[0]] = True
        loss_values = loss_values[nondominated_and_not_top]
        nondominated_indices = nondominated_indices[nondominated_and_not_top]

    return on_front

def master_is_pareto_front_for_unique_sorted(unique_lexsorted_loss_values: np.ndarray) -> np.ndarray:
    (n_trials, n_objectives) = unique_lexsorted_loss_values.shape
    if n_objectives == 1:
        on_front = np.zeros(len(unique_lexsorted_loss_values), dtype=bool)
        on_front[0] = True  # Only the first element is Pareto optimal.
        return on_front
    elif n_objectives == 2:
        return _is_pareto_front_2d(unique_lexsorted_loss_values)
    else:
        return master_is_pareto_front_nd(unique_lexsorted_loss_values)

def master_is_pareto_front(loss_values: np.ndarray, assume_unique_lexsorted: bool) -> np.ndarray:
    if assume_unique_lexsorted:
        return master_is_pareto_front_for_unique_sorted(loss_values)

    unique_lexsorted_loss_values, order_inv = np.unique(loss_values, axis=0, return_inverse=True)
    on_front = master_is_pareto_front_for_unique_sorted(unique_lexsorted_loss_values)
    return on_front[order_inv.reshape(-1)]

def master_compute_exclusive_hv(
    limited_sols: np.ndarray, inclusive_hv: float, reference_point: np.ndarray
) -> float:
    # print("limited_sols", limited_sols)
    if limited_sols.shape[0] == 0:
        # print("No limited solutions, returning inclusive_hv:", inclusive_hv)
        return inclusive_hv

    on_front = master_is_pareto_front(limited_sols, assume_unique_lexsorted=True)
    # print("on_front", on_front)
    return inclusive_hv - master_compute_hv(limited_sols[on_front], reference_point)


def master_compute_hv(sorted_loss_vals: np.ndarray, reference_point: np.ndarray) -> float:
    inclusive_hvs = np.prod(reference_point - sorted_loss_vals, axis=-1)
    if inclusive_hvs.shape[0] == 1:
        return float(inclusive_hvs[0])
    elif inclusive_hvs.shape[0] == 2:
        # S(A v B) = S(A) + S(B) - S(A ^ B).
        intersec = np.prod(reference_point - np.maximum(sorted_loss_vals[0], sorted_loss_vals[1]))
        return np.sum(inclusive_hvs) - intersec

    # c.f. Eqs. (6) and (7) of ``A Fast Way of Calculating Exact Hypervolumes``.
    limited_sols_array = np.maximum(sorted_loss_vals[:, np.newaxis], sorted_loss_vals)
    # print("limited_sols_array", limited_sols_array)
    # print("sorted_loss_vals", sorted_loss_vals)
    # print("sorted_loss_vals[:, np.newaxis]", sorted_loss_vals[:, np.newaxis])

    # return sum(
    #     master_compute_exclusive_hv(limited_sols_array[i, i + 1 :], inclusive_hv, reference_point)
    #     for i, inclusive_hv in enumerate(inclusive_hvs)
    # )
    # print("limited_sols_array", limited_sols_array)
    # print("inclusive_hvs", inclusive_hvs)
    # print()
    sum = 0.0
    for i, inclusive_hv in enumerate(inclusive_hvs):
        exclusive_hv = master_compute_exclusive_hv(limited_sols_array[i, i + 1 :], inclusive_hv, reference_point)
        # print("i:", i, ", exclusive_hv:", exclusive_hv)
        # print()
        sum += exclusive_hv
    return sum

def pr_is_pareto_front_nd(unique_lexsorted_loss_values: np.ndarray) -> np.ndarray:
    loss_values = unique_lexsorted_loss_values[:, 1:]
    n_trials = loss_values.shape[0]
    on_front = np.zeros(n_trials, dtype=bool)
    remaining_indices: np.ndarray[tuple[int, ...], np.dtype[np.signedinteger]] = np.arange(
        n_trials
    )
    while len(remaining_indices):
        on_front[(new_nondominated_index := remaining_indices[0])] = True
        nondominated_and_not_top = np.any(
            loss_values[remaining_indices] < loss_values[new_nondominated_index], axis=1
        )
        remaining_indices = remaining_indices[nondominated_and_not_top]

    return on_front

def pr_is_pareto_front_for_unique_sorted(unique_lexsorted_loss_values: np.ndarray) -> np.ndarray:
    (n_trials, n_objectives) = unique_lexsorted_loss_values.shape
    if n_objectives == 1:
        on_front = np.zeros(len(unique_lexsorted_loss_values), dtype=bool)
        on_front[0] = True  # Only the first element is Pareto optimal.
        return on_front
    elif n_objectives == 2:
        return _is_pareto_front_2d(unique_lexsorted_loss_values)
    else:
        return pr_is_pareto_front_nd(unique_lexsorted_loss_values)

def pr_is_pareto_front(loss_values: np.ndarray, assume_unique_lexsorted: bool) -> np.ndarray:
    if assume_unique_lexsorted:
        return pr_is_pareto_front_for_unique_sorted(loss_values)

    unique_lexsorted_loss_values, order_inv = np.unique(loss_values, axis=0, return_inverse=True)
    on_front = pr_is_pareto_front_for_unique_sorted(unique_lexsorted_loss_values)
    return on_front[order_inv.reshape(-1)]

def pr_compute_exclusive_hv(
    limited_sols: np.ndarray, inclusive_hv: float, reference_point: np.ndarray
) -> float:
    assert limited_sols.shape[0] >= 1
    # print()
    # print("exclusive_hv limited_sols", limited_sols)
    if limited_sols.shape[0] <= 3:
        # NOTE(nabenabe): Don't use _is_pareto_front for 3 or fewer points to avoid its overhead.
        # print("under three!!!")
        return inclusive_hv - pr_compute_hv(limited_sols, reference_point)

    on_front = pr_is_pareto_front(limited_sols, assume_unique_lexsorted=True)
    # print("on_front", on_front)
    return inclusive_hv - pr_compute_hv(limited_sols[on_front], reference_point)

def pr_compute_hv(sorted_loss_vals: np.ndarray, reference_point: np.ndarray) -> float:
    if sorted_loss_vals.shape[0] == 1:
        # print("pr 1d")
        # NOTE(nabenabe): NumPy overhead is slower than the simple for-loop here.
        inclusive_hv = 1.0
        for r, v in zip(reference_point, sorted_loss_vals[0]):
            inclusive_hv *= r - v
        return float(inclusive_hv)
    elif sorted_loss_vals.shape[0] == 2:
        # print("pr 2d")
        # NOTE(nabenabe): NumPy overhead is slower than the simple for-loop here.
        # S(A v B) = S(A) + S(B) - S(A ^ B).
        hv1, hv2, intersec = 1.0, 1.0, 1.0
        for r, v1, v2 in zip(reference_point, sorted_loss_vals[0], sorted_loss_vals[1]):
            hv1 *= r - v1
            hv2 *= r - v2
            intersec *= r - max(v1, v2)
        return hv1 + hv2 - intersec

    inclusive_hvs = np.prod(reference_point - sorted_loss_vals, axis=-1)
    # c.f. Eqs. (6) and (7) of ``A Fast Way of Calculating Exact Hypervolumes``.
    limited_sols_array = np.maximum(sorted_loss_vals[:, np.newaxis], sorted_loss_vals)
    # print("limited_sols_array", limited_sols_array)

    sum = inclusive_hvs[-1]
    # print("inclusive_hvs", inclusive_hvs)
    for i in range(inclusive_hvs.size - 1):
        exclusive_hv = pr_compute_exclusive_hv(limited_sols_array[i, i + 1 :], inclusive_hvs[i], reference_point)
        # print("i:", i, ", exclusive_hv:", exclusive_hv)
        sum += exclusive_hv
    return sum


    # return inclusive_hvs[-1] + sum(
    #     pr_compute_exclusive_hv(limited_sols_array[i, i + 1 :], inclusive_hvs[i], reference_point)
    #     for i in range(inclusive_hvs.size - 1)
    # )

# sorted_loss_vals = np.array([[-1.1, 2], [3., 4], [5, 6]])
# reference_point = np.array([10, 10])

from optuna._hypervolume.wfg import _compute_hv
import numpy as np
np.random.seed(42)
n = [1, 2, 3, 4, 10, 100]
n_objectives = [2, 3, 4]
for d in n_objectives:
    for n_ in n:
        loss_vals = np.random.rand(n_, d)
        sorted_loss_vals = loss_vals[np.argsort(loss_vals[:, 0])]
        reference_point = np.ones(d)
        print(f"n={n_}, d={d}, result:", _compute_hv(sorted_loss_vals, reference_point))
# loss_vals = np.random.rand(100, 2)
# sorted_loss_vals = loss_vals[np.argsort(loss_vals[:, 0])]
# print("sorted_loss_vals", sorted_loss_vals)
# reference_point = np.array([1, 1])
# print("result:", _compute_hv(sorted_loss_vals, reference_point))
# print("Master HV:", master_compute_hv(sorted_loss_vals, reference_point))
# print("PR HV:", pr_compute_hv(sorted_loss_vals, reference_point))