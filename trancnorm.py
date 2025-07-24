import math
import time

import functools
import math
import sys
import matplotlib.pyplot as plt


import numpy as np

from optuna.samplers._tpe._erf import erf

from optuna.samplers._tpe._truncnorm import rvs as optuna_rvs
from scipy.stats import truncnorm as scipy_truncnorm

def master_rvs(
    a: np.ndarray,
    b: np.ndarray,
    loc: np.ndarray | float = 0,
    scale: np.ndarray | float = 1,
    random_state: np.random.RandomState | None = None,
) -> np.ndarray:

    _norm_pdf_C = math.sqrt(2 * math.pi)
    _norm_pdf_logC = math.log(_norm_pdf_C)
    _ndtri_exp_approx_C = math.sqrt(3) / math.pi


    def _log_sum(log_p: np.ndarray, log_q: np.ndarray) -> np.ndarray:
        return np.logaddexp(log_p, log_q)


    def _log_diff(log_p: np.ndarray, log_q: np.ndarray) -> np.ndarray:
        return log_p + np.log1p(-np.exp(log_q - log_p))


    @functools.lru_cache(1000)
    def _ndtr_single(a: float) -> float:
        x = a / 2**0.5

        if x < -1 / 2**0.5:
            y = 0.5 * math.erfc(-x)
        elif x < 1 / 2**0.5:
            y = 0.5 + 0.5 * math.erf(x)
        else:
            y = 1.0 - 0.5 * math.erfc(x)

        return y


    def _ndtr(a: np.ndarray) -> np.ndarray:
        # todo(amylase): implement erfc in _erf.py and use it for big |a| inputs.
        return 0.5 + 0.5 * erf(a / 2**0.5)


    @functools.lru_cache(1000)
    def _log_ndtr_single(a: float) -> float:
        if a > 6:
            return -_ndtr_single(-a)
        if a > -20:
            return math.log(_ndtr_single(a))

        log_LHS = -0.5 * a**2 - math.log(-a) - 0.5 * math.log(2 * math.pi)
        last_total = 0.0
        right_hand_side = 1.0
        numerator = 1.0
        denom_factor = 1.0
        denom_cons = 1 / a**2
        sign = 1
        i = 0

        while abs(last_total - right_hand_side) > sys.float_info.epsilon:
            i += 1
            last_total = right_hand_side
            sign = -sign
            denom_factor *= denom_cons
            numerator *= 2 * i - 1
            right_hand_side += sign * numerator * denom_factor

        return log_LHS + math.log(right_hand_side)


    def _log_ndtr(a: np.ndarray) -> np.ndarray:
        return np.frompyfunc(_log_ndtr_single, 1, 1)(a).astype(float)


    def _norm_logpdf(x: np.ndarray) -> np.ndarray:
        return -(x**2) / 2.0 - _norm_pdf_logC


    def _log_gauss_mass(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Log of Gaussian probability mass within an interval"""

        # Calculations in right tail are inaccurate, so we'll exploit the
        # symmetry and work only in the left tail
        case_left = b <= 0
        case_right = a > 0
        case_central = ~(case_left | case_right)

        def mass_case_left(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return _log_diff(_log_ndtr(b), _log_ndtr(a))

        def mass_case_right(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return mass_case_left(-b, -a)

        def mass_case_central(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            # Previously, this was implemented as:
            # left_mass = mass_case_left(a, 0)
            # right_mass = mass_case_right(0, b)
            # return _log_sum(left_mass, right_mass)
            # Catastrophic cancellation occurs as np.exp(log_mass) approaches 1.
            # Correct for this with an alternative formulation.
            # We're not concerned with underflow here: if only one term
            # underflows, it was insignificant; if both terms underflow,
            # the result can't accurately be represented in logspace anyway
            # because sc.log1p(x) ~ x for small x.
            return np.log1p(-_ndtr(a) - _ndtr(-b))

        # _lazyselect not working; don't care to debug it
        out = np.full_like(a, fill_value=np.nan, dtype=np.complex128)
        if a[case_left].size:
            out[case_left] = mass_case_left(a[case_left], b[case_left])
        if a[case_right].size:
            out[case_right] = mass_case_right(a[case_right], b[case_right])
        if a[case_central].size:
            out[case_central] = mass_case_central(a[case_central], b[case_central])
        return np.real(out)  # discard ~0j


    def _ndtri_exp_single(y: float) -> float:
        """
        Use the Newton method to efficiently find the root.

        `ndtri_exp(y)` returns `x` such that `y = log_ndtr(x)`, meaning that the Newton method is
        supposed to find the root of `f(x) := log_ndtr(x) - y = 0`.

        Since `df/dx = d log_ndtr(x)/dx = (ndtr(x))'/ndtr(x) = norm_pdf(x)/ndtr(x)`, the Newton update
        is x[n + 1] := x[n] - (log_ndtr(x) - y) * ndtr(x) / norm_pdf(x).

        As an initial guess, we use the Gaussian tail asymptotic approximation:
            1 - ndtr(x) \\simeq norm_pdf(x) / x
            --> log(norm_pdf(x) / x) = -1/2 * x**2 - 1/2 * log(2pi) - log(x)

        First recall that y is a non-positive value and y = log_ndtr(inf) = 0 and
        y = log_ndtr(-inf) = -inf.

        If abs(y) is very small, x is very large, meaning that x**2 >> log(x) and
        ndtr(x) = exp(y) \\simeq 1 + y --> -y \\simeq 1 - ndtr(x). From this, we can calculate:
            log(1 - ndtr(x)) \\simeq log(-y) \\simeq -1/2 * x**2 - 1/2 * log(2pi) - log(x).
        Because x**2 >> log(x), we can ignore the second and third terms, leading to:
            log(-y) \\simeq -1/2 * x**2 --> x \\simeq sqrt(-2 log(-y)).
        where we take the positive `x` as abs(y) becomes very small only if x >> 0.
        The second order approximation version is sqrt(-2 log(-y) - log(-2 log(-y))).

        If abs(y) is very large, we use log_ndtr(x) \\simeq -1/2 * x**2 - 1/2 * log(2pi) - log(x).
        To solve this equation analytically, we ignore the log term, yielding:
            log_ndtr(x) = y \\simeq -1/2 * x**2 - 1/2 * log(2pi)
            --> y + 1/2 * log(2pi) = -1/2 * x**2 --> x**2 = -2 * (y + 1/2 * log(2pi))
            --> x = sqrt(-2 * (y + 1/2 * log(2pi))

        For the moderate y, we use Eq. (13), i.e., standard logistic CDF, in the following paper:

        - `Approximating the Cumulative Distribution Function of the Normal Distribution
        <https://jsr.isrt.ac.bd/wp-content/uploads/41n1_5.pdf>`__

        The standard logistic CDF approximates ndtr(x) with:
            exp(y) = ndtr(x) \\simeq 1 / (1 + exp(-pi * x / sqrt(3)))
            --> exp(-y) \\simeq 1 + exp(-pi * x / sqrt(3))
            --> log(exp(-y) - 1) \\simeq -pi * x / sqrt(3)
            --> x \\simeq -sqrt(3) / pi * log(exp(-y) - 1).
        """
        if y > -sys.float_info.min:
            return math.inf if y <= 0 else math.nan

        if y > -1e-2:  # Case 1. abs(y) << 1.
            u = -2.0 * math.log(-y)
            x = math.sqrt(u - math.log(u))
        elif y < -5:  # Case 2. abs(y) >> 1.
            x = -math.sqrt(-2.0 * (y + _norm_pdf_logC))
        else:  # Case 3. Moderate y.
            x = -_ndtri_exp_approx_C * math.log(math.exp(-y) - 1)

        for _ in range(100):
            log_ndtr_x = _log_ndtr_single(x)
            log_norm_pdf_x = -0.5 * x**2 - _norm_pdf_logC
            # NOTE(nabenabe): Use exp(log_ndtr_x - log_norm_pdf_x) instead of ndtr_x / norm_pdf_x for
            # numerical stability.
            dx = (log_ndtr_x - y) * math.exp(log_ndtr_x - log_norm_pdf_x)
            x -= dx
            if abs(dx) < 1e-8 * abs(x):  # Equivalent to np.isclose with atol=0.0 and rtol=1e-8.
                break

        return x


    def _ndtri_exp(y: np.ndarray) -> np.ndarray:
        return np.frompyfunc(_ndtri_exp_single, 1, 1)(y).astype(float)


    def ppf(q: np.ndarray, a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
        q, a, b = np.atleast_1d(q, a, b)
        q, a, b = np.broadcast_arrays(q, a, b)

        case_left = a < 0
        case_right = ~case_left

        def ppf_left(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            log_Phi_x = _log_sum(_log_ndtr(a), np.log(q) + _log_gauss_mass(a, b))
            return _ndtri_exp(log_Phi_x)

        def ppf_right(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            log_Phi_x = _log_sum(_log_ndtr(-b), np.log1p(-q) + _log_gauss_mass(a, b))
            return -_ndtri_exp(log_Phi_x)

        out = np.empty_like(q)

        q_left = q[case_left]
        q_right = q[case_right]

        if q_left.size:
            out[case_left] = ppf_left(q_left, a[case_left], b[case_left])
        if q_right.size:
            out[case_right] = ppf_right(q_right, a[case_right], b[case_right])

        out[q == 0] = a[q == 0]
        out[q == 1] = b[q == 1]
        out[a == b] = math.nan

        return out
            
            
    """
    This function generates random variates from a truncated normal distribution defined between
    `a` and `b` with the mean of `loc` and the standard deviation of `scale`.
    """
    random_state = random_state or np.random.RandomState()
    size = np.broadcast(a, b, loc, scale).shape
    quantiles = random_state.uniform(low=0, high=1, size=size)
    return ppf(quantiles, a, b) * scale + loc


rng = np.random.RandomState(42)
n_repeats = 100
size_list = list(range(100, 3100, 100))
runtimes_master = np.empty((len(size_list), n_repeats), dtype=float)
runtimes_this_pr = np.empty((len(size_list), n_repeats), dtype=float)
runtimes_scipy = np.empty((len(size_list), n_repeats), dtype=float)
runtimes_math = np.empty((len(size_list), n_repeats), dtype=float)
runtimes_numpy = np.empty((len(size_list), n_repeats), dtype=float)
for i, size in enumerate(size_list):
    print(f"{size=}")
    start = time.time()
    for j in range(n_repeats):
        rnd_low = rng.uniform(-2, 0, size=(size))
        rnd_high = rng.uniform(0, 2, size=(size))
        rnd_loc = rng.normal(0, 1, size=(size))
        rnd_scale = rng.uniform(0.1, 2, size=(size))

        start = time.time()
        res = optuna_rvs((rnd_low - rnd_loc) / rnd_scale, (rnd_high - rnd_loc) / rnd_scale, rnd_loc, rnd_scale, rng)
        runtimes_this_pr[i, j] = time.time() - start
        start = time.time()
        res = master_rvs((rnd_low - rnd_loc) / rnd_scale, (rnd_high - rnd_loc) / rnd_scale, rnd_loc, rnd_scale, rng)
        runtimes_master[i, j] = time.time() - start
        start = time.time()
        res = scipy_truncnorm.rvs(
            a=(rnd_low - rnd_loc) / rnd_scale, 
            b=(rnd_high - rnd_loc) / rnd_scale, 
            loc=rnd_loc, 
            scale=rnd_scale, 
            random_state=rng
        )
        runtimes_scipy[i, j] = time.time() - start

runtimes_this_pr *= 1000
runtimes_master *= 1000
runtimes_math *= 1000
runtimes_numpy *= 1000
runtimes_scipy *= 1000

_, ax = plt.subplots()
markevery = 10

means = np.mean(runtimes_this_pr, axis=-1)
stderrs = np.std(runtimes_this_pr, axis=-1) / np.sqrt(n_repeats)
ax.plot(size_list, means, color="darkred", ls="dashed", label="This PR", marker="*", markevery=markevery, ms=10)
ax.fill_between(size_list, means - stderrs, means + stderrs, color="darkred", alpha=0.2)

means = np.mean(runtimes_master, axis=-1)
stderrs = np.std(runtimes_master, axis=-1) / np.sqrt(n_repeats)
ax.plot(size_list, means, color="blue", ls="dashed", label="Master")
ax.fill_between(size_list, means - stderrs, means + stderrs, color="blue", alpha=0.2)

means = np.mean(runtimes_scipy, axis=-1)
stderrs = np.std(runtimes_scipy, axis=-1) / np.sqrt(n_repeats)
ax.plot(size_list, means, color="yellow", label="SciPy", ls="dotted", gapcolor="black")
ax.fill_between(size_list, means - stderrs, means + stderrs, color="yellow", alpha=0.2)

ax.legend()
ax.set_xlabel("Number of Points")
ax.set_ylabel("Elapsed Time [ms]")
ax.set_yscale("log")
ax.grid(which="minor", color="gray", linestyle=":")
ax.grid(which="major", color="black")
plt.savefig("runtime.png", bbox_inches="tight")
