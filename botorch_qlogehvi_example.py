import torch
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


torch.manual_seed(0)
problem = BraninCurrin(negate=True)

train_x = torch.rand(8, 2, dtype=torch.double)
train_y = problem(train_x)

model = ModelListGP(
    SingleTaskGP(train_x, train_y[:, [0]], outcome_transform=Standardize(m=1)),
    SingleTaskGP(train_x, train_y[:, [1]], outcome_transform=Standardize(m=1)),
)
fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))

partitioning = NondominatedPartitioning(ref_point=problem.ref_point, Y=train_y)
acqf = qLogExpectedHypervolumeImprovement(
    model=model,
    ref_point=problem.ref_point.tolist(),
    partitioning=partitioning,
)

candidates, _ = optimize_acqf(
    acq_function=acqf,
    bounds=problem.bounds,
    q=10,
    num_restarts=4,
    raw_samples=64,
)

print(candidates)
print(problem(candidates))
