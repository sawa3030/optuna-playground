from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import optuna


_FORMAT_VERSION = "study_snapshots_v1"


def _patch_optuna_pickle_compat() -> None:
    """Allow loading legacy Study pickles across Optuna internal API changes."""
    try:
        from optuna.samplers._tpe.parzen_estimator import _ParzenEstimatorParameters
    except Exception:
        return

    if len(getattr(_ParzenEstimatorParameters, "_fields", ())) != 7:
        return
    if _ParzenEstimatorParameters.__new__.__defaults__:
        return

    _ParzenEstimatorParameters.__new__.__defaults__ = ({},)


def _serialize_trial(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "state": trial.state.name,
        "user_attrs": dict(trial.user_attrs),
    }
    if trial.values is not None:
        payload["values"] = list(trial.values)
    elif trial.value is not None:
        payload["value"] = trial.value
    return payload


def _deserialize_trial(payload: dict[str, Any]) -> optuna.trial.FrozenTrial:
    kwargs: dict[str, Any] = {
        "state": optuna.trial.TrialState[payload["state"]],
        "user_attrs": payload.get("user_attrs", {}),
    }
    if "values" in payload:
        kwargs["values"] = payload["values"]
    elif "value" in payload:
        kwargs["value"] = payload["value"]
    return optuna.trial.create_trial(**kwargs)


def _serialize_study(study: optuna.Study) -> dict[str, Any]:
    return {
        "directions": [direction.name.lower() for direction in study.directions],
        "trials": [_serialize_trial(trial) for trial in study.trials],
    }


def _deserialize_study(payload: dict[str, Any]) -> optuna.Study:
    study = optuna.create_study(directions=payload["directions"])
    study.add_trials(_deserialize_trial(trial) for trial in payload["trials"])
    return study


def dump_study_list(path: Path, study_list: list[optuna.Study]) -> None:
    payload = {
        "format": _FORMAT_VERSION,
        "studies": [_serialize_study(study) for study in study_list],
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_study_list(path: Path) -> list[optuna.Study]:
    with path.open("rb") as f:
        _patch_optuna_pickle_compat()
        payload = pickle.load(f)

    if isinstance(payload, dict) and payload.get("format") == _FORMAT_VERSION:
        return [_deserialize_study(study) for study in payload["studies"]]

    return payload
