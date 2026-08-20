from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, BinaryIO


class _CompatParzenEstimatorParameters(tuple):
    """Fallback for Optuna TPE internals whose namedtuple fields changed across versions."""

    __slots__ = ()

    def __new__(cls, *args: Any) -> "_CompatParzenEstimatorParameters":
        return super().__new__(cls, args)

    def __getnewargs__(self) -> tuple[Any, ...]:
        return tuple(self)


class _OptunaCompatUnpickler(pickle.Unpickler):
    _CLASS_OVERRIDES = {
        ("optuna.samplers._tpe.parzen_estimator", "_ParzenEstimatorParameters"): (
            _CompatParzenEstimatorParameters
        ),
    }

    def find_class(self, module: str, name: str) -> type[Any]:
        override = self._CLASS_OVERRIDES.get((module, name))
        if override is not None:
            return override
        return super().find_class(module, name)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return _load_with_compat(f)


def _load_with_compat(file_obj: BinaryIO) -> Any:
    try:
        return pickle.load(file_obj)
    except TypeError as exc:
        if "_ParzenEstimatorParameters" not in str(exc):
            raise
        file_obj.seek(0)
        return _OptunaCompatUnpickler(file_obj).load()
