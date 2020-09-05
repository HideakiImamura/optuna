from typing import Any
from typing import Dict

import numpy as np

from optuna.samplers._gp.optimizer.base import BaseOptimizer
from optuna.samplers._gp.optimizer.lp import LP
from optuna.samplers._gp.optimizer.scipy import ScipyOptimizer


def optimizer_selector(
    optimizer: str, bounds: np.ndarray, kwargs: Dict[str, Any]
) -> BaseOptimizer:
    """Selector module for acquisition optimizers."""

    if optimizer == "L-BFGS-B":
        return ScipyOptimizer(bounds=bounds, method=optimizer, **kwargs)
    elif optimizer == "LP":
        n_batches = kwargs["n_batches"]
        del kwargs["n_batches"]
        base_optimizer = kwargs.get("base_optimizer", "L-BFGS-B")
        del kwargs["base_optimizer"]
        base_optimizer = ScipyOptimizer(bounds=bounds, method=base_optimizer, **kwargs)
        return LP(n_batches=n_batches, optimizer=base_optimizer)
    else:
        raise ValueError("The optimizer {} is not supported.".format(optimizer))
