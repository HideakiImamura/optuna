from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
from scipy import linalg
from scipy.special import erfc

from optuna.samplers._gp.model import BaseModel
from optuna.samplers._gp.optimizer import BaseOptimizer


_EPS = 1e-12


class LP(BaseOptimizer):
    """Acquisition optimizer for batch suggesting with local penalization.

    For the detail of the algorithm, please see the
    `original paper <https://arxiv.org/abs/1505.08052>`_.
    """

    def __init__(self, n_batches: int, optimizer: BaseOptimizer):

        self._n_batches = n_batches
        self._optimizer = optimizer

    def optimize(
        self,
        f: Callable[[Any], Any],
        df: Optional[Callable[[Any], Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:

        assert df is not None
        assert kwargs is not None
        assert "model" in kwargs

        model = kwargs["model"]
        input_dim = model.input_dim
        # output_dim = model.input_dim
        self._L = self._estimate_L(model)
        self._M = self._estimate_M(model)

        xs = np.array([]).reshape((0, input_dim))

        def _f_origin(z: np.ndarray) -> np.ndarray:
            return np.log(self._g(f(z).flatten()))

        def _df_origin(z: np.ndarray) -> np.ndarray:
            fz = f(z).flatten()
            return np.einsum("i,ij,i->ij", self._dg(fz), df(z)[:, :, 0], 1 / self._g(fz))

        _f = _f_origin
        _df = _df_origin
        for i in range(self._n_batches):
            x = self._optimizer.optimize(f=_f, df=_df, kwargs=kwargs)
            xs = np.vstack([xs, x])

            def _f(z: np.ndarray) -> np.ndarray:
                ret = _f_origin(z)
                for y in xs:
                    _phi_val = self._phi(z, y, model)
                    if _phi_val <= 0:
                        _phi_val = _EPS * np.ones(_phi_val.shape)
                    ret += np.log(_phi_val)
                return ret

            def _df(z: np.ndarray) -> np.ndarray:
                ret = _df_origin(z)
                for y in xs:
                    _phi_val, _dphi_val = self._phi_dphi(z, y, model)
                    if _phi_val <= 0:
                        _phi_val = _EPS * np.ones(_phi_val.shape)
                    ret += _dphi_val.dot(1. / _phi_val)
                return ret

        return xs

    @staticmethod
    def _g(x: np.ndarray) -> np.ndarray:

        if (x > 0).any():
            return x
        else:
            return np.log(1 + np.exp(x))

    @staticmethod
    def _dg(x: np.ndarray) -> np.ndarray:

       if (x > 0).any():
           return np.ones(x.shape[0])
       else:
           return np.exp(x) / (1 + np.exp(x))

    def _phi(self, x: np.ndarray, x_j: np.ndarray, model: BaseModel) -> np.ndarray:

        mus, sigmas = model.predict(x_j)
        mus = mus[:, 0, :]
        sigmas = sigmas[:, 0, :, :]

        numerator = np.einsum("ak,k,k->ak", mus, linalg.norm(x - x_j) * self._L, - self._M)
        denominator = np.array([linalg.inv(sigmas[a]) for a in range(model.n_mcmc_samples)])
        z = np.einsum("ak,akl->al", numerator, denominator / np.sqrt(2))
        return np.sum(erfc(-z) / 2, axis=0) / model.n_mcmc_samples

    def _phi_dphi(
        self, x: np.ndarray, x_j: np.ndarray, model: BaseModel
    ) -> Tuple[np.ndarray, np.ndarray]:

        mus, sigmas = model.predict(x_j)
        mus = mus[:, 0, :]
        sigmas = sigmas[:, 0, :, :]

        norm = linalg.norm(x - x_j)
        inv_sigmas = np.array([linalg.inv(sigmas[a]) for a in range(model.n_mcmc_samples)])
        numerator = np.einsum("ak,k,k->ak", mus, norm * self._L, - self._M)
        denominator = inv_sigmas / np.sqrt(2)
        z = np.einsum("ak,akl->al", numerator, denominator)

        _phi_val = np.sum(erfc(-z) / 2, axis=0) / model.n_mcmc_samples

        _dphi_val = np.einsum("akl,ak->al", inv_sigmas / np.sqrt(2 * np.pi), np.exp(- z ** 2))
        _dphi_val = np.einsum("ak,k,j->ajk", _dphi_val, 2 * self._L / norm, x_j - x)
        _dphi_val = np.sum(_dphi_val, axis=0) / model.n_mcmc_samples

        return _phi_val, _dphi_val

    def _estimate_L(self, model: BaseModel) -> np.ndarray:

        L = np.zeros(model.output_dim)
        for k in range(model.output_dim):
            def _f(x: np.ndarray) -> np.ndarray:
                dmus, _ = model.predict_gradient(x)

                return - np.sum(linalg.norm(dmus[:, :, :, k], axis=2), axis=0).flatten() / model.n_mcmc_samples

            x_max = self._optimizer.optimize(f=_f)
            dmus, _ = model.predict_gradient(x_max)
            L[k] = np.sum(linalg.norm(dmus[:, :, :, k], axis=2)) / model.n_mcmc_samples
        return L

    @staticmethod
    def _estimate_M(model: BaseModel) -> np.ndarray:

        return np.max(model.y, axis=0)
