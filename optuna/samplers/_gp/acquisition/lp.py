import numpy as np
from scipy import linalg
from scipy import stats
from scipy.special import erfc

from optuna.samplers._gp.acquisition.base import BaseAcquisitionFunction
from optuna.samplers._gp.acquisition.ei import EI
from optuna.samplers._gp.model import BaseModel


class LP(BaseAcquisitionFunction):
    """Acquisition function with local penalization.

    For the detail of the algorithm, please see the
    `original paper <https://arxiv.org/abs/1505.08052>`_.
    """

    def __init__(self, n_batches: int, base_acquisition: str = 'EI', sigma0: float = 1e-10):

        self._n_batches = n_batches

        if base_acquisition == 'EI':
            self._base_acquisition = EI(sigma0)

    def compute_acq(self, x: np.ndarray, x_j: np.ndarray, model: BaseModel) -> np.ndarray:

        y = self._base_acquisition.compute_acq(x, model)
        gy =



    def compute_grad(self, x: np.ndarray, model: BaseModel) -> np.ndarray:

        x = np.atleast_2d(x)
        self._verify_input(x, model)
        n = x.shape[0]
        mus, sigmas = model.predict(x)
        dmus, dsigmas = model.predict_gradient(x)

        def _compute(a: int) -> np.ndarray:
            mu, sigma, dmu, dsigma = mus[a], sigmas[a], dmus[a], dsigmas[a]
            y_best = np.min(model.y, axis=0)
            inv_sigma = np.asarray(
                [linalg.inv(sigma[i] + self._sigma0 * np.eye(model.output_dim)) for i in range(n)]
            )
            gamma = np.einsum("ijk,ik->ij", inv_sigma, y_best - mu)
            dgamma = -np.einsum("Ipr,Iirs,Is->Iip", inv_sigma, dsigma, gamma) + np.einsum(
                "Ipq,Iiq->Iip", inv_sigma, y_best - dmu
            )

            _Phi = stats.norm.cdf(gamma)
            _phi = stats.norm.pdf(gamma)
            z = gamma * _Phi + _phi
            dz = np.einsum("Iip,Ip->Iip", dgamma, _Phi)

            dy = np.einsum("Iijp,Ip->Iij", dsigma, z) + np.einsum("Ijp,Iip->Iij", sigma, dz)
            return dy

        dy = (
            np.sum([_compute(a) for a in range(model.n_mcmc_samples)], axis=0)
            / model.n_mcmc_samples
        )

        self._verify_output_grad(dy, model)

        return dy

    @staticmethod
    def _g(x: np.ndarray) -> np.ndarray:

        if (x > 0).any():
            return x
        else:
            return np.log(1 + np.exp(x))

    def _phi(self, x: np.ndarray, x_j: np.ndarray, model: BaseModel) -> np.ndarray:

        mus, sigmas = model.predict(x_j)
        z = (mus + self._L * linalg.norm(x - x_j) - self._M) / np.sqrt(2 * sigmas)
        return erfc(-z) / 2
