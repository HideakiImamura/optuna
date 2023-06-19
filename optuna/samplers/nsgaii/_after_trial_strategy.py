import abc
from typing import Callable
from typing import Optional
from typing import Sequence

from optuna.samplers._base import _process_constraints_after_trial
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


class BaseAfterTrialStrategy(abc.ABC):
    @abc.abstractmethod
    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        """This method is called after each trial by NSGA-II sampler.

        Args:
            study:
                Target study object.
            trial:
                Target trial object.
            state:
                Target trial state.
            values:
                Target trial values.
        """
        raise NotImplementedError


class NSGAIIAfterTrialStrategy(BaseAfterTrialStrategy):
    def __init__(
        self, *, constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]] = None
    ) -> None:
        self._constraints_func = constraints_func

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        if self._constraints_func is not None:
            _process_constraints_after_trial(self._constraints_func, study, trial, state)
