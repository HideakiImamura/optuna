from __future__ import annotations

import abc
from collections import defaultdict
import itertools
from typing import Callable
from typing import cast
from typing import DefaultDict
from typing import List
from typing import Optional
from typing import Sequence
import warnings

import numpy as np

import optuna
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.study._multi_objective import _dominates
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


class BaseElitePopulationSelectionStrategy(abc.ABC):
    @abc.abstractmethod
    def select_elite_population(
        self, study: Study, population: list[FrozenTrial]
    ) -> list[FrozenTrial]:
        """Select elite population from the given trials.

        Args:
            study:
                Target study object.
            population:
                Trials in the study.

        Returns:
            A list of trials that are selected as elite population.
        """
        raise NotImplementedError


class NSGAIIElitePopulationSelectionStrategy(BaseElitePopulationSelectionStrategy):
    def __init__(
        self,
        *,
        population_size: int,
        constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
    ) -> None:
        if not isinstance(population_size, int):
            raise TypeError("`population_size` must be an integer value.")

        if population_size < 2:
            raise ValueError("`population_size` must be greater than or equal to 2.")

        self._population_size = population_size
        self._constraints_func = constraints_func

    def select_elite_population(
        self, study: Study, population: list[FrozenTrial]
    ) -> list[FrozenTrial]:
        """Select elite population from the given trials by NSGA-II algorithm.

        Args:
            study:
                Target study object.
            population:
                Trials in the study.

        Returns:
            A list of trials that are selected as elite population.
        """
        _validate_constraints(population, self._constraints_func)

        elite_population: List[FrozenTrial] = []
        dominates = _dominates if self._constraints_func is None else _constrained_dominates
        population_per_rank = _fast_non_dominated_sort(population, study.directions, dominates)
        for population in population_per_rank:
            if len(elite_population) + len(population) < self._population_size:
                elite_population.extend(population)
            else:
                n = self._population_size - len(elite_population)
                _crowding_distance_sort(population)
                elite_population.extend(population[:n])
                break

        return elite_population


def _calc_crowding_distance(population: List[FrozenTrial]) -> DefaultDict[int, float]:
    """Calculates the crowding distance of population.

    We define the crowding distance as the summation of the crowding distance of each dimension
    of value calculated as follows:

    * If all values in that dimension are the same, i.e., [1, 1, 1] or [inf, inf],
      the crowding distances of all trials in that dimension are zero.
    * Otherwise, the crowding distances of that dimension is the difference between
      two nearest values besides that value, one above and one below, divided by the difference
      between the maximal and minimal finite value of that dimension. Please note that:
        * the nearest value below the minimum is considered to be -inf and the
          nearest value above the maximum is considered to be inf, and
        * inf - inf and (-inf) - (-inf) is considered to be zero.
    """

    manhattan_distances: DefaultDict[int, float] = defaultdict(float)
    if len(population) == 0:
        return manhattan_distances

    for i in range(len(population[0].values)):
        population.sort(key=lambda x: cast(float, x.values[i]))

        # If population have the same values[i], ignore that value.
        if population[0].values[i] == population[-1].values[i]:
            continue

        vs = (
            [-float("inf")]
            + [cast(List[float], population[j].values)[i] for j in range(len(population))]
            + [float("inf")]
        )

        # Smallest finite value.
        v_min = next(x for x in vs if x != -float("inf"))

        # Largest finite value.
        v_max = next(x for x in reversed(vs) if x != float("inf"))

        width = v_max - v_min
        if width <= 0:
            # width == 0 or width == -inf
            width = 1.0

        for j in range(len(population)):
            # inf - inf and (-inf) - (-inf) is considered to be zero.
            gap = 0.0 if vs[j] == vs[j + 2] else vs[j + 2] - vs[j]
            manhattan_distances[population[j].number] += gap / width
    return manhattan_distances


def _crowding_distance_sort(population: List[FrozenTrial]) -> None:
    manhattan_distances = _calc_crowding_distance(population)
    population.sort(key=lambda x: manhattan_distances[x.number])
    population.reverse()


def _constrained_dominates(
    trial0: FrozenTrial, trial1: FrozenTrial, directions: Sequence[StudyDirection]
) -> bool:
    """Checks constrained-domination.

    A trial x is said to constrained-dominate a trial y, if any of the following conditions is
    true:
    1) Trial x is feasible and trial y is not.
    2) Trial x and y are both infeasible, but solution x has a smaller overall constraint
    violation.
    3) Trial x and y are feasible and trial x dominates trial y.
    """

    constraints0 = trial0.system_attrs.get(_CONSTRAINTS_KEY)
    constraints1 = trial1.system_attrs.get(_CONSTRAINTS_KEY)

    if constraints0 is None:
        warnings.warn(
            f"Trial {trial0.number} does not have constraint values."
            " It will be dominated by the other trials."
        )

    if constraints1 is None:
        warnings.warn(
            f"Trial {trial1.number} does not have constraint values."
            " It will be dominated by the other trials."
        )

    if constraints0 is None and constraints1 is None:
        # Neither Trial x nor y has constraints values
        return _dominates(trial0, trial1, directions)

    if constraints0 is not None and constraints1 is None:
        # Trial x has constraint values, but y doesn't.
        return True

    if constraints0 is None and constraints1 is not None:
        # If Trial y has constraint values, but x doesn't.
        return False

    assert isinstance(constraints0, (list, tuple))
    assert isinstance(constraints1, (list, tuple))

    if len(constraints0) != len(constraints1):
        raise ValueError("Trials with different numbers of constraints cannot be compared.")

    if trial0.state != TrialState.COMPLETE:
        return False

    if trial1.state != TrialState.COMPLETE:
        return True

    satisfy_constraints0 = all(v <= 0 for v in constraints0)
    satisfy_constraints1 = all(v <= 0 for v in constraints1)

    if satisfy_constraints0 and satisfy_constraints1:
        # Both trials satisfy the constraints.
        return _dominates(trial0, trial1, directions)

    if satisfy_constraints0:
        # trial0 satisfies the constraints, but trial1 violates them.
        return True

    if satisfy_constraints1:
        # trial1 satisfies the constraints, but trial0 violates them.
        return False

    # Both trials violate the constraints.
    violation0 = sum(v for v in constraints0 if v > 0)
    violation1 = sum(v for v in constraints1 if v > 0)
    return violation0 < violation1


def _validate_constraints(
    population: List[FrozenTrial],
    constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
) -> None:
    if constraints_func is not None:
        for _trial in population:
            _constraints = _trial.system_attrs.get(_CONSTRAINTS_KEY)
            if _constraints is None:
                continue
            if np.any(np.isnan(np.array(_constraints))):
                raise ValueError("NaN is not acceptable as constraint value.")


def _fast_non_dominated_sort(
    population: List[FrozenTrial],
    directions: List[optuna.study.StudyDirection],
    dominates: Callable[[FrozenTrial, FrozenTrial, List[optuna.study.StudyDirection]], bool],
) -> List[List[FrozenTrial]]:
    dominated_count: DefaultDict[int, int] = defaultdict(int)
    dominates_list = defaultdict(list)

    for p, q in itertools.combinations(population, 2):
        if dominates(p, q, directions):
            dominates_list[p.number].append(q.number)
            dominated_count[q.number] += 1
        elif dominates(q, p, directions):
            dominates_list[q.number].append(p.number)
            dominated_count[p.number] += 1

    population_per_rank = []
    while population:
        non_dominated_population = []
        i = 0
        while i < len(population):
            if dominated_count[population[i].number] == 0:
                individual = population[i]
                if i == len(population) - 1:
                    population.pop()
                else:
                    population[i] = population.pop()
                non_dominated_population.append(individual)
            else:
                i += 1

        for x in non_dominated_population:
            for y in dominates_list[x.number]:
                dominated_count[y] -= 1

        assert non_dominated_population
        population_per_rank.append(non_dominated_population)

    return population_per_rank
