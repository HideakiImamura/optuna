from __future__ import annotations

from collections.abc import Container
from collections.abc import Iterable
from collections.abc import Sequence
import copy
import json
import threading
from typing import Any
import uuid

from optuna.distributions import BaseDistribution
from optuna.distributions import distribution_to_json
from optuna.exceptions import DuplicatedStudyError
from optuna.storages._base import BaseStorage
from optuna.storages._base import DEFAULT_STUDY_NAME_PREFIX
from optuna.storages.grpc._grpc_imports import _imports
from optuna.storages.grpc._server import _from_proto_frozen_trial
from optuna.storages.grpc._server import _to_proto_frozen_trial
from optuna.storages.grpc._server import _to_proto_trial_state
from optuna.study._frozen import FrozenStudy
from optuna.study._study_direction import StudyDirection
from optuna.trial._frozen import FrozenTrial
from optuna.trial._state import TrialState


if _imports.is_successful():
    from optuna.storages.grpc._grpc_imports import api_pb2
    from optuna.storages.grpc._grpc_imports import api_pb2_grpc
    from optuna.storages.grpc._grpc_imports import grpc


class _StudyInfo:
    def __init__(self) -> None:
        # Trial number to corresponding FrozenTrial.
        self.trials: dict[int, FrozenTrial] = {}
        # A list of trials and the last trial number which require storage access to read latest
        # attributes.
        self.unfinished_trial_ids: set[int] = set()
        self.last_finished_trial_id: int = -1
        self.directions: list[StudyDirection] | None = None
        self.name: str | None = None


class GrpcStorageProxy(BaseStorage):
    """gRPC client for :func:`~optuna.storages.grpc.run_grpc_server`.

    Example:

        This is a simple example of using :class:`~optuna.storages.grpc.GrpcStorageProxy` with
        :func:`~optuna.storages.grpc.run_grpc_server`.

        .. code::

            import optuna
            from optuna.storages.grpc import GrpcStorageProxy

            storage = GrpcStorageProxy(host="localhost", port=13000)
            study = optuna.create_study(storage=storage)

        Please refer to the example in :func:`~optuna.storages.grpc.run_grpc_server` for the
        server side code.

    Args:
        host: The host of the gRPC server.
        port: The port of the gRPC server.

    """

    def __init__(self, *, host: str = "localhost", port: int = 13000) -> None:
        self._stub = api_pb2_grpc.StorageServiceStub(
            grpc.insecure_channel(
                f"{host}:{port}",
                options=[("grpc.max_receive_message_length", -1)],
            )
        )  # type: ignore
        self._host = host
        self._port = port
        self._studies: dict[int, _StudyInfo] = {}
        self._trial_id_to_study_id_and_number: dict[int, tuple[int, int]] = {}
        self._study_id_and_number_to_trial_id: dict[tuple[int, int], int] = {}
        self._lock = threading.Lock()

    def __getstate__(self) -> dict[Any, Any]:
        state = self.__dict__.copy()
        del state["_stub"]
        del state["_lock"]
        return state

    def __setstate__(self, state: dict[Any, Any]) -> None:
        self.__dict__.update(state)
        self._stub = api_pb2_grpc.StorageServiceStub(
            grpc.insecure_channel(f"{self._host}:{self._port}")
        )  # type: ignore
        self._lock = threading.Lock()

    def create_new_study(
        self, directions: Sequence[StudyDirection], study_name: str | None = None
    ) -> int:
        request = api_pb2.CreateNewStudyRequest(
            directions=[
                api_pb2.MINIMIZE if d == StudyDirection.MINIMIZE else api_pb2.MAXIMIZE
                for d in directions
            ],
            study_name=study_name
            or DEFAULT_STUDY_NAME_PREFIX
            + str(uuid.uuid4()),  # TODO(HideakiImamura): Check if this is unique.
        )
        try:
            response = self._stub.CreateNewStudy(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.ALREADY_EXISTS:
                raise DuplicatedStudyError from e
            raise

        study_id = response.study_id

        with self._lock:
            study = _StudyInfo()
            study.directions = list(directions)
            study.name = study_name
            self._studies[study_id] = study

        return study_id

    def delete_study(self, study_id: int) -> None:
        request = api_pb2.DeleteStudyRequest(study_id=study_id)
        try:
            self._stub.DeleteStudy(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            raise

        with self._lock:
            if study_id in self._studies:
                for trial_number in self._studies[study_id].trials:
                    trial_id = self._study_id_and_number_to_trial_id.get((study_id, trial_number))
                    if trial_id in self._trial_id_to_study_id_and_number:
                        del self._trial_id_to_study_id_and_number[trial_id]
                    if (study_id, trial_number) in self._study_id_and_number_to_trial_id:
                        del self._study_id_and_number_to_trial_id[(study_id, trial_number)]
                del self._studies[study_id]

    def set_study_user_attr(self, study_id: int, key: str, value: Any) -> None:
        request = api_pb2.SetStudyUserAttributeRequest(
            study_id=study_id, key=key, value=json.dumps(value)
        )
        try:
            self._stub.SetStudyUserAttribute(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            raise

    def set_study_system_attr(self, study_id: int, key: str, value: Any) -> None:
        request = api_pb2.SetStudySystemAttributeRequest(
            study_id=study_id, key=key, value=json.dumps(value)
        )
        try:
            self._stub.SetStudySystemAttribute(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            raise

    def get_study_id_from_name(self, study_name: str) -> int:
        request = api_pb2.GetStudyIdFromNameRequest(study_name=study_name)
        try:
            response = self._stub.GetStudyIdFromName(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            raise
        return response.study_id

    def get_study_name_from_id(self, study_id: int) -> str:
        with self._lock:
            if study_id in self._studies:
                name = self._studies[study_id].name
                if name is not None:
                    return name

        request = api_pb2.GetStudyNameFromIdRequest(study_id=study_id)
        try:
            response = self._stub.GetStudyNameFromId(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            raise

        name = response.study_name
        assert name is not None
        with self._lock:
            if study_id not in self._studies:
                self._studies[study_id] = _StudyInfo()
            self._studies[study_id].name = name

        return name

    def get_study_directions(self, study_id: int) -> list[StudyDirection]:
        with self._lock:
            if study_id in self._studies:
                directions = self._studies[study_id].directions
                if directions is not None:
                    return directions

        request = api_pb2.GetStudyDirectionsRequest(study_id=study_id)
        try:
            response = self._stub.GetStudyDirections(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            raise

        directions = [
            StudyDirection.MINIMIZE if d == api_pb2.MINIMIZE else StudyDirection.MAXIMIZE
            for d in response.directions
        ]
        with self._lock:
            if study_id not in self._studies:
                self._studies[study_id] = _StudyInfo()
            self._studies[study_id].directions = directions
        return directions

    def get_study_user_attrs(self, study_id: int) -> dict[str, Any]:
        request = api_pb2.GetStudyUserAttributesRequest(study_id=study_id)
        try:
            response = self._stub.GetStudyUserAttributes(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            raise
        return {key: json.loads(value) for key, value in response.user_attributes.items()}

    def get_study_system_attrs(self, study_id: int) -> dict[str, Any]:
        request = api_pb2.GetStudySystemAttributesRequest(study_id=study_id)
        try:
            response = self._stub.GetStudySystemAttributes(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            raise
        return {key: json.loads(value) for key, value in response.system_attributes.items()}

    def get_all_studies(self) -> list[FrozenStudy]:
        request = api_pb2.GetAllStudiesRequest()
        response = self._stub.GetAllStudies(request)
        return [
            FrozenStudy(
                study_id=study.study_id,
                study_name=study.study_name,
                direction=None,
                directions=[
                    StudyDirection.MINIMIZE if d == api_pb2.MINIMIZE else StudyDirection.MAXIMIZE
                    for d in study.directions
                ],
                user_attrs={
                    key: json.loads(value) for key, value in study.user_attributes.items()
                },
                system_attrs={
                    key: json.loads(value) for key, value in study.system_attributes.items()
                },
            )
            for study in response.frozen_studies
        ]

    def create_new_trial(self, study_id: int, template_trial: FrozenTrial | None = None) -> int:
        if template_trial is None:
            request = api_pb2.CreateNewTrialRequest(study_id=study_id, template_trial_is_none=True)
        else:
            request = api_pb2.CreateNewTrialRequest(
                study_id=study_id,
                template_trial=_to_proto_frozen_trial(template_trial),
                template_trial_is_none=False,
            )
        try:
            response = self._stub.CreateNewTrial(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            raise
        # return response.trial_id

        frozen_trial = _from_proto_frozen_trial(response.frozen_trial)
        trial_id = frozen_trial._trial_id
        with self._lock:
            if study_id not in self._studies:
                self._studies[study_id] = _StudyInfo()
            study = self._studies[study_id]
            self._add_trials_to_cache(study_id, [frozen_trial])
            # Since finished trials will not be modified by any worker, we do not
            # need storage access for them.
            if frozen_trial.state.is_finished():
                study.last_finished_trial_id = max(study.last_finished_trial_id, trial_id)
            else:
                study.unfinished_trial_ids.add(trial_id)
        return trial_id

    def set_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: BaseDistribution,
    ) -> None:
        request = api_pb2.SetTrialParameterRequest(
            trial_id=trial_id,
            param_name=param_name,
            param_value_internal=param_value_internal,
            distribution=distribution_to_json(distribution),
        )
        try:
            self._stub.SetTrialParameter(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            elif e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                raise RuntimeError from e
            elif e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                raise ValueError from e
            else:
                raise

    def set_trial_state_values(
        self, trial_id: int, state: TrialState, values: Sequence[float] | None = None
    ) -> bool:
        request = api_pb2.SetTrialStateValuesRequest(
            trial_id=trial_id,
            state=_to_proto_trial_state(state),
            values=values,
        )
        try:
            response = self._stub.SetTrialStateValues(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            elif e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                raise RuntimeError from e
            else:
                raise

        return response.trial_updated

    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None:
        request = api_pb2.SetTrialIntermediateValueRequest(
            trial_id=trial_id, step=step, intermediate_value=intermediate_value
        )
        try:
            self._stub.SetTrialIntermediateValue(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            elif e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                raise RuntimeError from e
            else:
                raise

    def set_trial_user_attr(self, trial_id: int, key: str, value: Any) -> None:
        request = api_pb2.SetTrialUserAttributeRequest(
            trial_id=trial_id, key=key, value=json.dumps(value)
        )
        try:
            self._stub.SetTrialUserAttribute(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            elif e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                raise RuntimeError from e
            else:
                raise

    def set_trial_system_attr(self, trial_id: int, key: str, value: Any) -> None:
        request = api_pb2.SetTrialSystemAttributeRequest(
            trial_id=trial_id, key=key, value=json.dumps(value)
        )
        try:
            self._stub.SetTrialSystemAttribute(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            elif e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                raise RuntimeError from e
            else:
                raise

    def get_trial_id_from_study_id_trial_number(self, study_id: int, trial_number: int) -> int:
        request = api_pb2.GetTrialIdFromStudyIdTrialNumberRequest(
            study_id=study_id, trial_number=trial_number
        )
        key = (study_id, trial_number)
        with self._lock:
            if key in self._study_id_and_number_to_trial_id:
                trial_id = self._study_id_and_number_to_trial_id[key]
                return trial_id

        try:
            response = self._stub.GetTrialIdFromStudyIdTrialNumber(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            raise
        return response.trial_id

    def get_trial(self, trial_id: int) -> FrozenTrial:
        with self._lock:
            trial = self._get_cached_trial(trial_id)
            if trial is not None:
                return trial

        request = api_pb2.GetTrialRequest(trial_id=trial_id)
        try:
            response = self._stub.GetTrial(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            raise
        return _from_proto_frozen_trial(response.frozen_trial)

    def get_all_trials(
        self,
        study_id: int,
        deepcopy: bool = True,
        states: Container[TrialState] | None = None,
    ) -> list[FrozenTrial]:
        if states is None:
            states = [
                TrialState.RUNNING,
                TrialState.COMPLETE,
                TrialState.PRUNED,
                TrialState.FAIL,
                TrialState.WAITING,
            ]
        assert isinstance(states, Iterable)
        self._read_trials_from_remote_storage(study_id)

        with self._lock:
            study = self._studies[study_id]
            # We need to sort trials by their number because some samplers assume this behavior.
            # The following two lines are latency-sensitive.

            trials: dict[int, FrozenTrial] | list[FrozenTrial]

            if states is not None:
                trials = {number: t for number, t in study.trials.items() if t.state in states}
            else:
                trials = study.trials
            trials = list(sorted(trials.values(), key=lambda t: t.number))
            return copy.deepcopy(trials) if deepcopy else trials

    def _get_cached_trial(self, trial_id: int) -> FrozenTrial | None:
        if trial_id not in self._trial_id_to_study_id_and_number:
            return None
        study_id, number = self._trial_id_to_study_id_and_number[trial_id]
        study = self._studies[study_id]
        return study.trials[number] if trial_id not in study.unfinished_trial_ids else None

    def _read_trials_from_remote_storage(self, study_id: int) -> None:
        with self._lock:
            if study_id not in self._studies:
                self._studies[study_id] = _StudyInfo()
            study = self._studies[study_id]

            request = api_pb2.GetTrialsRequest(
                study_id=study_id,
                included_trial_ids=study.unfinished_trial_ids,
                trial_id_greater_than=study.last_finished_trial_id,
            )
            try:
                response = self._stub.GetTrials(request)
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.NOT_FOUND:
                    raise KeyError from e
                raise

            trials = [
                _from_proto_frozen_trial(proto_trial) for proto_trial in response.frozen_trials
            ]
            if not trials:
                return

            self._add_trials_to_cache(study_id, trials)
            for trial in trials:
                if not trial.state.is_finished():
                    study.unfinished_trial_ids.add(trial._trial_id)
                    continue

                study.last_finished_trial_id = max(study.last_finished_trial_id, trial._trial_id)
                if trial._trial_id in study.unfinished_trial_ids:
                    study.unfinished_trial_ids.remove(trial._trial_id)

    def _add_trials_to_cache(self, study_id: int, trials: list[FrozenTrial]) -> None:
        study = self._studies[study_id]
        for trial in trials:
            self._trial_id_to_study_id_and_number[trial._trial_id] = (
                study_id,
                trial.number,
            )
            self._study_id_and_number_to_trial_id[(study_id, trial.number)] = trial._trial_id
            study.trials[trial.number] = trial
