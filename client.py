from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import CleanxAction, CleanxObservation


class CleanxEnv(EnvClient[CleanxAction, CleanxObservation, State]):
    """Client for the Cleanx Environment."""

    def _step_payload(self, action: CleanxAction) -> Dict:
        return {
            "operation": action.operation,
            "args": action.args,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CleanxObservation]:
        obs_data = payload.get("observation", {})
        
        observation = CleanxObservation(
            dataset_preview=obs_data.get("dataset_preview", ""),
            columns=obs_data.get("columns", []),
            shape=obs_data.get("shape", [0, 0]),
            goal=obs_data.get("goal", ""),
            last_action_error=obs_data.get("last_action_error", None),
            progress=obs_data.get("progress", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
