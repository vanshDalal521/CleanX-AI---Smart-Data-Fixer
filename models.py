from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class CleanxAction(Action):
    """Action for the CleanX environment."""

    operation: Literal["drop_row", "rename_column", "cast_type", "fill_missing", "submit", "none"] = Field(
        ..., description="The cleaning operation to perform."
    )
    args: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments for the operation."
    )


class CleanxObservation(Observation):
    """Observation from the CleanX environment."""

    dataset_preview: str = Field(default="", description="CSV preview of the data.")
    columns: List[str] = Field(default_factory=list, description="Current columns.")
    shape: List[int] = Field(default_factory=lambda: [0, 0], description="[rows, cols]")
    goal: str = Field(default="", description="The specific task objective.")
    last_action_error: Optional[str] = Field(None, description="Error from last action if any.")
    progress: float = Field(0.0, description="Current progress / grading score (0.0 to 1.0).")
