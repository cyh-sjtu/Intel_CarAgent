from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict


class Base_Controller(ABC):
    """Abstract base class for controllers.

    Defines the minimal interface that concrete controllers must implement so
    that higher-level code (e.g. navigation tools) can rely on a consistent API.
    """

    @abstractmethod
    def update_path(self, new_path: List[List[float]]) -> None:
        """Send a new path to the controller. new_path is a list of waypoints.

        Each waypoint is expected to be a list of floats [x, y, z] or
        [x, y, z, yaw].
        """

    @abstractmethod
    def update_status(self, status: str) -> None:
        """Update the controller with a new status message."""

    @abstractmethod
    def update_latest_msg(self, msg: str) -> None:
        """Update the controller with a new latest message."""

    @abstractmethod
    def get_current_state(self) -> Dict[str, Any]:
        """Return a dictionary representing the current state of the controller.

        The dictionary may contain keys like 'position', 'orientation', 'status', etc.
        'position' should be a list of floats [x, y, z] or None when not available.
        'orientation' should be a list of floats [x, y, z, w] or None when not available.
        'status' should be a string indicating the current status (e.g. doing what task) of the controller.
        """

    @abstractmethod
    def get_current_image(self) -> Any:
        """Return the current image from the robot's camera."""

    @abstractmethod
    def check_for_new_messages(self) -> str:
        """Check for new messages from the controller, e.g. arrival notifications."""
