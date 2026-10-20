from abc import ABC, abstractmethod

from .models import Pose


class RobotController(ABC):
    """Common contract for simulated and physical grasp execution."""

    @abstractmethod
    def move_to_grasp(self, final_pose: Pose) -> None:
        """Move to the final object grasp pose after perception/refinement."""
        raise NotImplementedError

    @abstractmethod
    def acknowledge_grasp(self) -> bool:
        """Return whether the gripper/robot confirms a successful grasp."""
        raise NotImplementedError

    def grasp(self, final_pose: Pose) -> bool:
        self.move_to_grasp(final_pose)
        return self.acknowledge_grasp()
