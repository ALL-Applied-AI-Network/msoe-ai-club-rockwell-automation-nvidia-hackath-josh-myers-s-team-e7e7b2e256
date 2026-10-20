from abc import ABC, abstractmethod
from collections.abc import Sequence

from .models import TrialResult


class Evaluator(ABC):
    """Shared evaluation criteria for simulation and physical trials."""

    criteria = (
        "successful_picks_per_attempt",
        "placement_accuracy",
        "cycle_time",
        "failed_grasp_recovery",
        "unseen_object_performance",
    )

    @abstractmethod
    def evaluate(self, trial: TrialResult) -> dict[str, float]:
        raise NotImplementedError

    def summarize(self, trials: Sequence[TrialResult]) -> dict[str, float]:
        if not trials:
            return {"successful_picks_per_attempt": 0.0}
        return {
            "successful_picks_per_attempt": sum(t.success for t in trials) / len(trials),
            "mean_cycle_time_s": sum(t.cycle_time_s or 0.0 for t in trials) / len(trials),
            "mean_placement_error_m": sum(t.placement_error_m or 0.0 for t in trials) / len(trials),
        }

class DigitalEvaluator(Evaluator):
    def evaluate(self, trial: TrialResult) -> dict[str, float]:
        return {
            "success": float(trial.success),
            "placement_accuracy": 1.0 if trial.placement_error_m in (None, 0.0) else 0.0,
            "cycle_time_s": trial.cycle_time_s or 0.0,
        }

class RealityEvaluator(Evaluator):
    def evaluate(self, trial: TrialResult) -> dict[str, float]:
        return {
            "success": float(trial.success),
            "placement_accuracy": 1.0 if trial.placement_error_m in (None, 0.0) else 0.0,
            "cycle_time_s": trial.cycle_time_s or 0.0,
            "recovery_used": float(trial.recovery_used),
        }
