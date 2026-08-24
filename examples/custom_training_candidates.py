"""Trusted plugin: sequentially generate bounded training candidates."""

from agentic_al import register_training_candidate_generator
from agentic_al.types import TrainingCandidate, TrainingCandidateContext


class ConservativeTrainingCandidates:
    """Use completed trial evidence without changing the fixed compute budget."""

    def propose(self, context: TrainingCandidateContext, count: int) -> list[TrainingCandidate]:
        base_learning_rate = context.base_parameters.get("learning_rate")
        if base_learning_rate is None or "learning_rate" not in context.tunable_parameters:
            return []
        completed_candidates = [
            trial
            for trial in context.completed_trials
            if trial.get("status") == "completed" and trial.get("parameters")
        ]
        factor = 0.70 if not completed_candidates else 0.50
        return [
            TrainingCandidate(
                name=f"conservative_lr_step_{context.inner_step_index}",
                parameters={"learning_rate": base_learning_rate * factor},
                rationale=(
                    "Lower the optimizer step after inspecting aggregate metrics from completed "
                    "inner trials. Epochs, architecture, ensemble size, and label budget stay fixed."
                ),
                source="custom_conservative",
            )
        ][:count]


register_training_candidate_generator(
    "conservative_training",
    lambda config: ConservativeTrainingCandidates(),
)
