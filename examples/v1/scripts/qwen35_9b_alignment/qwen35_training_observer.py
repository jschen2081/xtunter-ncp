"""Read-only ConceptLM diagnostics attached to the native Trainer hook."""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from xtuner.v1.engine.train_engine import TrainStepInfo
    from xtuner.v1.train import Trainer


class ConceptLMTrainingObserver:
    """Report calibrated auxiliary objectives without replacing training math."""

    def connect_trainer(self, trainer: "Trainer") -> None:
        """Check the current recipe's optimizer assignment after initialization.

        Args:
            trainer (Trainer): Native trainer owning the model and optimizer.
        """
        engine = trainer._engine
        muon_ids = {
            id(param)
            for group in engine.optimizer.param_groups
            if group.get("algorithm") == "muon"
            for param in group["params"]
        }
        if not all(id(param) in muon_ids for param in engine.model.codebooks):
            raise ValueError("This alignment recipe requires Muon for all codebooks")

    def __call__(
        self, train_step_info: "TrainStepInfo", step: int, epoch: int | None,
        total_step: int, total_epoch: int | None,
    ) -> None:
        """Add globally reduced diagnostics to the native scalar logs.

        Args:
            train_step_info (TrainStepInfo): Backward results; tensors are read-only.
            step (int): Completed update count before this update.
            epoch (int | None): Current epoch.
            total_step (int): Requested number of updates.
            total_epoch (int | None): Optional epoch limit.
        """
        extra = train_step_info["extra_info"]
        # ConceptLM metrics are already globally reduced per microbatch (targets are
        # SP-normalized); summing over local microbatches yields step totals.
        vq, hlm, targets = (
            extra[key].detach().sum().item()
            for key in ("concept_vq_loss", "concept_hlm_loss", "concept_valid_targets")
        )
        logs = train_step_info["logs_info"]
        total = logs["reduced_llm_loss"]  # The model's loss already includes the auxiliary objectives.
        logs.update(
            reduced_total_loss=total, reduced_lm_loss=total - vq - hlm,
            reduced_vq_token_loss=vq, reduced_hlm_token_loss=hlm,
        )
        logs["valid_targets"] = targets
