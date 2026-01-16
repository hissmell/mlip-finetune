"""Checkpoint saving callback."""

from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING

from mlip_finetune.callbacks.base import Callback

if TYPE_CHECKING:
    from mlip_finetune.trainers import Trainer


class CheckpointCallback(Callback):
    """
    Callback for saving model checkpoints.
    
    Args:
        save_dir: Directory to save checkpoints
        save_interval: Save every N epochs
        save_best: Whether to track and save best model
    """
    
    def __init__(
        self,
        save_dir: str,
        save_interval: int = 1,
        save_best: bool = True
    ):
        self.save_dir = Path(save_dir)
        self.save_interval = save_interval
        self.save_best = save_best
        self.best_loss = float('inf')
    
    def on_train_begin(self, trainer: "Trainer") -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        # Regular interval saving
        if (epoch + 1) % self.save_interval == 0:
            trainer.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
        
        # Best model saving
        if self.save_best:
            val_loss = metrics.get('val_loss', float('inf'))
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                trainer.save_checkpoint('best_model.pt')

