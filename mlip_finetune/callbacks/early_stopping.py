"""Early stopping callback."""

import logging
from typing import Dict, Any, TYPE_CHECKING

from mlip_finetune.callbacks.base import Callback

if TYPE_CHECKING:
    from mlip_finetune.trainers import Trainer

logger = logging.getLogger(__name__)


class EarlyStoppingCallback(Callback):
    """
    Callback for early stopping based on validation loss.
    
    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        monitor: Metric to monitor (default: val_loss)
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-6,
        monitor: str = 'val_loss'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_value = float('inf')
        self.counter = 0
        self.should_stop = False
    
    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        current = metrics.get(self.monitor)
        
        if current is None:
            return
        
        if current < self.best_value - self.min_delta:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(no improvement for {self.patience} epochs)"
                )

