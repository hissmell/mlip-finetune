"""Base callback class and callback list."""

from typing import Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from mlip_finetune.trainers import Trainer


class Callback:
    """Base class for training callbacks."""
    
    def on_train_begin(self, trainer: "Trainer") -> None:
        """Called at the start of training."""
        pass
    
    def on_train_end(self, trainer: "Trainer") -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, trainer: "Trainer", batch_idx: int) -> None:
        """Called at the start of each batch."""
        pass
    
    def on_batch_end(
        self,
        trainer: "Trainer",
        batch_idx: int,
        loss: float
    ) -> None:
        """Called at the end of each batch."""
        pass


class CallbackList:
    """Container for managing multiple callbacks."""
    
    def __init__(self, callbacks: List[Callback] = None):
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)
    
    def on_train_begin(self, trainer: "Trainer") -> None:
        for callback in self.callbacks:
            callback.on_train_begin(trainer)
    
    def on_train_end(self, trainer: "Trainer") -> None:
        for callback in self.callbacks:
            callback.on_train_end(trainer)
    
    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch)
    
    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, metrics)
    
    def on_batch_begin(self, trainer: "Trainer", batch_idx: int) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin(trainer, batch_idx)
    
    def on_batch_end(
        self,
        trainer: "Trainer",
        batch_idx: int,
        loss: float
    ) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch_idx, loss)

