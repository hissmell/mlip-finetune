"""TensorBoard logging callback."""

from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING

from mlip_finetune.callbacks.base import Callback

if TYPE_CHECKING:
    from mlip_finetune.trainers import Trainer


class TensorBoardCallback(Callback):
    """
    Callback for TensorBoard logging.
    
    Args:
        log_dir: Directory for TensorBoard logs
    """
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.writer = None
    
    def on_train_begin(self, trainer: "Trainer") -> None:
        from torch.utils.tensorboard import SummaryWriter
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
    
    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        if self.writer is None:
            return
        
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, epoch)
        
        # Log learning rate
        if trainer.optimizer is not None:
            lr = trainer.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('learning_rate', lr, epoch)
    
    def on_train_end(self, trainer: "Trainer") -> None:
        if self.writer is not None:
            self.writer.close()

