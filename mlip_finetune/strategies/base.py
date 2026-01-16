"""Base strategy class for MLIP fine-tuning."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseStrategy(ABC):
    """
    Abstract base class for fine-tuning strategies.
    
    All fine-tuning strategies must inherit from this class and implement
    the required abstract methods.
    
    Attributes:
        model: The MLIP model to fine-tune
        config: Configuration dictionary
        device: PyTorch device for computation
        loss_coeffs: Loss coefficient weights for energy/force/stress
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """
        Initialize the strategy.
        
        Args:
            model: The MLIP model to fine-tune
            config: Configuration dictionary containing strategy parameters
        """
        self.model = model
        self.config = config
        self.device = torch.device(
            config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.loss_coeffs = config.get('loss_coeffs', {
            'energy': 1.0,
            'force': 100.0,
            'stress': 0.0
        })
    
    @abstractmethod
    def before_task(self, task_id: int, dataloader: Optional[DataLoader] = None) -> None:
        """
        Called before starting training on a new task.
        
        Args:
            task_id: Identifier for the current task
            dataloader: Optional dataloader for the task
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        model_output: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        task_id: int
    ) -> torch.Tensor:
        """
        Compute the training loss for a batch.
        
        Args:
            model_output: Dictionary containing model predictions
            batch: Dictionary containing ground truth data
            task_id: Identifier for the current task
            
        Returns:
            Scalar loss tensor
        """
        pass
    
    @abstractmethod
    def after_task(self, task_id: int, dataloader: Optional[DataLoader] = None) -> None:
        """
        Called after completing training on a task.
        
        Args:
            task_id: Identifier for the completed task
            dataloader: Optional dataloader for the task
        """
        pass
    
    @abstractmethod
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """
        Get the list of parameters that should be trained.
        
        Returns:
            List of trainable parameters
        """
        pass
    
    def to(self, device: torch.device) -> "BaseStrategy":
        """Move strategy to specified device."""
        self.device = device
        self.model = self.model.to(device)
        return self
    
    def state_dict(self) -> Dict[str, Any]:
        """Get strategy state for checkpointing."""
        return {}
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load strategy state from checkpoint."""
        pass

