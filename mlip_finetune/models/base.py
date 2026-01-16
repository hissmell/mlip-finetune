"""Base model wrapper interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn


class BaseModelWrapper(ABC, nn.Module):
    """
    Abstract base class for MLIP model wrappers.
    
    Provides a unified interface for different MLIP backends (NequIP, MACE, etc.)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model = None
        self.r_max = config.get('r_max', 6.0)
    
    @abstractmethod
    def load_pretrained(self, path: str) -> None:
        """Load pre-trained model from checkpoint or package."""
        pass
    
    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            batch: Input batch with atomic data
            
        Returns:
            Dictionary with 'energy', 'forces', and optionally 'stress'
        """
        pass
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model': self.model,
            'config': self.config,
            'state_dict': self.model.state_dict() if self.model else None
        }
        torch.save(checkpoint, path)
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """Get number of model parameters."""
        if self.model is None:
            return 0
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())

