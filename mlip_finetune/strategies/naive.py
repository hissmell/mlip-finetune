"""Naive fine-tuning strategy without regularization."""

import logging
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mlip_finetune.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class NaiveStrategy(BaseStrategy):
    """
    Naive fine-tuning strategy.
    
    This is the simplest fine-tuning approach - just train on the new data
    without any regularization to preserve previous knowledge.
    
    Use this as a baseline or when you don't need to retain knowledge
    from the pre-training dataset.
    
    Example:
        >>> strategy = NaiveStrategy(model, config)
        >>> loss = strategy.compute_loss(output, batch, task_id=0)
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        self.use_per_atom_energy = config.get('loss_per_atom_energy', True)
        self._logged_scale = False
    
    def before_task(self, task_id: int, dataloader: Optional[DataLoader] = None) -> None:
        """No preparation needed for naive strategy."""
        pass
    
    def compute_loss(
        self,
        model_output: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        task_id: int
    ) -> torch.Tensor:
        """
        Compute standard MSE loss for energy and forces.
        
        Args:
            model_output: Model predictions with 'energy' and 'forces' keys
            batch: Ground truth batch with corresponding keys
            task_id: Task identifier (unused in naive strategy)
            
        Returns:
            Weighted sum of energy and force losses
        """
        losses = []
        
        # Energy loss
        if 'energy' in model_output and 'total_energy' in batch:
            energy_loss = self._compute_energy_loss(model_output, batch)
            if energy_loss is not None:
                losses.append(self.loss_coeffs['energy'] * energy_loss)
        
        # Force loss
        if 'forces' in model_output and 'forces' in batch:
            force_loss = nn.functional.mse_loss(
                model_output['forces'],
                batch['forces']
            )
            losses.append(self.loss_coeffs['force'] * force_loss)
        
        # Stress loss (optional)
        if ('stress' in model_output and 'stress' in batch 
                and self.loss_coeffs.get('stress', 0) > 0):
            stress_loss = nn.functional.mse_loss(
                model_output['stress'],
                batch['stress']
            )
            losses.append(self.loss_coeffs['stress'] * stress_loss)
        
        if not losses:
            raise ValueError(
                f"No valid loss components found. "
                f"Model outputs: {list(model_output.keys())}, "
                f"Batch keys: {list(batch.keys())}"
            )
        
        return sum(losses)
    
    def _compute_energy_loss(
        self,
        model_output: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Compute energy loss with optional per-atom normalization."""
        if self.use_per_atom_energy and 'batch' in batch:
            # Per-atom energy loss for more stable training
            n_atoms = batch['batch'].bincount().float()
            pred_per_atom = model_output['energy'].squeeze() / n_atoms
            target_per_atom = batch['total_energy'].squeeze() / n_atoms
            
            if not self._logged_scale:
                logger.info(f"Using per-atom energy loss")
                logger.info(f"Example energy: {batch['total_energy'][0].item():.2f} eV, "
                           f"N atoms: {n_atoms[0].item():.0f}")
                self._logged_scale = True
            
            return nn.functional.mse_loss(pred_per_atom, target_per_atom)
        else:
            return nn.functional.mse_loss(
                model_output['energy'].squeeze(),
                batch['total_energy'].squeeze()
            )
    
    def after_task(self, task_id: int, dataloader: Optional[DataLoader] = None) -> None:
        """No cleanup needed for naive strategy."""
        pass
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """All parameters are trainable in naive strategy."""
        return list(self.model.parameters())

