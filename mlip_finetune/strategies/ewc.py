"""Elastic Weight Consolidation (EWC) strategy for continual learning."""

import logging
from typing import List, Dict, Any, Optional
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mlip_finetune.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class EWCStrategy(BaseStrategy):
    """
    Elastic Weight Consolidation (EWC) strategy for continual learning.
    
    EWC prevents catastrophic forgetting by adding a regularization term
    that penalizes changes to important parameters (measured by Fisher Information).
    
    Loss = L_new(θ) + λ/2 * Σ_i F_i * (θ_i - θ*_i)²
    
    Where:
        - L_new: Loss on new task
        - λ: Regularization strength (ewc_lambda)
        - F_i: Fisher Information for parameter i
        - θ*_i: Optimal parameters from previous task
    
    Args:
        model: The MLIP model to fine-tune
        config: Configuration dictionary with EWC-specific parameters:
            - ewc_lambda: Regularization strength (default: 1000.0)
            - fisher_samples: Number of samples for Fisher computation
            - diagonal_only: Use diagonal Fisher approximation (default: True)
            - precomputed_fisher_path: Path to pre-computed Fisher matrix
    
    Example:
        >>> config = {
        ...     'ewc_lambda': 1000.0,
        ...     'precomputed_fisher_path': 'fisher_matrices/fim.pt'
        ... }
        >>> strategy = EWCStrategy(model, config)
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        
        # EWC parameters
        self.ewc_lambda = config.get('ewc_lambda', 1000.0)
        self.fisher_samples = config.get('fisher_samples', None)
        self.diagonal_only = config.get('diagonal_only', True)
        self.fisher_normalize = config.get('fisher_normalize', 'max')
        self.fisher_threshold = config.get('fisher_threshold', 0.0)
        
        # Online EWC for multiple tasks
        self.online_ewc = config.get('online_ewc', False)
        self.gamma = config.get('gamma', 1.0)
        
        # Storage for Fisher information and optimal parameters
        self.fisher_dict: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self.task_count = 0
        
        # Loss component tracking
        self.last_task_loss: Optional[float] = None
        self.last_ewc_penalty: Optional[float] = None
        
        # Pre-computed Fisher loading
        self.use_precomputed_fisher = False
        precomputed_path = config.get('precomputed_fisher_path')
        if precomputed_path:
            self.load_precomputed_fisher(precomputed_path)
            self.use_precomputed_fisher = True
    
    def load_precomputed_fisher(self, path: str) -> None:
        """Load pre-computed Fisher information from file."""
        from mlip_finetune.utils.fisher import load_fisher_info
        
        logger.info(f"Loading pre-computed Fisher from: {path}")
        fisher_data = load_fisher_info(path)
        
        self.fisher_dict = fisher_data['fisher_information']
        self.optimal_params = fisher_data.get('optimal_params', {})
        
        # Move to device
        for name in self.fisher_dict:
            self.fisher_dict[name] = self.fisher_dict[name].to(self.device)
        for name in self.optimal_params:
            self.optimal_params[name] = self.optimal_params[name].to(self.device)
        
        # If no optimal params saved, use current model params
        if not self.optimal_params:
            logger.info("No optimal params in Fisher file, using current model params")
            for name, param in self.model.named_parameters():
                if name in self.fisher_dict:
                    self.optimal_params[name] = param.data.clone().detach()
        
        n_params = sum(f.numel() for f in self.fisher_dict.values())
        logger.info(f"Loaded Fisher for {len(self.fisher_dict)} layers, {n_params:,} parameters")
    
    def before_task(self, task_id: int, dataloader: Optional[DataLoader] = None) -> None:
        """Log EWC regularization info before training."""
        logger.info(f"EWC: Starting task {task_id}")
        
        if self.fisher_dict:
            n_important = sum(
                (f > self.fisher_threshold).sum().item()
                for f in self.fisher_dict.values()
            )
            logger.info(f"EWC: Regularizing {n_important:,} important parameters")
            logger.info(f"EWC lambda: {self.ewc_lambda}")
    
    def compute_loss(
        self,
        model_output: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        task_id: int
    ) -> torch.Tensor:
        """Compute EWC loss: task loss + regularization penalty."""
        # Standard task loss
        task_loss = self._compute_task_loss(model_output, batch)
        
        # EWC regularization penalty
        ewc_penalty = torch.tensor(0.0, device=self.device)
        if self.fisher_dict and self.optimal_params:
            ewc_penalty = self._compute_ewc_penalty()
        
        total_loss = task_loss + self.ewc_lambda * ewc_penalty
        
        # Store for tracking
        self.last_task_loss = task_loss.item()
        self.last_ewc_penalty = ewc_penalty.item()
        
        return total_loss
    
    def _compute_task_loss(
        self,
        model_output: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute standard task loss (energy + force + stress)."""
        losses = []
        
        # Energy loss
        if 'energy' in model_output and 'total_energy' in batch:
            if 'batch' in batch:
                n_atoms = batch['batch'].bincount().float()
                pred = model_output['energy'].squeeze() / n_atoms
                target = batch['total_energy'].squeeze() / n_atoms
            else:
                pred = model_output['energy'].squeeze()
                target = batch['total_energy'].squeeze()
            losses.append(self.loss_coeffs['energy'] * nn.functional.mse_loss(pred, target))
        
        # Force loss
        if 'forces' in model_output and 'forces' in batch:
            force_loss = nn.functional.mse_loss(model_output['forces'], batch['forces'])
            losses.append(self.loss_coeffs['force'] * force_loss)
        
        # Stress loss
        if ('stress' in model_output and 'stress' in batch 
                and self.loss_coeffs.get('stress', 0) > 0):
            stress_loss = nn.functional.mse_loss(model_output['stress'], batch['stress'])
            losses.append(self.loss_coeffs['stress'] * stress_loss)
        
        return sum(losses) if losses else torch.tensor(0.0, device=self.device)
    
    def _compute_ewc_penalty(self) -> torch.Tensor:
        """Compute EWC penalty: 1/2 * Σ F_i * (θ_i - θ*_i)²"""
        penalty = torch.tensor(0.0, device=self.device)
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_dict and name in self.optimal_params:
                fisher = self.fisher_dict[name]
                optimal = self.optimal_params[name]
                
                # Ensure same device
                if fisher.device != param.device:
                    fisher = fisher.to(param.device)
                if optimal.device != param.device:
                    optimal = optimal.to(param.device)
                
                diff = param - optimal
                penalty = penalty + (fisher * diff.pow(2)).sum()
        
        return penalty * 0.5
    
    def after_task(self, task_id: int, dataloader: Optional[DataLoader] = None) -> None:
        """Compute and store Fisher information after task completion."""
        # Skip Fisher computation if using precomputed Fisher
        if self.use_precomputed_fisher:
            logger.info("Using precomputed Fisher, skipping Fisher computation after task")
            return
        
        if dataloader is None:
            logger.warning("No dataloader provided for Fisher computation")
            return
        
        logger.info(f"Computing Fisher Information for task {task_id}")
        
        from mlip_finetune.utils.fisher import (
            compute_fisher_information_matrix,
            normalize_fisher,
            apply_fisher_threshold
        )
        
        new_fisher = compute_fisher_information_matrix(
            model=self.model,
            dataloader=dataloader,
            n_samples=self.fisher_samples,
            diagonal_only=self.diagonal_only,
            device=self.device,
            loss_coeffs=self.loss_coeffs
        )
        
        # Normalize
        if self.fisher_normalize != 'none':
            new_fisher = normalize_fisher(new_fisher, method=self.fisher_normalize)
        
        # Threshold
        if self.fisher_threshold > 0:
            new_fisher = apply_fisher_threshold(new_fisher, threshold=self.fisher_threshold)
        
        # Online EWC: combine with previous Fisher
        if self.online_ewc and self.fisher_dict:
            for name in new_fisher:
                if name in self.fisher_dict:
                    self.fisher_dict[name] = (
                        self.gamma * self.fisher_dict[name] + new_fisher[name]
                    )
                else:
                    self.fisher_dict[name] = new_fisher[name]
        else:
            self.fisher_dict = new_fisher
        
        # Store optimal parameters
        self.optimal_params = {}
        for name, param in self.model.named_parameters():
            if name in self.fisher_dict:
                self.optimal_params[name] = param.data.clone().detach()
        
        self.task_count += 1
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """All parameters are trainable in EWC."""
        return list(self.model.parameters())
    
    def get_loss_components(self) -> Dict[str, float]:
        """Get components of the last computed loss."""
        if self.last_task_loss is None:
            return {}
        return {
            'task_loss': self.last_task_loss,
            'ewc_penalty': self.last_ewc_penalty,
            'ewc_penalty_scaled': self.ewc_lambda * self.last_ewc_penalty,
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """Get strategy state for checkpointing."""
        return {
            'fisher_dict': {k: v.cpu() for k, v in self.fisher_dict.items()},
            'optimal_params': {k: v.cpu() for k, v in self.optimal_params.items()},
            'task_count': self.task_count,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load strategy state from checkpoint."""
        self.fisher_dict = {
            k: v.to(self.device) for k, v in state_dict['fisher_dict'].items()
        }
        self.optimal_params = {
            k: v.to(self.device) for k, v in state_dict['optimal_params'].items()
        }
        self.task_count = state_dict['task_count']

