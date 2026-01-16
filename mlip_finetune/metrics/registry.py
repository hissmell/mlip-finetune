"""Metric registry for configurable metric computation."""

from typing import Dict, List, Any, Optional, Callable
import torch
import numpy as np


class MetricRegistry:
    """
    Registry for metrics that can be computed during training.
    
    Allows users to specify which metrics to track via config,
    and computes them for train/val/test splits.
    
    Available metrics:
        - energy_mae: Mean Absolute Error for energy
        - energy_rmse: Root Mean Squared Error for energy
        - energy_mae_per_atom: Per-atom energy MAE
        - force_mae: Mean Absolute Error for forces
        - force_rmse: Root Mean Squared Error for forces
        - force_cosine: Cosine similarity for forces
        - force_magnitude_mae: MAE for force magnitudes
    
    Example config:
        metrics:
          - energy_mae
          - energy_rmse
          - force_mae
          - force_cosine
    
    Example usage:
        >>> registry = MetricRegistry(['energy_mae', 'force_mae', 'force_cosine'])
        >>> registry.update(pred_energy, true_energy, pred_forces, true_forces)
        >>> metrics = registry.compute()
        >>> registry.reset()
    """
    
    # All available metrics
    AVAILABLE_METRICS = [
        'energy_mae',
        'energy_rmse', 
        'energy_mae_per_atom',
        'energy_rmse_per_atom',
        'force_mae',
        'force_rmse',
        'force_magnitude_mae',
        'force_cosine',
        'force_cosine_std',
    ]
    
    def __init__(self, metrics: List[str]):
        """
        Initialize registry with list of metrics to compute.
        
        Args:
            metrics: List of metric names to track
        """
        self.metrics = metrics
        self._validate_metrics()
        
        # Accumulators
        self._pred_energy: List[torch.Tensor] = []
        self._true_energy: List[torch.Tensor] = []
        self._pred_forces: List[torch.Tensor] = []
        self._true_forces: List[torch.Tensor] = []
        self._n_atoms: List[torch.Tensor] = []
    
    def _validate_metrics(self):
        """Validate that all requested metrics are available."""
        for m in self.metrics:
            if m not in self.AVAILABLE_METRICS:
                raise ValueError(
                    f"Unknown metric: '{m}'. Available: {self.AVAILABLE_METRICS}"
                )
    
    def reset(self):
        """Reset all accumulators."""
        self._pred_energy = []
        self._true_energy = []
        self._pred_forces = []
        self._true_forces = []
        self._n_atoms = []
    
    def update(
        self,
        pred_energy: Optional[torch.Tensor] = None,
        true_energy: Optional[torch.Tensor] = None,
        pred_forces: Optional[torch.Tensor] = None,
        true_forces: Optional[torch.Tensor] = None,
        n_atoms: Optional[torch.Tensor] = None,
    ):
        """
        Update accumulators with batch predictions.
        
        Args:
            pred_energy: Predicted energies
            true_energy: Ground truth energies
            pred_forces: Predicted forces
            true_forces: Ground truth forces
            n_atoms: Number of atoms per structure
        """
        if pred_energy is not None:
            self._pred_energy.append(pred_energy.detach().cpu())
        if true_energy is not None:
            self._true_energy.append(true_energy.detach().cpu())
        if pred_forces is not None:
            self._pred_forces.append(pred_forces.detach().cpu())
        if true_forces is not None:
            self._true_forces.append(true_forces.detach().cpu())
        if n_atoms is not None:
            self._n_atoms.append(n_atoms.detach().cpu())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all registered metrics.
        
        Returns:
            Dictionary mapping metric names to values
        """
        results = {}
        
        # Concatenate accumulated tensors
        pred_energy = torch.cat(self._pred_energy) if self._pred_energy else None
        true_energy = torch.cat(self._true_energy) if self._true_energy else None
        pred_forces = torch.cat(self._pred_forces) if self._pred_forces else None
        true_forces = torch.cat(self._true_forces) if self._true_forces else None
        n_atoms = torch.cat(self._n_atoms) if self._n_atoms else None
        
        # Compute each requested metric
        for metric_name in self.metrics:
            value = self._compute_single_metric(
                metric_name, 
                pred_energy, true_energy,
                pred_forces, true_forces,
                n_atoms
            )
            if value is not None:
                results[metric_name] = value
        
        return results
    
    def _compute_single_metric(
        self,
        metric_name: str,
        pred_energy: Optional[torch.Tensor],
        true_energy: Optional[torch.Tensor],
        pred_forces: Optional[torch.Tensor],
        true_forces: Optional[torch.Tensor],
        n_atoms: Optional[torch.Tensor],
    ) -> Optional[float]:
        """Compute a single metric."""
        
        # Energy metrics
        if metric_name.startswith('energy') and pred_energy is not None and true_energy is not None:
            pred = pred_energy.numpy().flatten()
            true = true_energy.numpy().flatten()
            diff = pred - true
            
            if metric_name == 'energy_mae':
                return float(np.mean(np.abs(diff)))
            elif metric_name == 'energy_rmse':
                return float(np.sqrt(np.mean(diff ** 2)))
            elif metric_name == 'energy_mae_per_atom' and n_atoms is not None:
                n = n_atoms.numpy().flatten()
                return float(np.mean(np.abs(diff / n)))
            elif metric_name == 'energy_rmse_per_atom' and n_atoms is not None:
                n = n_atoms.numpy().flatten()
                return float(np.sqrt(np.mean((diff / n) ** 2)))
        
        # Force metrics
        if metric_name.startswith('force') and pred_forces is not None and true_forces is not None:
            pred = pred_forces.numpy()
            true = true_forces.numpy()
            diff = pred - true
            
            if metric_name == 'force_mae':
                return float(np.mean(np.abs(diff)))
            elif metric_name == 'force_rmse':
                return float(np.sqrt(np.mean(diff ** 2)))
            elif metric_name == 'force_magnitude_mae':
                pred_mag = np.linalg.norm(pred, axis=1)
                true_mag = np.linalg.norm(true, axis=1)
                return float(np.mean(np.abs(pred_mag - true_mag)))
            elif metric_name == 'force_cosine':
                dot = np.sum(pred * true, axis=1)
                pred_norm = np.linalg.norm(pred, axis=1)
                true_norm = np.linalg.norm(true, axis=1)
                valid = (pred_norm > 1e-8) & (true_norm > 1e-8)
                if np.sum(valid) == 0:
                    return 0.0
                cosine = dot[valid] / (pred_norm[valid] * true_norm[valid])
                return float(np.mean(np.clip(cosine, -1, 1)))
            elif metric_name == 'force_cosine_std':
                dot = np.sum(pred * true, axis=1)
                pred_norm = np.linalg.norm(pred, axis=1)
                true_norm = np.linalg.norm(true, axis=1)
                valid = (pred_norm > 1e-8) & (true_norm > 1e-8)
                if np.sum(valid) == 0:
                    return 0.0
                cosine = dot[valid] / (pred_norm[valid] * true_norm[valid])
                return float(np.std(np.clip(cosine, -1, 1)))
        
        return None


def get_metric(name: str) -> Callable:
    """Get a metric function by name."""
    from mlip_finetune.metrics.atomic import (
        compute_energy_mae,
        compute_force_mae,
        compute_force_cosine,
    )
    
    metrics = {
        'energy_mae': compute_energy_mae,
        'force_mae': compute_force_mae,
        'force_cosine': compute_force_cosine,
    }
    
    if name not in metrics:
        raise ValueError(f"Unknown metric: {name}")
    
    return metrics[name]

