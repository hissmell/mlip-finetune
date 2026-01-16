"""Metrics for atomic property predictions."""

from typing import Dict, Optional
import numpy as np
import torch


def compute_energy_mae(
    pred_energy: torch.Tensor,
    true_energy: torch.Tensor,
    n_atoms: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute energy MAE and RMSE.
    
    Args:
        pred_energy: Predicted energies
        true_energy: Ground truth energies
        n_atoms: Number of atoms per structure (for per-atom metrics)
        
    Returns:
        Dictionary with 'energy_mae', 'energy_rmse', and optionally per-atom versions
    """
    pred = pred_energy.detach().cpu().numpy().flatten()
    true = true_energy.detach().cpu().numpy().flatten()
    
    diff = pred - true
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    
    metrics = {
        'energy_mae': mae,
        'energy_rmse': rmse,
    }
    
    if n_atoms is not None:
        n = n_atoms.detach().cpu().numpy().flatten()
        per_atom_diff = diff / n
        metrics['energy_mae_per_atom'] = np.mean(np.abs(per_atom_diff))
        metrics['energy_rmse_per_atom'] = np.sqrt(np.mean(per_atom_diff ** 2))
    
    return metrics


def compute_force_mae(
    pred_forces: torch.Tensor,
    true_forces: torch.Tensor
) -> Dict[str, float]:
    """
    Compute force MAE and RMSE.
    
    Args:
        pred_forces: Predicted forces (N_atoms x 3)
        true_forces: Ground truth forces (N_atoms x 3)
        
    Returns:
        Dictionary with force metrics
    """
    pred = pred_forces.detach().cpu().numpy()
    true = true_forces.detach().cpu().numpy()
    
    diff = pred - true
    
    # Component-wise metrics
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    
    # Magnitude metrics
    pred_mag = np.linalg.norm(pred, axis=1)
    true_mag = np.linalg.norm(true, axis=1)
    mag_diff = pred_mag - true_mag
    mag_mae = np.mean(np.abs(mag_diff))
    
    return {
        'force_mae': mae,
        'force_rmse': rmse,
        'force_magnitude_mae': mag_mae,
    }


def compute_force_cosine(
    pred_forces: torch.Tensor,
    true_forces: torch.Tensor,
    eps: float = 1e-8
) -> Dict[str, float]:
    """
    Compute cosine similarity between predicted and true force vectors.
    
    Args:
        pred_forces: Predicted forces (N_atoms x 3)
        true_forces: Ground truth forces (N_atoms x 3)
        eps: Small value to avoid division by zero
        
    Returns:
        Dictionary with cosine similarity statistics
    """
    pred = pred_forces.detach().cpu().numpy()
    true = true_forces.detach().cpu().numpy()
    
    # Compute dot products and norms
    dot_products = np.sum(pred * true, axis=1)
    pred_norms = np.linalg.norm(pred, axis=1)
    true_norms = np.linalg.norm(true, axis=1)
    
    # Filter out near-zero vectors
    valid_mask = (pred_norms > eps) & (true_norms > eps)
    
    if np.sum(valid_mask) == 0:
        return {
            'force_cosine_mean': 0.0,
            'force_cosine_std': 0.0,
        }
    
    cosine = dot_products[valid_mask] / (pred_norms[valid_mask] * true_norms[valid_mask])
    cosine = np.clip(cosine, -1.0, 1.0)
    
    return {
        'force_cosine_mean': float(np.mean(cosine)),
        'force_cosine_std': float(np.std(cosine)),
        'force_cosine_min': float(np.min(cosine)),
        'force_cosine_max': float(np.max(cosine)),
    }


def compute_all_metrics(
    pred_energy: Optional[torch.Tensor] = None,
    true_energy: Optional[torch.Tensor] = None,
    pred_forces: Optional[torch.Tensor] = None,
    true_forces: Optional[torch.Tensor] = None,
    n_atoms: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute all available metrics.
    
    Args:
        pred_energy: Predicted energies
        true_energy: Ground truth energies
        pred_forces: Predicted forces
        true_forces: Ground truth forces
        n_atoms: Number of atoms per structure
        
    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}
    
    if pred_energy is not None and true_energy is not None:
        metrics.update(compute_energy_mae(pred_energy, true_energy, n_atoms))
    
    if pred_forces is not None and true_forces is not None:
        metrics.update(compute_force_mae(pred_forces, true_forces))
        metrics.update(compute_force_cosine(pred_forces, true_forces))
    
    return metrics

