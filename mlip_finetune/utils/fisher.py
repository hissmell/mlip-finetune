"""Fisher Information Matrix computation utilities."""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_fisher_information_matrix(
    model: nn.Module,
    dataloader: DataLoader,
    n_samples: Optional[int] = None,
    diagonal_only: bool = True,
    device: torch.device = None,
    loss_coeffs: Dict[str, float] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute Fisher Information Matrix for model parameters.
    
    The diagonal Fisher approximation is computed as:
    F_ii = E[∂L/∂θ_i)²]
    
    Args:
        model: Neural network model
        dataloader: DataLoader for computing Fisher
        n_samples: Number of samples to use (None = all)
        diagonal_only: Compute only diagonal (recommended for efficiency)
        device: Computation device
        loss_coeffs: Loss coefficients for energy/force/stress
        
    Returns:
        Dictionary mapping parameter names to Fisher values
    """
    if device is None:
        device = next(model.parameters()).device
    
    if loss_coeffs is None:
        loss_coeffs = {'energy': 1.0, 'force': 100.0, 'stress': 0.0}
    
    model.eval()
    
    # Initialize Fisher dictionary
    fisher_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param.data)
    
    # Count samples
    n_processed = 0
    max_samples = n_samples if n_samples else float('inf')
    
    pbar = tqdm(dataloader, desc="Computing Fisher")
    
    for batch in pbar:
        if n_processed >= max_samples:
            break
        
        # Move batch to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        
        # Enable gradients for positions
        if 'pos' in batch:
            batch['pos'].requires_grad_(True)
        
        # Forward pass
        model.zero_grad()
        
        with torch.enable_grad():
            output = model(batch)
            
            # Compute loss
            loss = _compute_loss(output, batch, loss_coeffs)
            
            if loss.requires_grad:
                loss.backward()
        
        # Accumulate squared gradients
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_dict[name] += param.grad.data.pow(2)
        
        batch_size = batch.get('num_nodes', torch.tensor([1])).shape[0]
        n_processed += batch_size
        
        pbar.set_postfix({'samples': n_processed})
    
    # Average over samples
    for name in fisher_dict:
        fisher_dict[name] /= n_processed
    
    logger.info(f"Computed Fisher over {n_processed} samples")
    
    return fisher_dict


def _compute_loss(
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    loss_coeffs: Dict[str, float]
) -> torch.Tensor:
    """Compute weighted loss for Fisher computation."""
    losses = []
    
    # Energy loss
    if 'energy' in output and 'total_energy' in batch:
        if 'batch' in batch:
            n_atoms = batch['batch'].bincount().float()
            pred = output['energy'].squeeze() / n_atoms
            target = batch['total_energy'].squeeze() / n_atoms
        else:
            pred = output['energy'].squeeze()
            target = batch['total_energy'].squeeze()
        
        energy_loss = nn.functional.mse_loss(pred, target)
        losses.append(loss_coeffs.get('energy', 1.0) * energy_loss)
    
    # Force loss
    if 'forces' in output and 'forces' in batch:
        force_loss = nn.functional.mse_loss(output['forces'], batch['forces'])
        losses.append(loss_coeffs.get('force', 100.0) * force_loss)
    
    # Stress loss
    if 'stress' in output and 'stress' in batch and loss_coeffs.get('stress', 0) > 0:
        stress_loss = nn.functional.mse_loss(output['stress'], batch['stress'])
        losses.append(loss_coeffs['stress'] * stress_loss)
    
    if not losses:
        return torch.tensor(0.0, requires_grad=True)
    
    return sum(losses)


def normalize_fisher(
    fisher_dict: Dict[str, torch.Tensor],
    method: str = 'max'
) -> Dict[str, torch.Tensor]:
    """
    Normalize Fisher Information values.
    
    Args:
        fisher_dict: Dictionary of Fisher values
        method: Normalization method ('max', 'mean', 'layer', 'none')
        
    Returns:
        Normalized Fisher dictionary
    """
    if method == 'none':
        return fisher_dict
    
    normalized = {}
    
    if method == 'max':
        # Global max normalization
        max_val = max(f.max().item() for f in fisher_dict.values())
        if max_val > 0:
            for name, fisher in fisher_dict.items():
                normalized[name] = fisher / max_val
        else:
            normalized = fisher_dict
    
    elif method == 'mean':
        # Global mean normalization
        total_sum = sum(f.sum().item() for f in fisher_dict.values())
        total_count = sum(f.numel() for f in fisher_dict.values())
        mean_val = total_sum / total_count if total_count > 0 else 1.0
        
        for name, fisher in fisher_dict.items():
            normalized[name] = fisher / mean_val
    
    elif method == 'layer':
        # Per-layer normalization
        for name, fisher in fisher_dict.items():
            max_val = fisher.max().item()
            if max_val > 0:
                normalized[name] = fisher / max_val
            else:
                normalized[name] = fisher
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def apply_fisher_threshold(
    fisher_dict: Dict[str, torch.Tensor],
    threshold: float = 1e-5
) -> Dict[str, torch.Tensor]:
    """
    Apply threshold to Fisher values (set small values to zero).
    
    Args:
        fisher_dict: Dictionary of Fisher values
        threshold: Minimum value threshold
        
    Returns:
        Thresholded Fisher dictionary
    """
    thresholded = {}
    
    for name, fisher in fisher_dict.items():
        thresholded[name] = torch.where(
            fisher > threshold,
            fisher,
            torch.zeros_like(fisher)
        )
    
    return thresholded


def save_fisher_info(
    fisher_dict: Dict[str, torch.Tensor],
    filepath: str,
    optimal_params: Optional[Dict[str, torch.Tensor]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save Fisher information to file.
    
    Args:
        fisher_dict: Fisher information dictionary
        filepath: Output file path
        optimal_params: Optional optimal parameter values
        metadata: Optional metadata dictionary
    """
    save_data = {
        'fisher_information': {k: v.cpu() for k, v in fisher_dict.items()},
        'metadata': metadata or {}
    }
    
    if optimal_params:
        save_data['optimal_params'] = {k: v.cpu() for k, v in optimal_params.items()}
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_data, filepath)
    
    logger.info(f"Saved Fisher information to {filepath}")


def load_fisher_info(filepath: str) -> Dict[str, Any]:
    """
    Load Fisher information from file.
    
    Args:
        filepath: Path to saved Fisher file
        
    Returns:
        Dictionary with 'fisher_information', 'optimal_params', 'metadata'
    """
    data = torch.load(filepath, map_location='cpu', weights_only=False)
    logger.info(f"Loaded Fisher information from {filepath}")
    return data

