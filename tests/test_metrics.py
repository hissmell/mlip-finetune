"""Tests for evaluation metrics."""

import pytest
import torch
import numpy as np


class TestEnergyMetrics:
    """Tests for energy metrics."""
    
    def test_energy_mae(self):
        from mlip_finetune.metrics import compute_energy_mae
        
        pred = torch.tensor([1.0, 2.0, 3.0])
        true = torch.tensor([1.1, 2.2, 2.8])
        
        metrics = compute_energy_mae(pred, true)
        
        assert 'energy_mae' in metrics
        assert 'energy_rmse' in metrics
        assert metrics['energy_mae'] > 0
    
    def test_energy_mae_per_atom(self):
        from mlip_finetune.metrics import compute_energy_mae
        
        pred = torch.tensor([10.0, 20.0, 30.0])
        true = torch.tensor([10.5, 19.5, 30.5])
        n_atoms = torch.tensor([10, 10, 10])
        
        metrics = compute_energy_mae(pred, true, n_atoms)
        
        assert 'energy_mae_per_atom' in metrics
        assert 'energy_rmse_per_atom' in metrics


class TestForceMetrics:
    """Tests for force metrics."""
    
    def test_force_mae(self):
        from mlip_finetune.metrics import compute_force_mae
        
        pred = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        true = torch.tensor([[1.1, 0.1, 0.0], [0.0, 0.9, 0.1]])
        
        metrics = compute_force_mae(pred, true)
        
        assert 'force_mae' in metrics
        assert 'force_rmse' in metrics
        assert 'force_magnitude_mae' in metrics
    
    def test_force_cosine(self):
        from mlip_finetune.metrics import compute_force_cosine
        
        # Parallel vectors
        pred = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        true = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        
        metrics = compute_force_cosine(pred, true)
        
        assert 'force_cosine_mean' in metrics
        assert abs(metrics['force_cosine_mean'] - 1.0) < 1e-6  # Perfect alignment
    
    def test_force_cosine_orthogonal(self):
        from mlip_finetune.metrics import compute_force_cosine
        
        # Orthogonal vectors
        pred = torch.tensor([[1.0, 0.0, 0.0]])
        true = torch.tensor([[0.0, 1.0, 0.0]])
        
        metrics = compute_force_cosine(pred, true)
        
        assert abs(metrics['force_cosine_mean']) < 1e-6  # Orthogonal = 0


class TestAllMetrics:
    """Tests for combined metrics."""
    
    def test_compute_all(self):
        from mlip_finetune.metrics import compute_all_metrics
        
        pred_energy = torch.tensor([1.0, 2.0])
        true_energy = torch.tensor([1.1, 2.1])
        pred_forces = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        true_forces = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        
        metrics = compute_all_metrics(
            pred_energy=pred_energy,
            true_energy=true_energy,
            pred_forces=pred_forces,
            true_forces=true_forces,
        )
        
        assert 'energy_mae' in metrics
        assert 'force_mae' in metrics
        assert 'force_cosine_mean' in metrics

