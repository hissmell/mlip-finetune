"""Tests for fine-tuning strategies."""

import pytest
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.output = nn.Linear(5, 1)
    
    def forward(self, x):
        return self.output(torch.relu(self.linear(x)))


class TestNaiveStrategy:
    """Tests for NaiveStrategy."""
    
    def test_init(self):
        from mlip_finetune.strategies import NaiveStrategy
        
        model = SimpleModel()
        config = {'loss_coeffs': {'energy': 1.0, 'force': 100.0}}
        
        strategy = NaiveStrategy(model, config)
        
        assert strategy.model is model
        assert strategy.loss_coeffs['energy'] == 1.0
        assert strategy.loss_coeffs['force'] == 100.0
    
    def test_get_trainable_parameters(self):
        from mlip_finetune.strategies import NaiveStrategy
        
        model = SimpleModel()
        config = {}
        
        strategy = NaiveStrategy(model, config)
        params = strategy.get_trainable_parameters()
        
        assert len(params) == len(list(model.parameters()))


class TestEWCStrategy:
    """Tests for EWCStrategy."""
    
    def test_init(self):
        from mlip_finetune.strategies import EWCStrategy
        
        model = SimpleModel()
        config = {'ewc_lambda': 500.0}
        
        strategy = EWCStrategy(model, config)
        
        assert strategy.ewc_lambda == 500.0
        assert len(strategy.fisher_dict) == 0
        assert len(strategy.optimal_params) == 0
    
    def test_ewc_penalty_no_fisher(self):
        from mlip_finetune.strategies import EWCStrategy
        
        model = SimpleModel()
        config = {'ewc_lambda': 1000.0}
        
        strategy = EWCStrategy(model, config)
        
        # Without Fisher, penalty should be 0
        penalty = strategy._compute_ewc_penalty()
        assert penalty.item() == 0.0

