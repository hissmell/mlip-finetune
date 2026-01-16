"""
Fine-tuning strategies for MLIP models.

Available Strategies:
    - NaiveStrategy: Standard fine-tuning without any regularization
    - EWCStrategy: Elastic Weight Consolidation for continual learning
    - PNCStrategy: Plasticity-Stability balance via selective freezing
    - ReplayStrategy: Experience replay with memory buffer
"""

from mlip_finetune.strategies.base import BaseStrategy
from mlip_finetune.strategies.naive import NaiveStrategy
from mlip_finetune.strategies.ewc import EWCStrategy

__all__ = [
    "BaseStrategy",
    "NaiveStrategy",
    "EWCStrategy",
]


def get_strategy(name: str):
    """Get strategy class by name."""
    strategies = {
        "naive": NaiveStrategy,
        "ewc": EWCStrategy,
    }
    
    if name.lower() not in strategies:
        available = ", ".join(strategies.keys())
        raise ValueError(f"Unknown strategy: '{name}'. Available: {available}")
    
    return strategies[name.lower()]

