"""
MLIP-Finetune: A flexible framework for fine-tuning Machine Learning Interatomic Potentials.

This library provides tools and strategies for efficiently fine-tuning pre-trained
MLIP models (NequIP, MACE, etc.) on new datasets while preserving previously learned knowledge.

Main Components:
    - trainers: Training loops and orchestration
    - strategies: Fine-tuning strategies (Naive, EWC, PNC, Replay, etc.)
    - models: MLIP backend wrappers (NequIP, MACE)
    - data: Dataset loading and preprocessing
    - callbacks: Logging, checkpointing, early stopping
    - metrics: Evaluation metrics for energy, forces, stress
    - configs: Configuration management

Example:
    >>> from mlip_finetune import Trainer, NaiveStrategy
    >>> from mlip_finetune.configs import load_config
    >>>
    >>> config = load_config("config.yaml")
    >>> trainer = Trainer(config)
    >>> trainer.fit()
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Core imports
from mlip_finetune.trainers import Trainer
from mlip_finetune.strategies import (
    BaseStrategy,
    NaiveStrategy,
    EWCStrategy,
    # PNCStrategy,
    # ReplayStrategy,
)
from mlip_finetune.models import NequIPWrapper
from mlip_finetune.configs import load_config

__all__ = [
    # Version
    "__version__",
    # Trainers
    "Trainer",
    # Strategies
    "BaseStrategy",
    "NaiveStrategy", 
    "EWCStrategy",
    # Models
    "NequIPWrapper",
    # Config
    "load_config",
]

