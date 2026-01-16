"""MLIP model wrappers for different backends."""

from mlip_finetune.models.nequip_wrapper import NequIPWrapper
from mlip_finetune.models.base import BaseModelWrapper

__all__ = [
    "BaseModelWrapper",
    "NequIPWrapper",
]


def get_model_wrapper(architecture: str):
    """Get model wrapper class by architecture name."""
    wrappers = {
        "nequip": NequIPWrapper,
    }
    
    if architecture.lower() not in wrappers:
        available = ", ".join(wrappers.keys())
        raise ValueError(f"Unknown architecture: '{architecture}'. Available: {available}")
    
    return wrappers[architecture.lower()]

