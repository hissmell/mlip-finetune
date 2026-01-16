"""NequIP model wrapper for fine-tuning."""

import logging
from typing import Dict, Any

import torch
import torch.nn as nn

from mlip_finetune.models.base import BaseModelWrapper

logger = logging.getLogger(__name__)


class NequIPWrapper(BaseModelWrapper):
    """
    Wrapper for NequIP models.
    
    Supports loading from:
        - .nequip.zip package files (recommended)
        - .pt checkpoint files
    
    Example:
        >>> config = {
        ...     'package_path': 'model.nequip.zip',
        ...     'r_max': 6.0
        ... }
        >>> wrapper = NequIPWrapper(config)
        >>> output = wrapper(batch)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Load model from package or checkpoint
        if 'package_path' in config and config['package_path']:
            self.load_from_package(
                config['package_path'],
                config.get('compile_mode', 'eager')
            )
        elif 'pretrained_path' in config and config['pretrained_path']:
            self.load_pretrained(config['pretrained_path'])
        else:
            raise ValueError(
                "Must provide either 'package_path' (for .nequip.zip) "
                "or 'pretrained_path' (for .pt checkpoints)"
            )
    
    def load_from_package(self, package_path: str, compile_mode: str = 'eager') -> None:
        """Load NequIP model from .nequip.zip package."""
        from nequip.model import ModelFromPackage
        
        if not package_path.endswith('.nequip.zip'):
            raise ValueError(f"Package must have .nequip.zip extension: {package_path}")
        
        model_dict = ModelFromPackage(
            package_path=package_path,
            compile_mode=compile_mode
        )
        
        # Extract model from ModuleDict
        if hasattr(model_dict, 'keys') and 'sole_model' in model_dict:
            self.model = model_dict['sole_model']
        else:
            self.model = model_dict
        
        # Enable training
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()
        
        logger.info(f"Loaded NequIP model from: {package_path}")
        logger.info(f"Parameters: {self.get_num_parameters():,}")
    
    def load_pretrained(self, path: str) -> None:
        """Load from checkpoint file."""
        if path.endswith('.nequip.zip'):
            self.load_from_package(path)
            return
        
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        if 'model' in checkpoint:
            self.model = checkpoint['model']
        elif 'state_dict' in checkpoint:
            if self.model is None:
                raise ValueError(
                    "Cannot load state_dict without initialized model. "
                    "Use .nequip.zip package file instead."
                )
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model = checkpoint
        
        logger.info(f"Loaded model from: {path}")
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through NequIP model."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        from nequip.data import AtomicDataDict
        
        # Convert batch to NequIP format
        nequip_batch = self._prepare_nequip_batch(batch)
        
        output = self.model(nequip_batch)
        
        # Map NequIP output keys to standard keys
        result = {}
        key_mapping = {
            AtomicDataDict.TOTAL_ENERGY_KEY: 'energy',
            AtomicDataDict.PER_ATOM_ENERGY_KEY: 'atomic_energy',
            AtomicDataDict.FORCE_KEY: 'forces',
            AtomicDataDict.STRESS_KEY: 'stress',
            AtomicDataDict.VIRIAL_KEY: 'virial',
        }
        
        for nequip_key, standard_key in key_mapping.items():
            if nequip_key in output:
                result[standard_key] = output[nequip_key]
        
        return result
    
    def _prepare_nequip_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert batch from our format to NequIP AtomicDataDict format."""
        from mlip_finetune.keys import to_backend
        
        # Use the key registry to convert standard keys to NequIP keys
        nequip_batch = to_backend(batch, 'nequip')
        
        return nequip_batch
    
    def parameters(self):
        """Get model parameters."""
        if self.model is None:
            return iter([])
        return self.model.parameters()
    
    def named_parameters(self):
        """Get named model parameters."""
        if self.model is None:
            return iter([])
        return self.model.named_parameters()
    
    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)
        if self.model is not None:
            self.model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        super().eval()
        if self.model is not None:
            self.model.eval()
        return self

