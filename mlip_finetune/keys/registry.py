"""Key Registry for managing backend-specific key mappings.

This module provides a unified interface for converting between
mlip-finetune's standard keys and backend-specific keys.
"""

from typing import Dict, Any, Callable, Optional
import logging

from .standard import KEYS

logger = logging.getLogger(__name__)


class KeyRegistry:
    """Registry for managing key mappings across different MLIP backends.
    
    Example:
        >>> registry = KeyRegistry()
        >>> registry.register_backend('nequip', nequip_mapping)
        >>> 
        >>> # Convert batch to NequIP format
        >>> nequip_batch = registry.convert(batch, to_backend='nequip')
        >>> 
        >>> # Convert NequIP output back to standard format
        >>> standard_output = registry.convert(output, from_backend='nequip')
    """
    
    def __init__(self):
        self._backends: Dict[str, Dict[str, str]] = {}
        self._reverse_mappings: Dict[str, Dict[str, str]] = {}
        
        # Auto-register available backends
        self._auto_register()
    
    def _auto_register(self):
        """Auto-register available MLIP backends."""
        # NequIP
        try:
            from .nequip import get_nequip_key_mapping
            self.register_backend('nequip', get_nequip_key_mapping())
        except ImportError:
            pass
        
        # MACE
        try:
            from .mace import get_mace_key_mapping
            self.register_backend('mace', get_mace_key_mapping())
        except ImportError:
            pass
    
    def register_backend(self, name: str, mapping: Dict[str, str]):
        """Register a new backend with its key mapping.
        
        Args:
            name: Backend name (e.g., 'nequip', 'mace')
            mapping: Dict mapping standard_key -> backend_key
        """
        self._backends[name] = mapping
        self._reverse_mappings[name] = {v: k for k, v in mapping.items()}
        logger.debug(f"Registered backend '{name}' with {len(mapping)} key mappings")
    
    def get_backends(self) -> list:
        """Get list of registered backend names."""
        return list(self._backends.keys())
    
    def get_mapping(self, backend: str) -> Dict[str, str]:
        """Get mapping for a specific backend (standard -> backend)."""
        if backend not in self._backends:
            raise ValueError(f"Unknown backend: {backend}. Available: {self.get_backends()}")
        return self._backends[backend]
    
    def get_reverse_mapping(self, backend: str) -> Dict[str, str]:
        """Get reverse mapping for a specific backend (backend -> standard)."""
        if backend not in self._reverse_mappings:
            raise ValueError(f"Unknown backend: {backend}. Available: {self.get_backends()}")
        return self._reverse_mappings[backend]
    
    def to_backend(self, key: str, backend: str) -> str:
        """Convert a standard key to backend-specific key.
        
        Args:
            key: Standard key
            backend: Target backend name
            
        Returns:
            Backend-specific key (or original if no mapping exists)
        """
        mapping = self.get_mapping(backend)
        return mapping.get(key, key)
    
    def from_backend(self, key: str, backend: str) -> str:
        """Convert a backend-specific key to standard key.
        
        Args:
            key: Backend-specific key
            backend: Source backend name
            
        Returns:
            Standard key (or original if no mapping exists)
        """
        reverse = self.get_reverse_mapping(backend)
        return reverse.get(key, key)
    
    def convert_dict(
        self, 
        data: Dict[str, Any],
        to_backend: Optional[str] = None,
        from_backend: Optional[str] = None,
        inplace: bool = False,
    ) -> Dict[str, Any]:
        """Convert all keys in a dictionary.
        
        Args:
            data: Dictionary with keys to convert
            to_backend: Convert standard keys to this backend's keys
            from_backend: Convert this backend's keys to standard keys
            inplace: If True, modify dict in place
            
        Returns:
            Dictionary with converted keys
        """
        if to_backend and from_backend:
            raise ValueError("Specify either to_backend or from_backend, not both")
        
        if not to_backend and not from_backend:
            return data
        
        if to_backend:
            mapping = self.get_mapping(to_backend)
        else:
            mapping = self.get_reverse_mapping(from_backend)
        
        if inplace:
            # Rename keys in place
            keys_to_rename = [(k, mapping[k]) for k in list(data.keys()) if k in mapping]
            for old_key, new_key in keys_to_rename:
                data[new_key] = data.pop(old_key)
            return data
        else:
            # Create new dict with converted keys
            return {mapping.get(k, k): v for k, v in data.items()}


# Global registry instance
_registry = None


def get_registry() -> KeyRegistry:
    """Get the global key registry instance."""
    global _registry
    if _registry is None:
        _registry = KeyRegistry()
    return _registry


# Convenience functions
def to_backend(data: Dict[str, Any], backend: str) -> Dict[str, Any]:
    """Convert standard keys to backend-specific keys."""
    return get_registry().convert_dict(data, to_backend=backend)


def from_backend(data: Dict[str, Any], backend: str) -> Dict[str, Any]:
    """Convert backend-specific keys to standard keys."""
    return get_registry().convert_dict(data, from_backend=backend)
