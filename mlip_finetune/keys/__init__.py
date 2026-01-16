"""Key management for multi-backend MLIP support.

This module provides a unified interface for managing property keys
across different MLIP backends (NequIP, MACE, etc.).

Usage:
    >>> from mlip_finetune.keys import KEYS, to_backend, from_backend
    >>> 
    >>> # Use standard keys in your code
    >>> data = {KEYS.POSITIONS: positions, KEYS.FORCES: forces}
    >>> 
    >>> # Convert to NequIP format before passing to model
    >>> nequip_data = to_backend(data, 'nequip')
    >>> 
    >>> # Convert output back to standard format
    >>> standard_output = from_backend(output, 'nequip')
"""

from .standard import (
    KEYS,
    StandardKeys,
    PER_STRUCTURE_KEYS,
    PER_ATOM_KEYS,
    PER_EDGE_KEYS,
)

from .registry import (
    KeyRegistry,
    get_registry,
    to_backend,
    from_backend,
)

__all__ = [
    # Standard keys
    'KEYS',
    'StandardKeys',
    'PER_STRUCTURE_KEYS',
    'PER_ATOM_KEYS', 
    'PER_EDGE_KEYS',
    # Registry
    'KeyRegistry',
    'get_registry',
    'to_backend',
    'from_backend',
]
