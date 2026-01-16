"""NequIP key mappings.

Maps between mlip-finetune standard keys and NequIP's AtomicDataDict keys.
"""

from typing import Dict
from .standard import KEYS


def get_nequip_key_mapping() -> Dict[str, str]:
    """Get mapping from standard keys to NequIP keys.
    
    Returns:
        Dict mapping standard_key -> nequip_key
    """
    try:
        from nequip.data import AtomicDataDict
    except ImportError:
        raise ImportError("NequIP is required. Install with: pip install nequip")
    
    return {
        # Atomic Structure
        KEYS.POSITIONS: AtomicDataDict.POSITIONS_KEY,  # 'pos'
        KEYS.ATOMIC_NUMBERS: AtomicDataDict.ATOMIC_NUMBERS_KEY,  # 'atomic_numbers'
        KEYS.ATOM_TYPES: AtomicDataDict.ATOM_TYPE_KEY,  # 'atom_types'
        
        # Cell & Periodicity
        KEYS.CELL: AtomicDataDict.CELL_KEY,  # 'cell'
        KEYS.PBC: AtomicDataDict.PBC_KEY,  # 'pbc'
        
        # Graph Structure
        KEYS.EDGE_INDEX: AtomicDataDict.EDGE_INDEX_KEY,  # 'edge_index'
        KEYS.EDGE_CELL_SHIFT: AtomicDataDict.EDGE_CELL_SHIFT_KEY,  # 'edge_cell_shift'
        KEYS.EDGE_VECTORS: AtomicDataDict.EDGE_VECTORS_KEY,  # 'edge_vectors'
        KEYS.EDGE_LENGTHS: AtomicDataDict.EDGE_LENGTH_KEY,  # 'edge_lengths'
        
        # Batch Information
        KEYS.BATCH: AtomicDataDict.BATCH_KEY,  # 'batch'
        KEYS.NUM_ATOMS: AtomicDataDict.NUM_NODES_KEY,  # 'num_atoms' (NequIP uses NUM_NODES)
        KEYS.NUM_NODES: AtomicDataDict.NUM_NODES_KEY,  # 'num_atoms'
        
        # Labels
        KEYS.TOTAL_ENERGY: AtomicDataDict.TOTAL_ENERGY_KEY,  # 'total_energy'
        KEYS.FORCES: AtomicDataDict.FORCE_KEY,  # 'forces'
        KEYS.STRESS: AtomicDataDict.STRESS_KEY,  # 'stress'
        KEYS.VIRIAL: AtomicDataDict.VIRIAL_KEY,  # 'virial'
        
        # Predictions
        KEYS.ATOMIC_ENERGY: AtomicDataDict.PER_ATOM_ENERGY_KEY,  # 'atomic_energy'
    }


def get_nequip_reverse_mapping() -> Dict[str, str]:
    """Get mapping from NequIP keys to standard keys.
    
    Returns:
        Dict mapping nequip_key -> standard_key
    """
    forward = get_nequip_key_mapping()
    return {v: k for k, v in forward.items()}


# Lazy-loaded mappings
_NEQUIP_TO_STANDARD = None
_STANDARD_TO_NEQUIP = None


def nequip_to_standard(key: str) -> str:
    """Convert NequIP key to standard key."""
    global _NEQUIP_TO_STANDARD
    if _NEQUIP_TO_STANDARD is None:
        _NEQUIP_TO_STANDARD = get_nequip_reverse_mapping()
    return _NEQUIP_TO_STANDARD.get(key, key)


def standard_to_nequip(key: str) -> str:
    """Convert standard key to NequIP key."""
    global _STANDARD_TO_NEQUIP
    if _STANDARD_TO_NEQUIP is None:
        _STANDARD_TO_NEQUIP = get_nequip_key_mapping()
    return _STANDARD_TO_NEQUIP.get(key, key)
