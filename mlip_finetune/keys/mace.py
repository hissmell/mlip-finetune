"""MACE key mappings.

Maps between mlip-finetune standard keys and MACE's expected keys.
"""

from typing import Dict
from .standard import KEYS


def get_mace_key_mapping() -> Dict[str, str]:
    """Get mapping from standard keys to MACE keys.
    
    MACE uses slightly different conventions than NequIP.
    
    Returns:
        Dict mapping standard_key -> mace_key
    """
    return {
        # Atomic Structure
        KEYS.POSITIONS: "positions",  # MACE uses 'positions' not 'pos'
        KEYS.ATOMIC_NUMBERS: "atomic_numbers",
        KEYS.ATOM_TYPES: "node_attrs",  # MACE uses one-hot encoding
        
        # Cell & Periodicity  
        KEYS.CELL: "cell",
        KEYS.PBC: "pbc",
        
        # Graph Structure
        KEYS.EDGE_INDEX: "edge_index",
        KEYS.EDGE_CELL_SHIFT: "shifts",  # MACE calls this 'shifts'
        KEYS.EDGE_VECTORS: "vectors",
        KEYS.EDGE_LENGTHS: "lengths",
        
        # Batch Information
        KEYS.BATCH: "batch",
        KEYS.NUM_ATOMS: "ptr",  # MACE uses pointer for batching
        KEYS.PTR: "ptr",
        
        # Labels
        KEYS.TOTAL_ENERGY: "energy",  # MACE uses 'energy' not 'total_energy'
        KEYS.FORCES: "forces",
        KEYS.STRESS: "stress",
        KEYS.VIRIAL: "virials",  # Note: plural
        
        # Predictions
        KEYS.PRED_ENERGY: "energy",
        KEYS.PRED_FORCES: "forces",
        KEYS.ATOMIC_ENERGY: "node_energy",  # MACE convention
        
        # Embeddings
        KEYS.NODE_FEATURES: "node_feats",
        KEYS.NODE_ATTRS: "node_attrs",
    }


def get_mace_reverse_mapping() -> Dict[str, str]:
    """Get mapping from MACE keys to standard keys."""
    forward = get_mace_key_mapping()
    return {v: k for k, v in forward.items()}


_MACE_TO_STANDARD = None
_STANDARD_TO_MACE = None


def mace_to_standard(key: str) -> str:
    """Convert MACE key to standard key."""
    global _MACE_TO_STANDARD
    if _MACE_TO_STANDARD is None:
        _MACE_TO_STANDARD = get_mace_reverse_mapping()
    return _MACE_TO_STANDARD.get(key, key)


def standard_to_mace(key: str) -> str:
    """Convert standard key to MACE key."""
    global _STANDARD_TO_MACE
    if _STANDARD_TO_MACE is None:
        _STANDARD_TO_MACE = get_mace_key_mapping()
    return _STANDARD_TO_MACE.get(key, key)
