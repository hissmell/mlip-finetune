"""Standard key definitions for mlip-finetune.

This module defines the internal standard keys used throughout the framework.
All MLIP backends are mapped to/from these standard keys.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StandardKeys:
    """Standard property keys used internally by mlip-finetune.
    
    These keys are the 'lingua franca' of the framework.
    All backend-specific keys are converted to/from these.
    """
    
    # === Atomic Structure ===
    POSITIONS: str = "pos"
    ATOMIC_NUMBERS: str = "atomic_numbers"
    ATOM_TYPES: str = "atom_types"  # Integer type indices
    SPECIES: str = "species"  # String species names
    
    # === Cell & Periodicity ===
    CELL: str = "cell"
    PBC: str = "pbc"
    
    # === Graph Structure ===
    EDGE_INDEX: str = "edge_index"
    EDGE_CELL_SHIFT: str = "edge_cell_shift"
    EDGE_VECTORS: str = "edge_vectors"
    EDGE_LENGTHS: str = "edge_lengths"
    
    # === Batch Information ===
    BATCH: str = "batch"
    NUM_ATOMS: str = "num_atoms"
    NUM_NODES: str = "num_nodes"  # Alias for NUM_ATOMS
    PTR: str = "ptr"  # Pointer for batched graphs
    
    # === Labels (Reference Values) ===
    TOTAL_ENERGY: str = "total_energy"
    ENERGY_PER_ATOM: str = "energy_per_atom"
    FORCES: str = "forces"
    STRESS: str = "stress"  # 3x3 tensor
    STRESS_VOIGT: str = "stress_voigt"  # 6-component Voigt notation
    VIRIAL: str = "virial"
    
    # === Predictions ===
    PRED_ENERGY: str = "pred_energy"
    PRED_FORCES: str = "pred_forces"
    PRED_STRESS: str = "pred_stress"
    ATOMIC_ENERGY: str = "atomic_energy"  # Per-atom energy contributions
    
    # === Embeddings ===
    NODE_FEATURES: str = "node_features"
    EDGE_FEATURES: str = "edge_features"
    NODE_ATTRS: str = "node_attrs"


# Global instance for easy access
KEYS = StandardKeys()


# Key categories for batch processing
PER_STRUCTURE_KEYS = [
    KEYS.TOTAL_ENERGY,
    KEYS.ENERGY_PER_ATOM,
    KEYS.STRESS,
    KEYS.STRESS_VOIGT,
    KEYS.VIRIAL,
    KEYS.CELL,
    KEYS.PBC,
    KEYS.NUM_ATOMS,
]

PER_ATOM_KEYS = [
    KEYS.POSITIONS,
    KEYS.ATOMIC_NUMBERS,
    KEYS.ATOM_TYPES,
    KEYS.FORCES,
    KEYS.ATOMIC_ENERGY,
    KEYS.NODE_FEATURES,
]

PER_EDGE_KEYS = [
    KEYS.EDGE_CELL_SHIFT,
    KEYS.EDGE_VECTORS,
    KEYS.EDGE_LENGTHS,
    KEYS.EDGE_FEATURES,
]
