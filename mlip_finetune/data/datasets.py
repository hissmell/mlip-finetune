"""Dataset classes for atomic structure data."""

import logging
from typing import Dict, List, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import ase.io
from ase.neighborlist import neighbor_list

logger = logging.getLogger(__name__)


class AtomicDataset(Dataset):
    """
    Dataset for atomic structures from XYZ files.
    
    Loads structures from extended XYZ format and prepares them
    for training with NequIP-style models.
    
    Args:
        file_path: Path to XYZ file
        r_max: Cutoff radius for neighbor list
        type_mapping: Optional dictionary mapping element symbols to type indices
    
    Example:
        >>> dataset = AtomicDataset("structures.xyz", r_max=6.0)
        >>> print(len(dataset))
        >>> batch = dataset[0]
    """
    
    # Elements excluded from MPtrj training (use this as default)
    EXCLUDED_Z = {84, 85, 86, 87, 88}  # Po, At, Rn, Fr, Ra
    SUPPORTED_Z = set(range(1, 95)) - EXCLUDED_Z  # Z=1-94 except excluded
    
    def __init__(
        self,
        file_path: str,
        r_max: float = 6.0,
        type_mapping: Optional[Dict[str, int]] = None
    ):
        self.file_path = file_path
        self.r_max = r_max
        self.type_mapping = type_mapping or {}
        
        # Load structures
        all_structures = ase.io.read(file_path, index=':')
        if not isinstance(all_structures, list):
            all_structures = [all_structures]
        
        # Filter unsupported elements
        self.structures = []
        skipped = 0
        for structure in all_structures:
            if set(structure.numbers).issubset(self.SUPPORTED_Z):
                self.structures.append(structure)
            else:
                skipped += 1
        
        if skipped > 0:
            logger.warning(f"Skipped {skipped} structures with unsupported elements")
        
        logger.info(f"Loaded {len(self.structures)} structures from {file_path}")
        
        # Setup type mapping
        self._setup_type_mapping()
    
    def _setup_type_mapping(self) -> None:
        """Create element to type index mapping."""
        if self.type_mapping:
            return
        
        from ase.data import chemical_symbols
        
        # Build mapping for Z=1-94 excluding Po, At, Rn, Fr, Ra
        self.type_mapping = {}
        type_idx = 0
        for z in range(1, 95):
            if z not in self.EXCLUDED_Z:
                self.type_mapping[chemical_symbols[z]] = type_idx
                type_idx += 1
        
        # Log unique species in dataset
        unique_species = set()
        for s in self.structures:
            unique_species.update(s.get_chemical_symbols())
        logger.info(f"Unique species: {sorted(unique_species)}")
    
    def __len__(self) -> int:
        return len(self.structures)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        structure = self.structures[idx]
        
        # Basic properties
        positions = torch.tensor(structure.positions, dtype=torch.float64)
        atomic_numbers = torch.tensor(structure.numbers, dtype=torch.long)
        
        # Type indices
        species = structure.get_chemical_symbols()
        atom_types = torch.tensor(
            [self.type_mapping[s] for s in species],
            dtype=torch.long
        )
        
        # Cell and PBC
        cell = torch.tensor(structure.cell.array, dtype=torch.float64)
        pbc = torch.tensor(structure.pbc, dtype=torch.bool)
        
        # Neighbor list
        i_indices, j_indices, D, S = neighbor_list(
            'ijDS', structure, self.r_max, self_interaction=False
        )
        edge_index = torch.tensor(np.stack([i_indices, j_indices]), dtype=torch.long)
        edge_cell_shift = torch.tensor(S, dtype=torch.float64)
        
        # Build data dictionary
        data = {
            'pos': positions,
            'atomic_numbers': atomic_numbers,
            'atom_types': atom_types,
            'cell': cell,
            'pbc': pbc,
            'edge_index': edge_index,
            'edge_cell_shift': edge_cell_shift,
            'num_nodes': torch.tensor(len(structure), dtype=torch.long),
        }
        
        # Labels (if available) - support both info/arrays and calculator
        energy = None
        forces = None
        
        # Try info/arrays first (older format)
        if 'energy' in structure.info:
            energy = structure.info['energy']
        elif 'Energy' in structure.info:
            energy = structure.info['Energy']
        
        if 'forces' in structure.arrays:
            forces = structure.arrays['forces']
        elif 'Forces' in structure.arrays:
            forces = structure.arrays['Forces']
        
        # Try calculator (ASE 3.26+ format)
        if structure.calc is not None:
            try:
                if energy is None:
                    energy = structure.get_potential_energy()
            except:
                pass
            try:
                if forces is None:
                    forces = structure.get_forces()
            except:
                pass
        
        if energy is not None:
            data['total_energy'] = torch.tensor(energy, dtype=torch.float64)
        
        if forces is not None:
            data['forces'] = torch.tensor(forces, dtype=torch.float64)
        
        # Stress
        stress = None
        if 'stress' in structure.info:
            stress = structure.info['stress']
        elif structure.calc is not None:
            try:
                stress = structure.get_stress()
            except:
                pass
        
        if stress is not None:
            if len(stress) == 6:  # Voigt notation
                stress = self._voigt_to_matrix(stress)
            data['stress'] = torch.tensor(stress, dtype=torch.float64)
        
        return data
    
    @staticmethod
    def _voigt_to_matrix(voigt) -> np.ndarray:
        """Convert Voigt stress to 3x3 matrix."""
        return np.array([
            [voigt[0], voigt[5], voigt[4]],
            [voigt[5], voigt[1], voigt[3]],
            [voigt[4], voigt[3], voigt[2]]
        ])


def create_dataloader(
    file_path: str,
    batch_size: int = 8,
    r_max: float = 6.0,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for atomic structure data.
    
    Args:
        file_path: Path to XYZ file
        batch_size: Batch size
        r_max: Cutoff radius for neighbor list
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        
    Returns:
        DataLoader instance
    """
    from mlip_finetune.data.collate import collate_atomic_data
    
    dataset = AtomicDataset(file_path, r_max=r_max, **kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_atomic_data,
        pin_memory=True,
    )

