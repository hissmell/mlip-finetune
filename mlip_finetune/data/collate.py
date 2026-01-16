"""Collate functions for batching atomic data."""

from typing import List, Dict
import torch


def collate_atomic_data(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for NequIP-style data format.
    
    Handles proper batching of variable-size atomic structures:
    - Per-structure properties are stacked
    - Per-atom properties are concatenated with batch indices
    - Edge indices are offset appropriately
    
    Args:
        batch: List of data dictionaries from AtomicDataset
        
    Returns:
        Batched data dictionary
    """
    # Keys to stack (per-structure properties)
    keys_to_stack = ['total_energy', 'stress', 'virial', 'cell', 'pbc']
    
    # Keys to concatenate (per-atom properties)  
    keys_to_cat = ['pos', 'atomic_numbers', 'atom_types', 'forces']
    
    batch_dict = {}
    
    # Calculate cumulative atom counts
    n_atoms_cumsum = [0]
    for data in batch:
        n_atoms_cumsum.append(n_atoms_cumsum[-1] + data['num_nodes'].item())
    
    # Create batch index for each atom
    batch_dict['batch'] = torch.cat([
        torch.full((n_atoms_cumsum[i+1] - n_atoms_cumsum[i],), i, dtype=torch.long)
        for i in range(len(batch))
    ])
    
    # Stack per-structure properties (only if ALL items have the key)
    for key in keys_to_stack:
        if all(key in data for data in batch):
            batch_dict[key] = torch.stack([data[key] for data in batch])
    
    # Concatenate per-atom properties (only if ALL items have the key)
    for key in keys_to_cat:
        if all(key in data for data in batch):
            batch_dict[key] = torch.cat([data[key] for data in batch], dim=0)
    
    # Handle edge indices with offset adjustment
    edge_indices = []
    edge_cell_shifts = []
    
    for i, data in enumerate(batch):
        edge_index = data['edge_index'] + n_atoms_cumsum[i]
        edge_indices.append(edge_index)
        edge_cell_shifts.append(data['edge_cell_shift'])
    
    if edge_indices:
        batch_dict['edge_index'] = torch.cat(edge_indices, dim=1)
        batch_dict['edge_cell_shift'] = torch.cat(edge_cell_shifts, dim=0)
    
    # Store number of atoms per structure
    batch_dict['num_nodes'] = torch.tensor(
        [data['num_nodes'].item() for data in batch],
        dtype=torch.long
    )
    
    return batch_dict

