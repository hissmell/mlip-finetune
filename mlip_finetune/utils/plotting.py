"""Plotting utilities for MLIP finetuning visualization."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Set font
plt.rcParams['font.family'] = 'Arial'

# Colorblind-friendly colors
COLORS = {
    'primary': '#0072B2',
    'secondary': '#D55E00', 
    'tertiary': '#009E73',
    'quaternary': '#F0E442',
    'quinary': '#CC79A7',
    'diagonal': '#333333',
    'edge_primary': '#005080',
    'edge_secondary': '#A04800',
    'edge_tertiary': '#007050',
}


def setup_axis_style(ax):
    """Apply standard axis styling."""
    # Spine width
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    # Tick settings
    ax.tick_params(axis='both', width=1.5, labelsize=22)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')


def create_parity_plot(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    xlabel: str = 'True Values',
    ylabel: str = 'Predicted Values',
    title: str = 'Parity Plot',
    unit: str = '',
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Create a parity plot comparing true vs predicted values.
    
    Args:
        true_values: Array of true/reference values
        pred_values: Array of predicted values
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        unit: Unit string to append to labels (e.g., 'eV', 'eV/Å')
        save_path: Path to save the figure
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Scatter plot
    ax.scatter(
        true_values, pred_values,
        color=COLORS['primary'],
        s=50,
        alpha=0.7,
        edgecolor=COLORS['edge_primary'],
        linewidth=0.5
    )
    
    # Calculate axis limits
    all_data = np.concatenate([true_values, pred_values])
    data_min, data_max = all_data.min(), all_data.max()
    margin = (data_max - data_min) * 0.05
    lims = [data_min - margin, data_max + margin]
    
    # y=x diagonal line
    ax.plot(lims, lims, '--', color=COLORS['diagonal'], linewidth=1.5)
    
    # Set equal aspect ratio
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    
    # Labels and title
    unit_str = f' ({unit})' if unit else ''
    ax.set_xlabel(f'{xlabel}{unit_str}', fontsize=24, fontweight='bold')
    ax.set_ylabel(f'{ylabel}{unit_str}', fontsize=24, fontweight='bold')
    ax.set_title(title, fontsize=26, fontweight='bold')
    
    # Apply standard styling
    setup_axis_style(ax)
    
    # Add statistics text
    mae = np.mean(np.abs(true_values - pred_values))
    rmse = np.sqrt(np.mean((true_values - pred_values) ** 2))
    textstr = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}'
    ax.text(
        0.05, 0.95, textstr,
        transform=ax.transAxes,
        fontsize=18,
        fontweight='bold',
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8)
    )
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.debug(f"Saved parity plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def create_energy_parity_plot(
    true_energies: np.ndarray,
    pred_energies: np.ndarray,
    per_atom: bool = True,
    n_atoms: Optional[np.ndarray] = None,
    title: str = 'E Parity',
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Create energy parity plot.
    
    Args:
        true_energies: True total energies
        pred_energies: Predicted total energies
        per_atom: If True, plot per-atom energies
        n_atoms: Number of atoms per structure (required if per_atom=True)
        title: Plot title
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        matplotlib Figure
    """
    if per_atom and n_atoms is not None:
        true_values = true_energies / n_atoms
        pred_values = pred_energies / n_atoms
        unit = 'eV/atom'
        xlabel = 'True E'
        ylabel = 'Predicted E'
    else:
        true_values = true_energies
        pred_values = pred_energies
        unit = 'eV'
        xlabel = 'True E'
        ylabel = 'Predicted E'
    
    return create_parity_plot(
        true_values, pred_values,
        xlabel=xlabel, ylabel=ylabel,
        title=title, unit=unit,
        save_path=save_path, show=show
    )


def create_force_parity_plot(
    true_forces: np.ndarray,
    pred_forces: np.ndarray,
    title: str = '|F| Parity',
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Create force magnitude parity plot.
    
    Args:
        true_forces: True forces (N, 3) or flattened
        pred_forces: Predicted forces (N, 3) or flattened
        title: Plot title
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        matplotlib Figure
    """
    # Reshape to (N, 3) if needed
    if true_forces.ndim == 1:
        true_forces = true_forces.reshape(-1, 3)
    if pred_forces.ndim == 1:
        pred_forces = pred_forces.reshape(-1, 3)
    
    # Compute magnitudes
    true_mag = np.linalg.norm(true_forces, axis=1)
    pred_mag = np.linalg.norm(pred_forces, axis=1)
    
    return create_parity_plot(
        true_mag, pred_mag,
        xlabel='True |F|',
        ylabel='Predicted |F|',
        title=title,
        unit='eV/Å',
        save_path=save_path,
        show=show
    )


def create_force_component_parity_plot(
    true_forces: np.ndarray,
    pred_forces: np.ndarray,
    title: str = 'Force Component Parity Plot',
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Create force component (x, y, z) parity plot.
    
    Args:
        true_forces: True forces (N, 3)
        pred_forces: Predicted forces (N, 3)
        title: Plot title
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        matplotlib Figure
    """
    # Flatten to all components
    true_flat = true_forces.flatten()
    pred_flat = pred_forces.flatten()
    
    return create_parity_plot(
        true_flat, pred_flat,
        xlabel='True Force',
        ylabel='Predicted Force',
        title=title,
        unit='eV/Å',
        save_path=save_path,
        show=show
    )
