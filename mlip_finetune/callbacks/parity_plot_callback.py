"""Callback for generating parity plots during training."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

from .base import Callback

logger = logging.getLogger(__name__)


class ParityPlotCallback(Callback):
    """
    Callback to generate and save parity plots for energy and forces.
    
    Plots are saved to:
    - experiments/{exp_name}/plots/epoch_{N}/
    - wandb (if enabled)
    
    Args:
        save_interval: Generate plots every N epochs
        plot_energy: Whether to plot energy parity
        plot_force: Whether to plot force magnitude parity
        per_atom_energy: Use per-atom energy for plotting
    """
    
    def __init__(
        self,
        save_dir: Path,
        save_interval: int = 10,
        plot_energy: bool = True,
        plot_force: bool = True,
        per_atom_energy: bool = True,
    ):
        self.save_dir = Path(save_dir) / 'plots'
        self.save_interval = save_interval
        self.plot_energy = plot_energy
        self.plot_force = plot_force
        self.per_atom_energy = per_atom_energy
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        """Generate parity plots at specified intervals."""
        if epoch % self.save_interval != 0 and epoch != trainer.config['training'].get('epochs', 100):
            return
        
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        from mlip_finetune.utils.plotting import (
            create_energy_parity_plot,
            create_force_parity_plot,
        )
        
        epoch_dir = self.save_dir / f'epoch_{epoch:04d}'
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating parity plots for epoch {epoch}...")
        
        # Collect predictions for each dataset
        datasets = [
            ('train', trainer.train_loader),
            ('valid', trainer.val_loader),
        ]
        
        if trainer.test_loader is not None:
            datasets.append(('test', trainer.test_loader))
        
        wandb_images = {}
        
        for split_name, dataloader in datasets:
            try:
                predictions = self._collect_predictions(trainer, dataloader)
                
                if predictions is None:
                    continue
                
                # Energy parity plot
                if self.plot_energy and 'true_energy' in predictions:
                    fig = create_energy_parity_plot(
                        predictions['true_energy'],
                        predictions['pred_energy'],
                        per_atom=self.per_atom_energy,
                        n_atoms=predictions.get('n_atoms'),
                        title=f'{split_name.capitalize()} E (Epoch {epoch})',
                        save_path=epoch_dir / f'{split_name}_energy_parity.png',
                    )
                    wandb_images[f'{split_name}/energy_parity'] = fig
                    plt.close(fig)
                
                # Force magnitude parity plot
                if self.plot_force and 'true_forces' in predictions:
                    fig = create_force_parity_plot(
                        predictions['true_forces'],
                        predictions['pred_forces'],
                        title=f'{split_name.capitalize()} |F| (Epoch {epoch})',
                        save_path=epoch_dir / f'{split_name}_force_parity.png',
                    )
                    wandb_images[f'{split_name}/force_parity'] = fig
                    plt.close(fig)
                
            except Exception as e:
                logger.warning(f"Failed to generate parity plot for {split_name}: {e}")
        
        # Log to wandb if available
        self._log_to_wandb(wandb_images, epoch)
        
        logger.info(f"Parity plots saved to {epoch_dir}")
    
    def _collect_predictions(
        self,
        trainer: Any,
        dataloader: torch.utils.data.DataLoader,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Collect model predictions for a dataset."""
        trainer.model.eval()
        
        all_true_energy = []
        all_pred_energy = []
        all_true_forces = []
        all_pred_forces = []
        all_n_atoms = []
        
        device = trainer.device
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                
                # Enable gradients for force computation
                if 'pos' in batch:
                    batch['pos'] = batch['pos'].clone().requires_grad_(True)
                
                # Forward pass
                with torch.enable_grad():
                    output = trainer.model(batch)
                
                # Collect energies (flatten to 1D)
                if 'total_energy' in batch and 'energy' in output:
                    true_e = batch['total_energy'].cpu().numpy().flatten()
                    pred_e = output['energy'].detach().cpu().numpy().flatten()
                    all_true_energy.append(true_e)
                    all_pred_energy.append(pred_e)
                    
                    if 'num_nodes' in batch:
                        n_atoms = batch['num_nodes'].cpu().numpy().flatten()
                        all_n_atoms.append(n_atoms)
                
                # Collect forces
                if 'forces' in batch and 'forces' in output:
                    true_f = batch['forces'].cpu().numpy()
                    pred_f = output['forces'].detach().cpu().numpy()
                    # Ensure 2D shape (N, 3)
                    if true_f.ndim == 1:
                        true_f = true_f.reshape(-1, 3)
                    if pred_f.ndim == 1:
                        pred_f = pred_f.reshape(-1, 3)
                    all_true_forces.append(true_f)
                    all_pred_forces.append(pred_f)
        
        if not all_true_energy and not all_true_forces:
            return None
        
        result = {}
        
        if all_true_energy:
            result['true_energy'] = np.concatenate(all_true_energy)
            result['pred_energy'] = np.concatenate(all_pred_energy)
            if all_n_atoms:
                result['n_atoms'] = np.concatenate(all_n_atoms)
        
        if all_true_forces:
            result['true_forces'] = np.vstack(all_true_forces)
            result['pred_forces'] = np.vstack(all_pred_forces)
        
        return result
    
    def _log_to_wandb(self, images: Dict[str, Any], epoch: int) -> None:
        """Log images to Weights & Biases."""
        try:
            import wandb
            if wandb.run is None:
                return
            
            log_dict = {}
            for key, fig in images.items():
                log_dict[key] = wandb.Image(fig)
            
            if log_dict:
                wandb.log(log_dict, step=epoch)
                
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Failed to log to wandb: {e}")
